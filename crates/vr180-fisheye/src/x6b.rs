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
//! Its records: a 4096-byte header (signature, request number, width, rows,
//! tile count as `u32`; the sensor's frame number, the exposure-start stamp
//! and the integration time as `u64` nanoseconds; then 24 payload lengths,
//! 24 checksums and a flag that says whether the checksums were computed),
//! then 24 tiles of 7680×320 samples, each padded to 4096 bytes. A tile is
//! a predictor stream with no tail stream: a symbol is the 10-bit sample. A
//! checksum is the CRC-32 of its tile's samples packed four to five bytes
//! (the high eight bits of each, then a byte of their low two bits).
//!
//! The signature names the predictor stream's block layout, record by
//! record (one capture may hold both):
//!
//! - `X6S8B01\0`: the paired format's, a bit width for every 32 samples.
//! - `X6S8B02\0`: per 32 samples one header byte holds two bit widths, the
//!   low four bits for the first 16 samples and the high four for the last
//!   16. Where they agree the 32 samples are one block as above; where they
//!   differ each half is a block of its own, eight lanes of two values. A
//!   block of `n` values of `b` bits is `n × b / 8` bytes: every lane's
//!   whole bytes interleaved, then the bits left over of each lane, lane
//!   after lane, low bits first.
//!
//! ## Other frames of both sensors
//!
//! A paired take may hold frames of another size ([`PAIRED_FRAMES`]), which
//! `clip.json` states with the sample format (`size`, `sensor_bits`,
//! `cfa_pattern`, `black_level`, `white_level`, and `encoded_bayer_size`
//! where the tiles are coded on a larger canvas): 5504×5504 or 3968×3968
//! samples of 10 bits, or 4912×4912 of 14 bits, every sensor value kept.
//! Their records have the paired header with 16 tiles an eye and a zero
//! where the threshold was, and 32 payload lengths.
//!
//! - `X6D5B02\0` (the 10-bit frames): a tile is a predictor stream in the
//!   block layout of `X6S8B02` and nothing else, a symbol is the sample.
//! - `X6D4B01\0` (the 14-bit frame): a tile is coded as a 30 fps paired
//!   take's, predictor stream and tail stream, on a canvas of 4928×4928 of
//!   which the frame is the first 4912 columns of the first 4912 rows.
//!
//! ## A record for each eye
//!
//! A take may store each sensor's frame as a record of its own (`layout`
//! `paired-bayer-eye-records-v2`): 5760×5760 or 6000×6000 samples at 30
//! fps, or 4608×3968 at 60 fps, of 10 bits (black 64, white 1023). A record
//! is a 4096-byte header, the frame's stripes one after the other, and
//! zeros up to the next 4096-byte boundary. The header:
//!
//! | at | type | |
//! |---|---|---|
//! | 0 | 8 bytes | signature |
//! | 8 | `u32` | 2 |
//! | 12 | `u32` | length of the record, header and padding included |
//! | 16 | `u32` | length of the stripes |
//! | 20 | `u32` | the sensor, 0 or 1 (a paired take's channel) |
//! | 24 | `u32` | the camera's request number for that sensor |
//! | 28 | `u32` | the record's place in the file, rising |
//! | 32, 36 | `u32` | the frame's width and height |
//! | 40 | `u32` | the width it is coded at |
//! | 44 | `u32` | stripes |
//! | 48, 52, 56 | `u32` | the quantiser: its step, the value up to which and the value from which every value is kept |
//! | 60 | `u32` | CRC-32C of the stripes |
//! | 80, 84 | `u32` | zero (the predictor and the block coder in use) |
//! | 88 | `u32` | the sensor's readout mode |
//! | 96 | `u64` | the sensor's frame number |
//! | 104 | `u64` | exposure start, nanoseconds |
//! | 112 | `u64` | integration time, nanoseconds |
//! | 128 | `u32` × stripes | the length of each stripe |
//!
//! A stripe is 32 rows of the coded width: a `u32` length (the header's,
//! again), then a predictor stream in the block layout of `X6S8B02`, its
//! prediction starting anew as a tile's does. A frame of 6000×6000 is coded
//! 6016 wide in 188 stripes, zeros beyond the frame.
//!
//! The signature says how the sensor was read and how the frame is stored:
//!
//! - `X6F5O02\0`: 5760×5760, the sensor's whole view scaled on the sensor;
//!   every value kept.
//! - `X6F6Q02\0`: 6000×6000 of the same view, stored to within one code
//!   value.
//! - `X6C5O02\0`: 5760×5760, the middle of the sensor at its own pixels;
//!   every value kept.
//! - `X6C6Q02\0`: 6000×6000 of the same, to within one code value.
//! - `X6C9Q02\0`: 4608×3968 of the same at 60 fps, to within one code
//!   value.
//!
//! "To within one code value": a symbol is the value itself up to 84,
//! `84 + (value − 84 + 1) / 2` up to 895 and `value − 406` from 896, and
//! stands for the value, for `2 × symbol − 84` or for `symbol + 406`. A
//! stripe's prediction runs on the symbols.
//!
//! The two sensors' records are not written in step, and their request
//! numbers do not agree: a frame of one belongs with the frame of the other
//! whose exposure started within a millisecond of its own, and a frame with
//! no such partner is in no pair. A pair goes by the first sensor's frame
//! number, which is what `seal.json`'s `request_window` and the first
//! column of `exposure.csv` count in.

use crate::{Error, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Frame side of one eye of a paired take.
pub const EYE_SIDE: usize = 3840;
/// Rows of one of its tiles.
pub const TILE_ROWS: usize = 160;
/// Tiles of one eye of a 3840×3840 paired take and of a take of one lens
/// (the most a layout has: [`Layout::tiles_per_eye`]).
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
/// A record of one lens: where its header holds the tiles' checksums, and
/// the flag that says they were computed.
const CRC_AT: usize = 144;
const CRC_FLAG_AT: usize = 240;
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
pub const WHITE_BALANCE_FILE: &str = "white-balance.json";
/// The index of a folder that holds a take's frames as DNG files (one line
/// per pair) in place of the capture.
pub const FRAMES_FILE: &str = "frames.jsonl";

/// A frame both sensors may record besides 3840×3840.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairedFrame {
    /// Samples a side.
    pub side: u16,
    /// Side of the canvas the tiles are coded on: the frame's columns and
    /// rows come first, padding after them.
    pub coded: u16,
    /// Bits of a sample.
    pub bits: u8,
}

impl PairedFrame {
    /// Tiles of one eye.
    pub const TILES: usize = 16;

    /// The frame rate settings this frame is recorded at.
    fn rates(self) -> &'static [u32] {
        if self.side == 3968 { &[50, 60] } else { &[30] }
    }
}

/// The frames a paired take may hold besides 3840×3840.
pub const PAIRED_FRAMES: [PairedFrame; 3] = [
    PairedFrame { side: 5504, coded: 5504, bits: 10 },
    PairedFrame { side: 4912, coded: 4928, bits: 14 },
    PairedFrame { side: 3968, coded: 3968, bits: 10 },
];

/// A frame each sensor records as a record of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EyeFrame {
    pub width: u16,
    pub height: u16,
    /// Width of the canvas the stripes are coded on: the frame's columns
    /// come first, zeros after them.
    pub coded: u16,
}

impl EyeFrame {
    /// Rows of one stripe.
    pub const STRIPE_ROWS: usize = 32;

    /// Stripes of a frame: its rows, the last stripe filled up with zeros.
    pub fn stripes(self) -> usize {
        (self.height as usize).div_ceil(Self::STRIPE_ROWS)
    }
}

/// The frames a take with a record for each eye may hold.
pub const EYE_FRAMES: [EyeFrame; 3] = [
    EyeFrame { width: 5760, height: 5760, coded: 5760 },
    EyeFrame { width: 6000, height: 6000, coded: 6016 },
    EyeFrame { width: 4608, height: 3968, coded: 4608 },
];

/// How a capture holds its frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Both sensors: a 3840×3840 frame of each per record, 14 bits.
    Paired,
    /// One sensor (`channel`: the channel a paired take stores it as): a
    /// 7680×7680 frame per record, 10 bits.
    Single { channel: usize },
    /// Both sensors, a frame of another size of each per record.
    PairedOf(PairedFrame),
    /// Both sensors, each one's frame a record of its own, 10 bits. A pair
    /// is two records ([`PairEntry::other`]).
    EyeRecords(EyeFrame),
}

impl Layout {
    /// The `layout` of `clip.json`.
    fn name(self) -> &'static str {
        match self {
            Self::Paired | Self::PairedOf(_) => "paired-bayer-v1",
            Self::Single { .. } => "single-bayer-v1",
            Self::EyeRecords(_) => "paired-bayer-eye-records-v2",
        }
    }

    /// The layout a `clip.json` states (`eye` names a single layout's
    /// sensor, `size` a paired layout's frame: one this module does not
    /// know reads as the 3840×3840 layout, or as a frame of that size with
    /// a record for each eye, and [`read_take`] then names it).
    fn of_clip(clip: &serde_json::Value) -> Option<Self> {
        let size = |k: &str| -> Option<(u64, u64)> {
            let a = clip.get(k)?.as_array()?;
            Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?))
        };
        match clip.get("layout").and_then(|l| l.as_str())? {
            "paired-bayer-v1" => {
                let frame = size("size").and_then(|(w, h)| PAIRED_FRAMES.into_iter().find(|f| (w, h) == (f.side as u64, f.side as u64)));
                Some(frame.map_or(Self::Paired, Self::PairedOf))
            }
            "single-bayer-v1" => match clip.get("eye").and_then(|e| e.as_str())? {
                "CAM0" => Some(Self::Single { channel: 0 }),
                "CAM1" => Some(Self::Single { channel: 1 }),
                _ => None,
            },
            "paired-bayer-eye-records-v2" => {
                let (w, h) = size("size")?;
                let own = |v: u64| v.min(u16::MAX as u64) as u16;
                Some(Self::EyeRecords(EYE_FRAMES.into_iter().find(|f| (w, h) == (f.width as u64, f.height as u64))
                    .unwrap_or(EyeFrame { width: own(w), height: own(h), coded: own(w) })))
            }
            _ => None,
        }
    }

    /// Channels a pair holds.
    pub fn channels(self) -> usize {
        match self {
            Self::Paired | Self::PairedOf(_) | Self::EyeRecords(_) => 2,
            Self::Single { .. } => 1,
        }
    }

    /// The one sensor a take of one lens holds, as a paired take's channel.
    pub fn single_channel(self) -> Option<usize> {
        match self {
            Self::Paired | Self::PairedOf(_) | Self::EyeRecords(_) => None,
            Self::Single { channel } => Some(channel),
        }
    }

    /// Frame width of one channel.
    pub fn width(self) -> usize {
        match self {
            Self::Paired => EYE_SIDE,
            Self::Single { .. } => SINGLE_SIDE,
            Self::PairedOf(frame) => frame.side as usize,
            Self::EyeRecords(frame) => frame.width as usize,
        }
    }

    /// Frame height of one channel.
    pub fn height(self) -> usize {
        match self {
            Self::EyeRecords(frame) => frame.height as usize,
            _ => self.width(),
        }
    }

    /// Width of the canvas a channel's tiles are coded on: the frame's own,
    /// or a wider one the frame is the first columns of (and, where the
    /// tiles hold more rows than the frame has, the first rows).
    pub fn coded_width(self) -> usize {
        match self {
            Self::PairedOf(frame) => frame.coded as usize,
            Self::EyeRecords(frame) => frame.coded as usize,
            _ => self.width(),
        }
    }

    /// Bits of a sample.
    pub fn bits(self) -> u8 {
        match self {
            Self::Paired => 14,
            Self::Single { .. } | Self::EyeRecords(_) => 10,
            Self::PairedOf(frame) => frame.bits,
        }
    }

    /// Tiles of one channel (the stripes of a frame that is a record of its
    /// own).
    pub fn tiles_per_eye(self) -> usize {
        match self {
            Self::PairedOf(_) => PairedFrame::TILES,
            Self::EyeRecords(frame) => frame.stripes(),
            _ => TILES_PER_EYE,
        }
    }

    /// Rows of one tile.
    pub fn tile_rows(self) -> usize {
        match self {
            Self::EyeRecords(_) => EyeFrame::STRIPE_ROWS,
            // A square canvas, cut into its tiles.
            _ => self.coded_width() / self.tiles_per_eye(),
        }
    }

    /// Samples of one tile as it is coded.
    pub fn tile_samples(self) -> usize {
        self.coded_width() * self.tile_rows()
    }

    /// Samples of one channel.
    pub fn eye_samples(self) -> usize {
        self.width() * self.height()
    }

    /// Samples of one pair: its channels one after the other.
    pub fn samples(self) -> usize {
        self.channels() * self.eye_samples()
    }

    /// Tiles of one pair.
    pub fn tiles(self) -> usize {
        self.channels() * self.tiles_per_eye()
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
    /// The same samples in the later block layout: a bit width for every 16.
    Single10Split,
    /// Both sensors at another frame size, 10 bits: every sensor value, in
    /// the block layout of [`Self::Single10Split`].
    Frame10,
    /// Both sensors at another frame size, 14 bits: every sensor value,
    /// coded as [`Self::Exact30`].
    Frame14,
    /// A record for each eye, the sensor's whole view scaled on the sensor
    /// to 5760×5760: every sensor value kept.
    EyeScaled,
    /// The same view at 6000×6000, stored to within one code value.
    EyeScaledNear,
    /// A record for each eye, the middle of the sensor at its own pixels,
    /// 5760×5760: every sensor value kept.
    EyeNative,
    /// The same at 6000×6000, stored to within one code value.
    EyeNativeNear,
    /// The same at 4608×3968 and 60 fps, stored to within one code value.
    EyeWideNear,
}

impl Profile {
    /// A kept pair list names a profile by its place here: new ones go last.
    pub const ALL: [Profile; 14] = [
        Self::Exact30, Self::Mid50, Self::Raised50, Self::Coarse60, Self::Raised60, Self::Single10, Self::Single10Split,
        Self::Frame10, Self::Frame14,
        Self::EyeScaled, Self::EyeScaledNear, Self::EyeNative, Self::EyeNativeNear, Self::EyeWideNear,
    ];

    /// The record signature.
    pub fn signature(self) -> &'static [u8; 8] {
        match self {
            Self::Exact30 => b"L1024___",
            Self::Mid50 => b"X6M50ABC",
            Self::Raised50 => b"X6M50AB2",
            Self::Coarse60 => b"X6QBLK05",
            Self::Raised60 => b"X6QBLK0B",
            Self::Single10 => b"X6S8B01\0",
            Self::Single10Split => b"X6S8B02\0",
            Self::Frame10 => b"X6D5B02\0",
            Self::Frame14 => b"X6D4B01\0",
            Self::EyeScaled => b"X6F5O02\0",
            Self::EyeScaledNear => b"X6F6Q02\0",
            Self::EyeNative => b"X6C5O02\0",
            Self::EyeNativeNear => b"X6C6Q02\0",
            Self::EyeWideNear => b"X6C9Q02\0",
        }
    }

    /// Records of a take set to `fps` in `layout` may carry this signature.
    pub fn records(self, layout: Layout, fps: u32) -> bool {
        match (self, layout) {
            (Self::Exact30, Layout::Paired) => fps == 30,
            (Self::Mid50 | Self::Raised50, Layout::Paired) => fps == 50,
            (Self::Coarse60 | Self::Raised60, Layout::Paired) => fps == 60,
            (Self::Single10 | Self::Single10Split, Layout::Single { .. }) => fps == 30,
            (Self::Frame10, Layout::PairedOf(frame)) => frame.bits == 10 && frame.rates().contains(&fps),
            (Self::Frame14, Layout::PairedOf(frame)) => frame.bits == 14 && frame.rates().contains(&fps),
            (Self::EyeScaled | Self::EyeNative, Layout::EyeRecords(frame)) => frame == EYE_FRAMES[0] && fps == 30,
            (Self::EyeScaledNear | Self::EyeNativeNear, Layout::EyeRecords(frame)) => frame == EYE_FRAMES[1] && fps == 30,
            (Self::EyeWideNear, Layout::EyeRecords(frame)) => frame == EYE_FRAMES[2] && fps == 60,
            _ => false,
        }
    }

    /// What the header of a record for one eye states beside its signature:
    /// the sensor's readout mode, and the quantiser (its step, the value up
    /// to which and the value from which every value is kept). `None` for a
    /// profile of another layout.
    fn eye_record(self) -> Option<(u32, [u32; 3])> {
        const EXACT: [u32; 3] = [1, 0, 1023];
        const NEAR: [u32; 3] = [2, 84, 896];
        match self {
            Self::EyeScaled => Some((83, EXACT)),
            Self::EyeScaledNear => Some((83, NEAR)),
            Self::EyeNative => Some((61, EXACT)),
            Self::EyeNativeNear => Some((61, NEAR)),
            Self::EyeWideNear => Some((36, NEAR)),
            _ => None,
        }
    }

    /// The profile of a signature in a take of `format`; `None` for an
    /// unknown signature or one that belongs to another layout or frame
    /// rate.
    pub fn from_signature(signature: &[u8], format: Format) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.signature()[..] == *signature && p.records(format.layout, format.fps))
    }

    /// Records of a take in `layout` may carry this signature, at one frame
    /// rate setting or another.
    pub fn layout_is(self, layout: Layout) -> bool {
        [30, 50, 60].into_iter().any(|fps| self.records(layout, fps))
    }

    /// A tile is its predictor stream alone: a symbol is the sample.
    fn plain(self) -> bool {
        matches!(self, Self::Single10 | Self::Single10Split | Self::Frame10 | Self::EyeScaled | Self::EyeNative)
    }

    /// The predictor stream has a bit width for every 16 samples.
    fn split(self) -> bool {
        matches!(self, Self::Single10Split | Self::Frame10) || self.eye_record().is_some()
    }

    /// Widest bit width a block may use.
    fn max_width(self) -> usize {
        if matches!(self, Self::Exact30 | Self::Frame14) { 15 } else { 11 }
    }

    /// `(first value, step, count)` runs of the level table.
    fn runs(self) -> &'static [(u16, u16, u16)] {
        match self {
            Self::Exact30 => &[(0, 1, 1024), (1024, 16, 960)],
            Self::Mid50 => &[(0, 4, 64), (256, 16, 48), (1024, 64, 48), (4096, 128, 73), (13440, 32, 1), (13472, 16, 182)],
            Self::Raised50 => &[(0, 4, 64), (256, 16, 48), (1024, 32, 32), (2048, 64, 32), (4096, 128, 73), (13440, 32, 1), (13472, 16, 182)],
            Self::Coarse60 => &[(0, 8, 32), (256, 32, 24), (1024, 128, 24), (4096, 256, 16), (8192, 512, 10), (13312, 160, 1), (13472, 16, 182)],
            Self::Raised60 => &[(0, 8, 32), (256, 32, 24), (1024, 64, 16), (2048, 128, 16), (4096, 256, 16), (8192, 512, 10), (13312, 160, 1), (13472, 16, 182)],
            Self::Single10 | Self::Single10Split | Self::Frame10 | Self::EyeScaled | Self::EyeNative => &[(0, 1, 1024)],
            Self::Frame14 => &[(0, 1, 1024), (1024, 16, 960)],
            // Every value up to 84 and from 896, every other one between them.
            Self::EyeScaledNear | Self::EyeNativeNear | Self::EyeWideNear => &[(0, 1, 85), (86, 2, 405), (896, 1, 128)],
        }
    }

    /// Symbol → sensor level, in symbol order.
    pub fn levels(self) -> &'static [u16] {
        static TABLES: OnceLock<[Vec<u16>; 14]> = OnceLock::new();
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
        matches!(self, Self::Exact30 | Self::Single10 | Self::Single10Split | Self::Frame10 | Self::Frame14 | Self::EyeScaled | Self::EyeNative)
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
    /// Where each sensor's frame is a record of its own and the closing
    /// record counts them: each sensor's frames in the capture
    /// (`eye_frames`), and the frames each sensor numbered that the capture
    /// does not hold (`eye_hardware_gaps`). A frame without a partner is in
    /// no pair: a sensor has `eye_frames − pairs` of those.
    pub eye_frames: Option<[u64; 2]>,
    pub eye_gaps: Option<[u64; 2]>,
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
    /// The black and white level `clip.json` states, where they are not the
    /// ones above ([`sensor_levels`]).
    pub stated_levels: Option<(u16, u16)>,
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
    /// The record of the white balance the camera had found when the take
    /// began (`clip.json` → `white_balance_file`); a take that does not
    /// name one may still have [`WHITE_BALANCE_FILE`] beside the capture.
    pub white_balance_file: Option<String>,
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
        (1u32 << self.layout.bits()).saturating_sub(1) as u16
    }

    /// The seal's window is of the numbers the records carry (a paired
    /// take's request numbers), so every record lies inside it and a number
    /// of it without a record is a frame that was never saved.
    pub fn window_numbers_records(&self) -> bool {
        self.layout.channels() == 2
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
    let (width, height) = (layout.width(), layout.height());
    let known = match layout {
        Layout::EyeRecords(frame) => EYE_FRAMES.contains(&frame),
        _ => true,
    };
    if !known || size != [width as u64, height as u64] {
        return Err(err(format!("unsupported frame size {size:?}")));
    }
    if !Profile::ALL.iter().any(|p| p.records(layout, fps_setting)) {
        return Err(err(format!("unsupported frame rate setting {fps_setting} for frames of {width} x {height}")));
    }
    // What the take says of its frames has to be what this layout is.
    if let Layout::PairedOf(frame) = layout {
        let stated = |k: &str| clip.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<u64>>());
        if stated("encoded_bayer_size").is_some_and(|coded| coded != [frame.coded as u64, frame.coded as u64]) {
            return Err(err(format!("unsupported coded frame size {:?} for frames of {width} x {height}", stated("encoded_bayer_size").unwrap_or_default())));
        }
    }
    if matches!(layout, Layout::PairedOf(_) | Layout::EyeRecords(_))
        && clip.get("sensor_bits").and_then(|v| v.as_u64()).is_some_and(|bits| bits != layout.bits() as u64)
    {
        return Err(err(format!("unsupported sample depth for frames of {width} x {height}")));
    }
    // The sample format: a 3840×3840 paired take's is fixed, any other take states its own.
    let mut stated_levels = None;
    let (cfa, black_level, white_level) = match layout {
        Layout::Paired => (CFA, BLACK_LEVEL, WHITE_LEVEL),
        Layout::Single { .. } | Layout::PairedOf(_) | Layout::EyeRecords(_) => {
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
            let unstated = if layout.bits() == 14 { (BLACK_LEVEL, WHITE_LEVEL) } else { (SINGLE_BLACK_LEVEL, SINGLE_WHITE_LEVEL) };
            let (black, white) = (level("black_level", unstated.0)?, level("white_level", unstated.1)?);
            let top = (1u32 << layout.bits()).saturating_sub(1) as u16;
            if black >= white || white > top {
                return Err(err(format!("unsupported levels: black {black}, white {white}")));
            }
            let (black, white) = match sensor_levels(layout, (black, white)) {
                Some(own) => {
                    stated_levels = Some((black, white));
                    own
                }
                None => (black, white),
            };
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
        Layout::Paired | Layout::PairedOf(_) | Layout::EyeRecords(_) => ("request_window", "pairs"),
        Layout::Single { .. } => ("hardware_window", "frames"),
    };
    // Counts of each of the two sensors, where the record has them.
    let per_eye = |k: &str| -> Option<[u64; 2]> {
        match seal.get(k)?.as_array()?.as_slice() {
            [first, second] => Some([first.as_u64()?, second.as_u64()?]),
            _ => None,
        }
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
    let eye_frames = per_eye("eye_frames").filter(|_| matches!(layout, Layout::EyeRecords(_)));
    if eye_frames.is_some_and(|frames| frames.iter().any(|n| *n < pairs)) {
        return Err(err("seal.json is inconsistent"));
    }
    Ok(Ok(Take {
        dir: dir.to_path_buf(),
        layout,
        fps_setting,
        width: width as u32,
        height: height as u32,
        cfa,
        black_level,
        white_level,
        stated_levels,
        clock_start: clip.get("clock_start").and_then(ClockAnchor::from_json),
        audio_file: text("audio_file"),
        audio_timing: text("audio_timing"),
        sensor_metadata: text("sensor_metadata"),
        exposure_file: text("exposure_file"),
        white_balance_file: text("white_balance_file"),
        seal: Seal {
            fps_setting, request_window: (lo as u32, hi as u32), received_pairs: received, pairs, bytes,
            eye_frames, eye_gaps: per_eye("eye_hardware_gaps").filter(|_| eye_frames.is_some()),
        },
        capture_bytes: meta.len(),
    }))
}

/// The black and white level of `layout`'s samples where the levels a take
/// `states` are another layout's; `None` where the stated ones stand.
///
/// A 14-bit frame of [`PAIRED_FRAMES`] holds the sensor's 10-bit values four
/// bits up: black 64 << 4, saturation 1023 << 4. (Measured on a take of
/// 4912×4912: no sample of six frames has one of its low four bits set,
/// none lies below 976, the darkest patch of a night scene sits at 1048 and
/// saturated lights read 16368.) A take that states 64 and 15408 for such a
/// frame states the levels of the 3840×3840 layout, whose values are the
/// sensor's less 960.
pub fn sensor_levels(layout: Layout, states: (u16, u16)) -> Option<(u16, u16)> {
    match layout {
        Layout::PairedOf(frame) if frame.bits == 14 && states == (BLACK_LEVEL, WHITE_LEVEL) =>
            Some((SINGLE_BLACK_LEVEL << 4, SINGLE_WHITE_LEVEL << 4)),
        _ => None,
    }
}

/// One pair of the capture: one record (a take of one lens has one frame in
/// it), or two where each sensor's frame is a record of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairEntry {
    /// The pair's identity and order: the camera's request number, or,
    /// where each sensor's frame is a record of its own, the first sensor's
    /// frame number.
    pub request: u32,
    /// Byte offset of the record's header (channel 0's record, of two).
    pub offset: u64,
    /// Length of that record, header included.
    pub len: u64,
    pub profile: Profile,
    /// Exposure start of each channel, nanoseconds on the take's counter
    /// (a take of one lens: its one stamp, twice).
    pub exposure_ns: [u64; 2],
    /// Channel 1's record, where it is one of its own: its byte offset and
    /// its length.
    pub other: Option<(u64, u64)>,
    /// Integration time of each channel as its record states it,
    /// nanoseconds; 0 where the record states none (the layouts whose
    /// takes list it in a file of its own, [`read_exposures`]).
    pub shutter_ns: [u64; 2],
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
    /// The checksums of the tiles' samples, where the record carries them
    /// (a take of one lens).
    tile_crc: Option<[u32; TILES_PER_EYE]>,
}

impl Header {
    fn parse(h: &[u8], format: Format, at: u64) -> Result<Self> {
        if h.len() < BLOCK as usize {
            return Err(err(format!("truncated header at byte {at}")));
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let u64_at = |o: usize| u64::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3], h[o + 4], h[o + 5], h[o + 6], h[o + 7]]);
        let layout = format.layout;
        let profile = signed(h, format, at)?;
        let (side, per_eye) = (layout.width() as u32, layout.tiles_per_eye() as u32);
        // Where the fields sit: a paired record's header, or one lens's. The
        // field after the tile count is the threshold of a 3840×3840 record
        // and zero in any other paired one.
        let (fields_ok, request, exposure_ns, lens_at) = match layout {
            Layout::Paired | Layout::PairedOf(_) => (
                (u32_at(8), u32_at(16), u32_at(20), u32_at(24), u32_at(28))
                    == (BLOCK as u32, side, side, per_eye, if layout == Layout::Paired { THRESHOLD } else { 0 }),
                u32_at(12), [u64_at(32), u64_at(40)], 48,
            ),
            Layout::Single { .. } => (
                (u32_at(12), u32_at(16), u32_at(20)) == (side, side, per_eye),
                u32_at(8), [u64_at(32); 2], 48,
            ),
            // A record of one eye has a header of its own ([`EyeHeader`]).
            Layout::EyeRecords(_) => return Err(err(format!("invalid header at byte {at}"))),
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
        let tile_crc = (matches!(layout, Layout::Single { .. }) && u32_at(CRC_FLAG_AT) == 1)
            .then(|| std::array::from_fn(|i| u32_at(CRC_AT + 4 * i)));
        Ok(Self { profile, request, exposure_ns, tile_len, tiles, tile_crc })
    }

    /// Length of the record, header and padding included.
    fn record_len(&self) -> u64 {
        self.tile_at(self.tiles)
    }

    /// Where tile `t`'s payload starts, from the record's first byte.
    fn tile_at(&self, t: usize) -> u64 {
        BLOCK + self.tile_len[..t].iter().map(|&n| pad(n as u64)).sum::<u64>()
    }

    /// Both headers state tile `t` alike: the same stored length and, where
    /// both carry checksums, the same checksum.
    fn states_tile_as(&self, other: &Header, t: usize) -> bool {
        self.tile_len[t] == other.tile_len[t] && match (&self.tile_crc, &other.tile_crc) {
            (Some(a), Some(b)) => a[t] == b[t],
            _ => true,
        }
    }

    /// The stored bytes of tile `t` of the record at `entry`, into `buf`.
    fn read_tile(&self, file: &mut File, entry: &PairEntry, t: usize, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize(self.tile_len[t] as usize, 0);
        file.seek(SeekFrom::Start(entry.offset + self.tile_at(t)))?;
        file.read_exact(buf).map_err(|_| err(format!("truncated pair at byte {}", entry.offset)))?;
        Ok(())
    }
}

fn pad(n: u64) -> u64 {
    n.div_ceil(BLOCK) * BLOCK
}

/// The profile a record's signature (the first eight bytes of `h`) names in
/// a take of `format`.
fn signed(h: &[u8], format: Format, at: u64) -> Result<Profile> {
    Profile::from_signature(&h[..8], format).ok_or_else(|| {
        let shown: String = h[..8].iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect();
        err(format!("unsupported signature \"{shown}\" for frames of {} x {} at {} fps at byte {at}", format.layout.width(), format.layout.height(), format.fps))
    })
}

/// The header of a record that holds one eye's frame
/// ([`Layout::EyeRecords`]), checked.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EyeHeader {
    profile: Profile,
    /// The sensor: a paired take's channel.
    eye: usize,
    /// The record's place in the file.
    sequence: u32,
    /// The sensor's frame number.
    hardware: u64,
    exposure_ns: u64,
    /// Integration time, ns (0: the record states none).
    shutter_ns: u64,
    /// Length of the record, header and padding included.
    len: u64,
    /// Length of its stripes, each with the four bytes of its length.
    payload: usize,
    /// CRC-32C of those bytes.
    crc: u32,
    stripe_len: Vec<u32>,
}

impl EyeHeader {
    fn parse(h: &[u8], format: Format, at: u64) -> Result<Self> {
        let Layout::EyeRecords(frame) = format.layout else {
            return Err(err(format!("invalid header at byte {at}")));
        };
        if h.len() < BLOCK as usize {
            return Err(err(format!("truncated header at byte {at}")));
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let u64_at = |o: usize| u64::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3], h[o + 4], h[o + 5], h[o + 6], h[o + 7]]);
        let profile = signed(h, format, at)?;
        let Some((mode, quantiser)) = profile.eye_record() else {
            return Err(err(format!("invalid header at byte {at}")));
        };
        let stripes = frame.stripes();
        let fields_ok = u32_at(8) == 2
            && (u32_at(32), u32_at(36), u32_at(40), u32_at(44)) == (frame.width as u32, frame.height as u32, frame.coded as u32, stripes as u32)
            && [u32_at(48), u32_at(52), u32_at(56)] == quantiser
            && (u32_at(80), u32_at(84), u32_at(88)) == (0, 0, mode)
            && u32_at(20) <= 1;
        if !fields_ok {
            return Err(err(format!("invalid header at byte {at}")));
        }
        // A stripe at its largest: every block at the widest width.
        let stripe_samples = frame.coded as usize * EyeFrame::STRIPE_ROWS;
        let mut stripe_len = Vec::with_capacity(stripes);
        let mut total = 0u64;
        for i in 0..stripes {
            let n = u32_at(128 + 4 * i);
            if n == 0 || n as usize > stripe_samples * 2 {
                return Err(err(format!("invalid stripe size at byte {at}")));
            }
            total += 4 + n as u64;
            stripe_len.push(n);
        }
        let (len, payload) = (u32_at(12) as u64, u32_at(16) as u64);
        let longest = BLOCK + pad(stripes as u64 * (4 + 2 * stripe_samples as u64));
        if len % BLOCK != 0 || len < BLOCK || len > longest || payload != total || payload > len - BLOCK {
            return Err(err(format!("invalid record size at byte {at}")));
        }
        Ok(Self {
            profile, eye: u32_at(20) as usize, sequence: u32_at(28), hardware: u64_at(96),
            exposure_ns: u64_at(104), shutter_ns: u64_at(112), len, payload: payload as usize, crc: u32_at(60), stripe_len,
        })
    }
}

/// CRC-32C (Castagnoli) of `bytes`.
pub fn crc32c(bytes: &[u8]) -> u32 {
    static TABLES: OnceLock<Box<[[u32; 256]; 8]>> = OnceLock::new();
    let t = TABLES.get_or_init(|| {
        let mut t = Box::new([[0u32; 256]; 8]);
        for i in 0..256 {
            let mut c = i as u32;
            for _ in 0..8 {
                c = (c >> 1) ^ if c & 1 == 1 { 0x82f6_3b78 } else { 0 };
            }
            t[0][i] = c;
        }
        for i in 0..256 {
            for k in 1..8 {
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][(t[k - 1][i] & 0xff) as usize];
            }
        }
        t
    });
    let mut c = !0u32;
    let mut eights = bytes.chunks_exact(8);
    for b in &mut eights {
        let lo = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) ^ c;
        let hi = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        c = t[7][(lo & 0xff) as usize] ^ t[6][((lo >> 8) & 0xff) as usize] ^ t[5][((lo >> 16) & 0xff) as usize] ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xff) as usize] ^ t[2][((hi >> 8) & 0xff) as usize] ^ t[1][((hi >> 16) & 0xff) as usize] ^ t[0][(hi >> 24) as usize];
    }
    for &b in eights.remainder() {
        c = (c >> 8) ^ t[0][((c ^ b as u32) & 0xff) as usize];
    }
    !c
}

/// One eye's frame of a capture that holds a record for each: what its
/// header says of it and where the record lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EyeRecord {
    /// The sensor's frame number.
    hardware: u64,
    exposure_ns: u64,
    shutter_ns: u64,
    offset: u64,
    len: u64,
}

/// How far apart two sensors' exposures may start and still be one pair, ns.
const PAIR_WITHIN_NS: u64 = 1_000_000;

/// The pairs of two sensors' frames (channel 0's, then channel 1's, each in
/// any order): a frame of channel 0 with the frame of channel 1 whose
/// exposure started nearest to its own, where that is within
/// [`PAIR_WITHIN_NS`]. A frame with no such partner is in no pair. The
/// pairs come in the order of channel 0's frame numbers, which they go by.
fn pair_eye_records(mut eyes: [Vec<EyeRecord>; 2], profile: Profile) -> Result<Vec<PairEntry>> {
    for (channel, frames) in eyes.iter_mut().enumerate() {
        frames.sort_by_key(|f| f.hardware);
        // A sensor numbers its frames once each, and exposes them in turn.
        if let Some(w) = frames.windows(2).find(|w| w[0].hardware == w[1].hardware) {
            return Err(err(format!("sensor {channel} has frame {} twice (bytes {} and {})", w[0].hardware, w[0].offset, w[1].offset)));
        }
        if let Some(w) = frames.windows(2).find(|w| w[1].exposure_ns <= w[0].exposure_ns) {
            return Err(err(format!("sensor {channel}'s frame {} at byte {} does not start after the frame before it", w[1].hardware, w[1].offset)));
        }
    }
    let [first, second] = eyes;
    let mut taken = vec![false; second.len()];
    let mut pairs = Vec::with_capacity(first.len().min(second.len()));
    for a in &first {
        let after = second.partition_point(|b| b.exposure_ns < a.exposure_ns);
        let nearest = [after.checked_sub(1), (after < second.len()).then_some(after)].into_iter().flatten()
            .min_by_key(|&j| second[j].exposure_ns.abs_diff(a.exposure_ns));
        let Some(j) = nearest.filter(|&j| second[j].exposure_ns.abs_diff(a.exposure_ns) <= PAIR_WITHIN_NS) else { continue };
        let b = &second[j];
        if std::mem::replace(&mut taken[j], true) {
            return Err(err(format!("two frames of sensor 0 start within a millisecond of sensor 1's frame {} (byte {})", b.hardware, b.offset)));
        }
        let request = u32::try_from(a.hardware).map_err(|_| err(format!("frame number {} at byte {} is out of range", a.hardware, a.offset)))?;
        pairs.push(PairEntry {
            request, offset: a.offset, len: a.len, profile,
            exposure_ns: [a.exposure_ns, b.exposure_ns],
            other: Some((b.offset, b.len)),
            shutter_ns: [a.shutter_ns, b.shutter_ns],
        });
    }
    Ok(pairs)
}

/// [`scan_pairs`] of a capture with a record for each eye: one header per
/// record, the two sensors' frames paired by when their exposures started
/// ([`pair_eye_records`]).
///
/// Checks: every header ([`EyeHeader`]), one signature through the whole
/// capture, the records' places in the file rising, every record inside the
/// file, positive stamps; then each sensor's frame count against the seal's
/// where it states them, the number of pairs against the seal's, and the
/// pairs inside its window.
fn scan_eye_records(take: &Take, mut progress: impl FnMut(u64, u64) -> bool) -> Result<Option<Vec<PairEntry>>> {
    let mut file = File::open(take.capture_path())?;
    let size = take.capture_bytes;
    let mut eyes: [Vec<EyeRecord>; 2] = Default::default();
    let mut profile: Option<Profile> = None;
    let mut before: Option<u32> = None;
    let mut head = vec![0u8; BLOCK as usize];
    let mut at = 0u64;
    while at < size {
        if !progress(at, size) {
            return Ok(None);
        }
        file.seek(SeekFrom::Start(at))?;
        file.read_exact(&mut head).map_err(|_| err(format!("truncated header at byte {at}")))?;
        let h = EyeHeader::parse(&head, take.format(), at)?;
        if profile.replace(h.profile).is_some_and(|p| p != h.profile) {
            return Err(err(format!("the record at byte {at} is of another kind than the ones before it")));
        }
        if before.replace(h.sequence).is_some_and(|b| h.sequence <= b) {
            return Err(err(format!("the record at byte {at} is out of its place in the file")));
        }
        if h.exposure_ns == 0 {
            return Err(err(format!("frame {} of sensor {} at byte {at} has no exposure time", h.hardware, h.eye)));
        }
        if at + h.len > size {
            return Err(err(format!("truncated record at byte {at}")));
        }
        eyes[h.eye].push(EyeRecord { hardware: h.hardware, exposure_ns: h.exposure_ns, shutter_ns: h.shutter_ns, offset: at, len: h.len });
        at += h.len;
    }
    let held = [eyes[0].len() as u64, eyes[1].len() as u64];
    if take.seal.eye_frames.is_some_and(|stated| stated != held) {
        return Err(err(format!("the capture holds {} and {} frames of its two sensors, the camera closed it with {:?}", held[0], held[1], take.seal.eye_frames.unwrap_or_default())));
    }
    let Some(profile) = profile else {
        return Err(err("the capture holds no records"));
    };
    let pairs = pair_eye_records(eyes, profile)?;
    if pairs.len() as u64 != take.seal.pairs {
        return Err(err(format!("the capture holds {} pairs, the camera closed it with {}", pairs.len(), take.seal.pairs)));
    }
    let (lo, hi) = take.seal.request_window;
    if let Some(p) = pairs.iter().find(|p| p.request < lo || p.request > hi) {
        return Err(err(format!("frame {} at byte {} is outside the take's window {lo}..={hi}", p.request, p.offset)));
    }
    progress(size, size);
    Ok(Some(pairs))
}

/// Walk the capture and list its pairs in request order. Reads one header
/// per record; `progress(bytes walked, bytes total)` returning `false`
/// stops the walk (`Ok(None)`).
///
/// Checks what the camera states against what the file holds: every
/// signature against the layout and frame rate, the header fields, every
/// record inside the file, request numbers used once (and inside the
/// window, where the window is of request numbers), positive stamps, and
/// the number of pairs against the seal.
pub fn scan_pairs(take: &Take, mut progress: impl FnMut(u64, u64) -> bool) -> Result<Option<Vec<PairEntry>>> {
    if matches!(take.layout, Layout::EyeRecords(_)) {
        return scan_eye_records(take, progress);
    }
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
        pairs.push(PairEntry { request: h.request, offset: at, len, profile: h.profile, exposure_ns: h.exposure_ns, other: None, shutter_ns: [0; 2] });
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

/// Records before the last one in which the frame an unfinished record was
/// left with is looked for (the camera's own distance is twenty requests).
const LEFTOVER_REACH: usize = 32;

/// What the camera left in a record it did not finish
/// ([`leftover_in_last_record`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leftover {
    /// The earlier record whose tiles the last one still holds.
    pub of_request: u32,
    /// How many of the last record's tiles are that record's.
    pub tiles: usize,
}

/// Whether the last record of a take of one lens is one the camera did not
/// finish.
///
/// Such a recording can stop while its last frame is being taken over. The
/// record is written all the same: the top of the frame and, below it, what
/// the camera's buffer held before, which is the frame of twenty requests
/// earlier, unchanged. Neither the record nor the take's lists say so: the
/// request number and the stamp are the next ones, the checksums are of the
/// samples as they stand, and the camera's closing record counts a frame.
/// It shows in the tiles: one stored with the bytes of the same tile of an
/// earlier record was not written for this frame, since two exposures never
/// give the same samples. (Seen on four takes of five, in the last record
/// only and nowhere in 23 paired takes, whose last record is not looked
/// at.)
///
/// `pairs` in request order. Reads the header of the last record and of up
/// to [`LEFTOVER_REACH`] before it, nearest first, and a tile's stored
/// bytes only where the two headers state it alike.
pub fn leftover_in_last_record(take: &Take, pairs: &[PairEntry]) -> Result<Option<Leftover>> {
    if take.layout.channels() != 1 {
        return Ok(None);
    }
    let Some((last, before)) = pairs.split_last() else { return Ok(None) };
    let before = &before[before.len().saturating_sub(LEFTOVER_REACH)..];
    if before.is_empty() {
        return Ok(None);
    }
    let mut file = File::open(take.capture_path())?;
    let format = take.format();
    let mut head = vec![0u8; BLOCK as usize];
    let mut header_of = |file: &mut File, entry: &PairEntry| -> Result<Header> {
        file.seek(SeekFrom::Start(entry.offset))?;
        file.read_exact(&mut head).map_err(|_| err(format!("truncated header at byte {}", entry.offset)))?;
        Header::parse(&head, format, entry.offset)
    };
    let mine = header_of(&mut file, last)?;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for entry in before.iter().rev() {
        let theirs = header_of(&mut file, entry)?;
        let alike: Vec<usize> = (0..mine.tiles.min(theirs.tiles)).filter(|&t| mine.states_tile_as(&theirs, t)).collect();
        // The stored bytes settle it, and one tile is enough.
        for &t in &alike {
            mine.read_tile(&mut file, last, t, &mut a)?;
            theirs.read_tile(&mut file, entry, t, &mut b)?;
            if a == b {
                return Ok(Some(Leftover { of_request: entry.request, tiles: alike.len() }));
            }
        }
    }
    Ok(None)
}

/// Which channels of a pair to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Both,
    /// Only channel 0 or 1; the other half of the output is left untouched.
    /// (A take of one lens has its one channel whatever is asked for.)
    One(usize),
}

/// Read one pair's record whole (`entry.len` bytes) into `buf`; where each
/// channel has a record of its own, channel 0's and then channel 1's.
pub fn read_record(file: &mut File, entry: &PairEntry, buf: &mut Vec<u8>) -> Result<()> {
    let first = entry.len as usize;
    buf.resize(first + entry.other.map_or(0, |(_, len)| len as usize), 0);
    file.seek(SeekFrom::Start(entry.offset))?;
    file.read_exact(&mut buf[..first]).map_err(|_| err(format!("truncated pair at byte {}", entry.offset)))?;
    if let Some((offset, _)) = entry.other {
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buf[first..]).map_err(|_| err(format!("truncated record at byte {offset}")))?;
    }
    Ok(())
}

/// Decode `jobs` (a tile's stored bytes and the rows of the frame it holds)
/// on `threads` threads (0 = one per core, at most one per tile), and check
/// `sums` beside them: stored bytes and the CRC-32C they must have.
fn decode_tiles(jobs: Vec<(&[u8], &mut [u16])>, sums: &[(&[u8], u32)], profile: Profile, layout: Layout, threads: usize) -> Result<()> {
    let summed = |bytes: &[u8], crc: u32| -> Result<()> {
        if crc32c(bytes) == crc { Ok(()) } else { Err(err("a record's checksum is not that of its bytes: damaged")) }
    };
    let workers = match threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        n => n,
    }.clamp(1, jobs.len().max(1));
    if workers == 1 {
        for &(bytes, crc) in sums {
            summed(bytes, crc)?;
        }
        let mut canvas = Vec::new();
        for (bytes, dst) in jobs {
            decode_tile(bytes, profile, layout, dst, &mut canvas)?;
        }
        return Ok(());
    }
    let queue = std::sync::Mutex::new(jobs.into_iter());
    let failed: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for &(bytes, crc) in sums {
            let (failed, summed) = (&failed, &summed);
            scope.spawn(move || {
                if let Err(e) = summed(bytes, crc) {
                    if let Ok(mut slot) = failed.lock() { slot.get_or_insert(e); }
                }
            });
        }
        for _ in 0..workers {
            scope.spawn(|| {
                let mut canvas = Vec::new();
                loop {
                    let job = queue.lock().ok().and_then(|mut q| q.next());
                    let Some((bytes, dst)) = job else { break };
                    if let Err(e) = decode_tile(bytes, profile, layout, dst, &mut canvas) {
                        if let Ok(mut slot) = failed.lock() { slot.get_or_insert(e); }
                        break;
                    }
                }
            });
        }
    });
    match failed.into_inner().ok().flatten() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// [`decode_record`] of a pair whose channels are records of their own:
/// `record` holds channel 0's and then channel 1's.
fn decode_eye_records(record: &[u8], format: Format, channels: Channels, out: &mut [u16], threads: usize) -> Result<PairEntry> {
    let layout = format.layout;
    let first = EyeHeader::parse(record, format, 0)?;
    if (record.len() as u64) < first.len {
        return Err(err("truncated pair"));
    }
    let (first_bytes, rest) = record.split_at(first.len as usize);
    let second = EyeHeader::parse(rest, format, first.len)?;
    if (rest.len() as u64) < second.len {
        return Err(err("truncated pair"));
    }
    if (first.eye, second.eye) != (0, 1) || first.profile != second.profile {
        return Err(err("a pair's two records are not one of each sensor, of one kind"));
    }
    let request = u32::try_from(first.hardware).map_err(|_| err(format!("frame number {} is out of range", first.hardware)))?;
    let wanted = |channel: usize| !matches!(channels, Channels::One(c) if c != channel);
    // Each stripe's rows of its channel's frame: 32, and what is left of the
    // frame for a last stripe that is filled up.
    let frame_rows = layout.width() * layout.tile_rows();
    let mut jobs: Vec<(&[u8], &mut [u16])> = Vec::with_capacity(layout.tiles());
    let mut sums: Vec<(&[u8], u32)> = Vec::with_capacity(2);
    for (channel, ((h, bytes), eye)) in [(&first, first_bytes), (&second, rest)].into_iter().zip(out.chunks_mut(layout.eye_samples())).enumerate() {
        if !wanted(channel) {
            continue;
        }
        let stripes = &bytes[BLOCK as usize..BLOCK as usize + h.payload];
        sums.push((stripes, h.crc));
        let mut at = 0usize;
        for (&n, dst) in h.stripe_len.iter().zip(eye.chunks_mut(frame_rows)) {
            // A stripe states its own length, as the header does.
            if stripes[at..at + 4] != n.to_le_bytes() {
                return Err(err(format!("a stripe of frame {} of sensor {channel} is not the length its header gives", h.hardware)));
            }
            jobs.push((&stripes[at..at + 4 + n as usize], dst));
            at += 4 + n as usize;
        }
    }
    decode_tiles(jobs, &sums, first.profile, layout, threads)?;
    Ok(PairEntry {
        request, offset: 0, len: first.len, profile: first.profile,
        exposure_ns: [first.exposure_ns, second.exposure_ns],
        other: Some((first.len, second.len)),
        shutter_ns: [first.shutter_ns, second.shutter_ns],
    })
}

/// Decode a pair read by [`read_record`] into `out`, `format.layout`'s
/// samples: a paired take's channel 0 (its frame, row after row), then its
/// channel 1; a take of one lens's one channel. Tiles decode on `threads`
/// threads (0 = one per core, at most one per tile). Where a record carries
/// a checksum of its stored bytes, those are checked against it.
pub fn decode_record(record: &[u8], format: Format, channels: Channels, out: &mut [u16], threads: usize) -> Result<PairEntry> {
    let layout = format.layout;
    if out.len() != layout.samples() {
        return Err(err("output buffer is not the record's size"));
    }
    if matches!(layout, Layout::EyeRecords(_)) {
        return decode_eye_records(record, format, channels, out, threads);
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
        Channels::One(c) if layout.channels() > 1 => tile / layout.tiles_per_eye() == c,
        _ => true,
    };
    // Each tile's rows of its channel's frame: a tile's worth, and what is
    // left of the frame for the last tile of a larger canvas.
    let frame_rows = layout.width() * layout.tile_rows();
    let jobs: Vec<(&[u8], &mut [u16])> = tiles.iter()
        .zip(out.chunks_mut(layout.eye_samples()).flat_map(|eye| eye.chunks_mut(frame_rows)))
        .filter(|((i, _), _)| wanted(*i))
        .map(|((_, bytes), dst)| (*bytes, dst))
        .collect();
    let profile = h.profile;
    decode_tiles(jobs, &[], profile, layout, threads)?;
    Ok(PairEntry { request: h.request, offset: 0, len: h.record_len(), profile, exposure_ns: h.exposure_ns, other: None, shutter_ns: [0; 2] })
}

/// Decode one tile (`bytes` = its payload without padding) into `dst`, the
/// rows of the frame it holds (`layout.width()` samples each). A tile coded
/// on a larger canvas is decoded whole into `canvas` (scratch space, reused
/// from call to call) and its part of the frame copied out: the canvas's
/// first columns, and its first rows where the frame ends inside the tile.
fn decode_tile(bytes: &[u8], profile: Profile, layout: Layout, dst: &mut [u16], canvas: &mut Vec<u16>) -> Result<()> {
    let (side, coded) = (layout.width(), layout.coded_width());
    debug_assert!(dst.len() % side == 0 && dst.len() / side <= layout.tile_rows());
    if coded == side && dst.len() == layout.tile_samples() {
        return decode_coded_tile(bytes, profile, coded, dst);
    }
    canvas.resize(layout.tile_samples(), 0);
    decode_coded_tile(bytes, profile, coded, canvas)?;
    for (row, coded_row) in dst.chunks_exact_mut(side).zip(canvas.chunks_exact(coded)) {
        row.copy_from_slice(&coded_row[..side]);
    }
    Ok(())
}

/// Decode one tile as it is coded: rows of `width` samples filling `dst`.
fn decode_coded_tile(bytes: &[u8], profile: Profile, width: usize, dst: &mut [u16]) -> Result<()> {
    let main = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if main > bytes.len() - 4 {
        return Err(err("invalid predictor length"));
    }
    let levels = profile.levels();
    let (stream, tail) = bytes[4..].split_at(main);
    if profile.split() {
        unpack_split(stream, dst, width, profile.max_width(), levels.len() as i32)?;
    } else {
        unpack_symbols(stream, dst, width, profile.max_width(), levels.len() as i32)?;
    }
    // Where a symbol is the sample nothing follows the predictor stream.
    if profile.plain() {
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

// ── The later block layout of a take of one lens ───────────────────────

/// The eight lanes of a block of 32 values of `b` bits (`4 × b` bytes), as
/// [`unpack_symbols`] reads them: every lane's whole bytes interleaved
/// (byte `k` of lane `l` at `k × 8 + l`), then, for an odd width, each
/// lane's last four bits, two lanes to a byte.
#[inline(always)]
fn lanes_of_32(block: &[u8], b: usize) -> [u64; 8] {
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
    lanes
}

/// The eight lanes of a block of 16 values of `b` bits (`2 × b` bytes):
/// every lane's whole bytes interleaved as in a block of 32, then the bits
/// left over of each lane (`2 × b` mod 8 of them), lane after lane, low
/// bits first.
#[inline(always)]
fn lanes_of_16(block: &[u8], b: usize) -> [u64; 8] {
    let (whole, rest) = (b / 4, 2 * b % 8);
    let mut last = 0u64;
    for (k, &byte) in block[whole * 8..].iter().enumerate() {
        last |= (byte as u64) << (8 * k);
    }
    let rest_mask = (1u64 << rest) - 1;
    let mut lanes = [0u64; 8];
    for (lane, word) in lanes.iter_mut().enumerate() {
        let mut bits = 0u64;
        for k in 0..whole {
            bits |= (block[k * 8 + lane] as u64) << (8 * k);
        }
        *word = bits | (((last >> (lane * rest)) & rest_mask) << (8 * whole));
    }
    lanes
}

/// One block, its lanes read: `N` values of `b` bits (value `j` in lane
/// `j % 8`, `b × (j / 8)` bits up) → the symbols of `row` from column `x0`.
/// A value is a zigzag difference from `(left₂ + up₂ + 1) >> 1`, the
/// neighbours of the same Bayer site (zero outside the tile), as in
/// [`unpack_symbols`]; `up` is the row two above.
#[inline(always)]
fn restore_block<const N: usize>(lanes: [u64; 8], b: usize, x0: usize, row: &mut [u16], up: Option<&[u16]>, count: i32) -> Result<()> {
    let mask = (1u64 << b) - 1;
    for j in 0..N {
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
    Ok(())
}

/// The predictor stream of one tile in the later layout of a take of one
/// lens (rows of `width` samples) → symbols in `dst`: per 32 samples one
/// header byte, its low four bits the width of the first 16 samples and its
/// high four that of the last 16; one block of 32 where the two agree, a
/// block of 16 for each half where they differ.
fn unpack_split(stream: &[u8], dst: &mut [u16], width: usize, max_width: usize, count: i32) -> Result<()> {
    let w = width;
    let rows = dst.len() / w;
    let mut p = 0usize;
    for y in 0..rows {
        let (done, rest) = dst.split_at_mut(y * w);
        let row = &mut rest[..w];
        let up: Option<&[u16]> = (y >= 2).then(|| &done[(y - 2) * w..(y - 1) * w]);
        for x in (0..w).step_by(32) {
            let Some(&widths) = stream.get(p) else { return Err(err("predictor stream ends early")) };
            p += 1;
            let (first, last) = ((widths & 15) as usize, (widths >> 4) as usize);
            if first > max_width || last > max_width || stream.len() - p < 2 * (first + last) {
                return Err(err("invalid block length"));
            }
            if first == last {
                restore_block::<32>(lanes_of_32(&stream[p..p + 4 * first], first), first, x, row, up, count)?;
            } else {
                restore_block::<16>(lanes_of_16(&stream[p..p + 2 * first], first), first, x, row, up, count)?;
                restore_block::<16>(lanes_of_16(&stream[p + 2 * first..p + 2 * (first + last)], last), last, x + 16, row, up, count)?;
            }
            p += 2 * (first + last);
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
    if got.request != entry.request || got.exposure_ns != entry.exposure_ns {
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

/// The white balance the camera had found for each sensor when the take
/// began (channel 0, then 1): the sensor's values of a grey, green at 1
/// (what a DNG calls the as-shot neutral). From the take's white balance
/// record (the file `clip.json` names, else `white-balance.json`), which
/// gives each sensor (`eyes.CAM0`, `eyes.CAM1`) a `status`, the gains the
/// camera multiplied red, green and blue by (`rgb_gains`) and the neutral
/// that follows from them (`as_shot_neutral`: green's gain over red's, 1,
/// green's over blue's).
///
/// `None` for a sensor the record does not vouch for: no record (many
/// takes have none), another kind of record, a status that is not `ok`,
/// gains that are no numbers or lie outside 0.01…64, or a neutral that is
/// not the gains'. The record is of one moment before the first frame, not
/// of the frames one by one.
pub fn read_white_balance(take: &Take) -> [Option<[f64; 3]>; 2] {
    let name = take.white_balance_file.as_deref().unwrap_or(WHITE_BALANCE_FILE);
    let Some(v) = std::fs::read(take.dir.join(name)).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()) else {
        return [None; 2];
    };
    if !v.get("schema").and_then(|s| s.as_str()).is_some_and(|s| s.ends_with("white-balance-v1")) {
        return [None; 2];
    }
    let eye = |key: &str| -> Option<[f64; 3]> {
        let e = v.get("eyes")?.get(key)?;
        if e.get("status")?.as_str()? != "ok" {
            return None;
        }
        let three = |k: &str| -> Option<[f64; 3]> {
            match e.get(k)?.as_array()?.as_slice() {
                [r, g, b] => Some([r.as_f64()?, g.as_f64()?, b.as_f64()?]),
                _ => None,
            }
        };
        let gains = three("rgb_gains")?;
        if !gains.iter().all(|g| g.is_finite() && (0.01..=64.0).contains(g)) {
            return None;
        }
        let neutral = [gains[1] / gains[0], 1.0, gains[1] / gains[2]];
        match three("as_shot_neutral") {
            Some(stated) if stated.iter().zip(&neutral).any(|(s, n)| !s.is_finite() || (s - n).abs() > 1e-6) => None,
            _ => Some(neutral),
        }
    };
    [eye("CAM0"), eye("CAM1")]
}

/// One eye's row timing, as the take's sensor record states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeReadout {
    /// From one row's start to the next, ns.
    pub line_period_ns: f64,
    /// From the frame's first row to its last, ns.
    pub first_to_last_row_ns: f64,
}

/// Where one eye's frame sits on its sensor, as the take's sensor record
/// states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeWindow {
    /// The sensor pixel (its register coordinates) the frame's first sample
    /// starts at.
    pub origin: (f64, f64),
    /// Frame samples per sensor pixel, across and down.
    pub scale: (f64, f64),
    /// Sensor rows the frame was read from, top to bottom.
    pub rows_read: f64,
    /// From one of those rows' start to the next, ns.
    pub line_period_ns: f64,
    /// From the first of them to the last, where the record states it, ns.
    pub stated_span_ns: Option<f64>,
}

impl EyeWindow {
    /// From the frame's first row to its last, ns: the rows read, a line
    /// period apart. The record's own figure where it agrees with that
    /// (within 1 %); it has none for a frame the sensor scales, and its
    /// rows-of-the-frame figure does not count the rows the scaler took in.
    pub fn first_to_last_row_ns(&self) -> f64 {
        let rows = self.line_period_ns * (self.rows_read - 1.0);
        self.stated_span_ns.filter(|s| (s - rows).abs() <= rows * 0.01).unwrap_or(rows)
    }
}

/// The place of the take's two eyes on their sensors (channel 0, then 1)
/// from its sensor record. Per eye the record gives the sensor area read
/// (`analog_roi_register_xyxy_inclusive`), its binning (`binning_xy`), the
/// part of the binned picture kept (`digital_crop_after_binning_xywh`), the
/// size that part leaves the sensor at (`bayer_output_wh`: smaller where
/// the sensor scales it) and, where the frame is a window of that output,
/// the window (`recorded_bayer_window_xywh`). `None` when the take has no
/// such record or it does not describe the take's frame. A take of one lens
/// has its one sensor in both places.
///
/// The scale of a frame the sensor scales is its scaler's setting where the
/// record carries the sensor's registers (`scaling_registers`: `0x401`, the
/// scaler's mode, and `0x404` / `0x405`, the `M` of 16 samples out for
/// every `M` in), else what the rows give: the output holding every kept
/// row.
///
/// A scaled frame may hold less than its kept part scaled (5504 columns
/// are 7912 of a kept 7920; 6000 columns and rows scaled 16 / 21 are 7875
/// of a kept 7880). It then holds the kept part's first columns and rows:
/// the frame starts where the kept part starts. Measured on takes: the
/// sensor's fixed pattern in the dark corners of such a frame lies on that
/// of a frame whose place is not in doubt (5760×5760, which is all of its
/// kept part, itself checked against the frames at the sensor's own
/// pixels) within a fifth of a sensor pixel, in both sensors and in each
/// of the four corners, which also bears out the scaler's own scale.
pub fn read_sensor_windows(take: &Take) -> Option<[EyeWindow; 2]> {
    let name = take.sensor_metadata.as_deref()?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(name)).ok()?).ok()?;
    let eyes = v.get("eyes")?;
    let eye = |key: &str| -> Option<EyeWindow> {
        let e = eyes.get(key)?;
        let list = |k: &str| -> Option<Vec<f64>> { Some(e.get(k)?.as_array()?.iter().filter_map(|x| x.as_f64()).collect()) };
        let (roi, bin, kept, out) = (list("analog_roi_register_xyxy_inclusive")?, list("binning_xy")?, list("digital_crop_after_binning_xywh")?, list("bayer_output_wh")?);
        if roi.len() != 4 || bin.len() != 2 || kept.len() != 4 || out.len() != 2 || bin.iter().chain(&kept[2..]).chain(&out).any(|v| *v < 1.0) {
            return None;
        }
        // The frame inside the sensor's output: all of it unless a window is named.
        let window = list("recorded_bayer_window_xywh").filter(|w| w.len() == 4).unwrap_or(vec![0.0, 0.0, out[0], out[1]]);
        if (window[2], window[3]) != (take.width as f64, take.height as f64) || window[0] + window[2] > out[0] || window[1] + window[3] > out[1] {
            return None;
        }
        // Output samples per binned pixel: one, or the sensor's scale. Its
        // scaler's setting where the record has it, else what the rows give.
        let stated = e.get("scaling_registers").and_then(|r| {
            let register = |k: &str| r.get(k).and_then(|x| x.as_u64());
            let m = register("0x404").unwrap_or(0) << 8 | register("0x405")?;
            (register("0x401")? != 0 && m >= 16).then(|| 16.0 / m as f64)
        });
        let scale = stated.unwrap_or(out[1] / kept[3]);
        if !(scale > 0.0 && scale <= 1.0) || out[0] > kept[2] * scale + 0.5 || out[1] > kept[3] * scale + 0.5 {
            return None;
        }
        let line = e.get("line_period_ns")?.as_f64().filter(|l| l.is_finite() && *l > 0.0)?;
        let window = EyeWindow {
            origin: (
                roi[0] + bin[0] * (kept[0] + window[0] / scale),
                roi[1] + bin[1] * (kept[1] + window[1] / scale),
            ),
            scale: (scale / bin[0], scale / bin[1]),
            rows_read: take.height as f64 / scale,
            line_period_ns: line,
            stated_span_ns: e.get("rolling_shutter_first_to_last_row_ns").and_then(|x| x.as_f64()),
        };
        (window.first_to_last_row_ns() < 1e9).then_some(window)
    };
    match take.layout {
        Layout::Paired | Layout::PairedOf(_) | Layout::EyeRecords(_) => Some([eye("CAM0")?, eye("CAM1")?]),
        Layout::Single { channel } => eye(["CAM0", "CAM1"][channel.min(1)]).map(|e| [e, e]),
    }
}

/// The row timing of the take's two eyes (channel 0, then 1) from its
/// sensor record ([`read_sensor_windows`]): a frame's stamp is the start of
/// its first row's exposure and every row the sensor reads starts a line
/// period after the one above it. `None` when the take has no such record,
/// or the record is of another frame.
pub fn read_sensor_readout(take: &Take) -> Option<[EyeReadout; 2]> {
    if let Some(eyes) = read_sensor_windows(take) {
        return Some(eyes.map(|w| EyeReadout { line_period_ns: w.line_period_ns, first_to_last_row_ns: w.first_to_last_row_ns() }));
    }
    // A record that states the timing without the sensor's windows: of the
    // take's own frame, read row for row.
    let name = take.sensor_metadata.as_deref()?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(name)).ok()?).ok()?;
    let eyes = v.get("eyes")?;
    let eye = |key: &str| -> Option<EyeReadout> {
        let e = eyes.get(key)?;
        if e.get("analog_roi_register_xyxy_inclusive").is_some() {
            return None;
        }
        let size: Vec<u64> = e.get("bayer_output_wh")?.as_array()?.iter().filter_map(|x| x.as_u64()).collect();
        if size != [take.width as u64, take.height as u64] {
            return None;
        }
        let line = e.get("line_period_ns")?.as_f64()?;
        let rows = line * (take.height as f64 - 1.0);
        let span = e.get("rolling_shutter_first_to_last_row_ns").and_then(|x| x.as_f64())
            .filter(|s| (s - rows).abs() <= rows * 0.01).unwrap_or(rows);
        (line.is_finite() && line > 0.0 && span < 1e9).then_some(EyeReadout { line_period_ns: line, first_to_last_row_ns: span })
    };
    match take.layout {
        Layout::Paired | Layout::PairedOf(_) | Layout::EyeRecords(_) => Some([eye("CAM0")?, eye("CAM1")?]),
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

    /// A block of a predictor stream: `vals` (eight lanes, value `j` in lane
    /// `j % 8`) at `b` bits each.
    fn pack_block(vals: &[u32], b: usize) -> Vec<u8> {
        let per = vals.len() / 8;
        let (whole, rest) = (per * b / 8, per * b % 8);
        let mut block = vec![0u8; per * b];
        let mut last = 0u64;
        for lane in 0..8 {
            let mut w = 0u64;
            for q in 0..per {
                w |= (vals[lane + 8 * q] as u64) << (b * q);
            }
            for k in 0..whole {
                block[k * 8 + lane] = (w >> (8 * k)) as u8;
            }
            last |= ((w >> (8 * whole)) & ((1u64 << rest) - 1)) << (lane * rest);
        }
        for k in 0..rest {
            block[whole * 8 + k] = (last >> (8 * k)) as u8;
        }
        block
    }

    /// One tile of a take of one lens in the later block layout: rows of
    /// `w` samples.
    fn encode_tile_split(samples: &[u16], w: usize) -> Vec<u8> {
        let mut main = Vec::new();
        for y in 0..samples.len() / w {
            for x in (0..w).step_by(32) {
                let mut vals = [0u32; 32];
                for (j, v) in vals.iter_mut().enumerate() {
                    let col = x + j;
                    let i = y * w + col;
                    let left = if col >= 2 { samples[i - 2] as i32 } else { 0 };
                    let up = if y >= 2 { samples[i - 2 * w] as i32 } else { 0 };
                    let d = samples[i] as i32 - ((left + up + 1) >> 1);
                    *v = if d < 0 { (-d * 2 - 1) as u32 } else { (d * 2) as u32 };
                }
                let width = |v: &[u32]| (32 - v.iter().fold(0, |a, v| a | v).leading_zeros()) as usize;
                let (first, last) = (width(&vals[..16]), width(&vals[16..]));
                assert!(first <= 11 && last <= 11);
                main.push((first | last << 4) as u8);
                if first == last {
                    main.extend(pack_block(&vals, first));
                } else {
                    main.extend(pack_block(&vals[..16], first));
                    main.extend(pack_block(&vals[16..], last));
                }
            }
        }
        let mut out = (main.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&main);
        out
    }

    /// The checksum a record of one lens carries for a tile: the CRC-32 of
    /// its samples packed four to five bytes (the high eight bits of each,
    /// then a byte of their low two bits).
    pub fn packed_crc(tile: &[u16]) -> u32 {
        let mut c = !0u32;
        let mut feed = |byte: u8| {
            c ^= byte as u32;
            for _ in 0..8 {
                c = (c >> 1) ^ if c & 1 == 1 { 0xedb8_8320 } else { 0 };
            }
        };
        for four in tile.chunks(4) {
            let mut low = 0u8;
            for (j, v) in four.iter().enumerate() {
                feed((v >> 2) as u8);
                low |= ((v & 3) as u8) << (2 * j);
            }
            feed(low);
        }
        !c
    }

    /// One record of a take of one lens holding `samples` (7680×7680, below
    /// 1024) in `profile`'s block layout, as the camera lays it out: the
    /// sensor's frame number, the stamp and the integration time after the
    /// four `u32` fields, the payload lengths, then the checksums (left at
    /// zero: the first record alone carries any).
    pub fn encode_single_record(profile: Profile, request: u32, hardware: u64, exposure_ns: u64, duration_ns: u64, samples: &[u16]) -> Vec<u8> {
        let layout = Layout::Single { channel: 0 };
        assert!(samples.len() == layout.samples() && profile.layout_is(layout));
        let tiles: Vec<Vec<u8>> = samples.chunks(layout.tile_samples()).map(|t| match profile {
            Profile::Single10Split => encode_tile_split(t, SINGLE_SIDE),
            _ => encode_tile(profile, t, SINGLE_SIDE),
        }).collect();
        let mut out = vec![0u8; BLOCK as usize];
        out[..8].copy_from_slice(profile.signature());
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

    /// A 10-bit test picture of one lens in bands: flat (differences of
    /// zero), a ramp with noise, steps between 0 and 1023 (the widest
    /// differences), and noise on every other 16 samples (halves of unlike
    /// widths).
    pub fn banded_test_samples() -> Vec<u16> {
        let mut s = 0x9e37_79b9u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        (0..SINGLE_SIDE * SINGLE_SIDE).map(|i| {
            let (x, y) = (i % SINGLE_SIDE, i / SINGLE_SIDE);
            match (y / 40) % 4 {
                0 => 64,
                1 => (((x + y * 3) % 1000) as u32 + next() % 24).min(1023) as u16,
                2 => if (x / 2 + y / 2) % 2 == 0 { 0 } else { 1023 },
                _ => if (x / 16) % 2 == 0 { 512 } else { (480 + next() % 64) as u16 },
            }
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

    /// One tile's picture for a paired take of another frame size, as it is
    /// coded: `width` samples a row, `rows` rows, of `bits` bits. `which`
    /// names the picture (two tiles of different pictures tell a tile put
    /// in the wrong place). A 14-bit one has values with low bits set above
    /// 1024 and without them, and runs of 128 samples that need no tail.
    pub fn frame_tile_picture(width: usize, rows: usize, bits: u8, which: u32) -> Vec<u16> {
        let top = (1u32 << bits) - 1;
        let mut s = (which + 1).wrapping_mul(2654435761).max(1);
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        (0..width * rows).map(|i| {
            let (x, y) = (i % width, i / width);
            let ramp = ((x * (3 + which as usize) + y * 5) as u32) % (top + 1);
            let v = (ramp + next() % 48).min(top);
            // Rows of a 14-bit picture in turn: whole values, and values on the 16-step grid.
            (if bits == 14 && (y / 4) % 2 == 1 { v & !15 } else { v }) as u16
        }).collect()
    }

    /// Which of the two tile pictures tile `t` of channel `eye` holds.
    pub fn frame_tile_which(eye: usize, t: usize) -> u32 {
        ((t * 7 + eye * 3) % 5 < 2) as u32
    }

    /// The frames (channel 0, then channel 1) of a paired take of `frame`
    /// whose tiles hold the two pictures of [`frame_tile_picture`] in the
    /// order of [`frame_tile_which`], and one record of them.
    pub fn frame_record(profile: Profile, frame: PairedFrame, request: u32, exposure_ns: [u64; 2]) -> (Vec<u16>, Vec<u8>) {
        let layout = Layout::PairedOf(frame);
        let (side, coded, rows) = (layout.width(), layout.coded_width(), layout.tile_rows());
        let pictures: Vec<Vec<u16>> = (0..2).map(|which| frame_tile_picture(coded, rows, frame.bits, which)).collect();
        let coded_tiles: Vec<Vec<u8>> = pictures.iter().map(|p| match profile {
            Profile::Frame10 => encode_tile_split(p, coded),
            _ => encode_tile(profile, p, coded),
        }).collect();
        let mut samples = Vec::with_capacity(layout.samples());
        let mut tiles: Vec<&Vec<u8>> = Vec::new();
        for eye in 0..2 {
            for t in 0..layout.tiles_per_eye() {
                let which = frame_tile_which(eye, t) as usize;
                tiles.push(&coded_tiles[which]);
                // The frame's part of the tile: its first columns, and the rows the frame still has.
                let kept = rows.min(side - (t * rows).min(side));
                for row in pictures[which].chunks(coded).take(kept) {
                    samples.extend_from_slice(&row[..side]);
                }
            }
        }
        assert_eq!(samples.len(), layout.samples());
        let mut out = vec![0u8; BLOCK as usize];
        out[..8].copy_from_slice(profile.signature());
        for (o, v) in [(8, BLOCK as u32), (12, request), (16, side as u32), (20, side as u32), (24, PairedFrame::TILES as u32), (28, 0)] {
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
        (samples, out)
    }

    /// Write a finished paired take of `frame` holding `records` into `dir`,
    /// stating the black and white level `levels`.
    pub fn write_frame_take(dir: &Path, frame: PairedFrame, fps: u32, levels: (u16, u16), window: (u32, u32), records: &[Vec<u8>]) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes: Vec<u8> = records.concat();
        std::fs::write(dir.join(CAPTURE_FILE), &bytes).unwrap();
        std::fs::write(dir.join(CLIP_FILE), format!(
            r#"{{"layout":"paired-bayer-v1","fps_setting":{fps},"size":[{side},{side}],"encoded_bayer_size":[{coded},{coded}],"sensor_bits":{bits},"black_level":{},"white_level":{},"cfa_pattern":[1,2,0,1],"frames_file":"frames.csv","exposure_file":"exposure.csv","sensor_metadata":"sensor-metadata.json","clock_start":{{"counter_ticks":1920000,"counter_hz":19200000,"mono_before_ns":1,"mono_after_ns":2,"boottime_ns":1}}}}"#,
            levels.0, levels.1, side = frame.side, coded = frame.coded, bits = frame.bits)).unwrap();
        std::fs::write(dir.join(SEAL_FILE), format!(
            r#"{{"fps_setting":{fps},"request_window":[{},{}],"received_pairs":{n},"pairs":{n},"bytes":{}}}"#,
            window.0, window.1, bytes.len(), n = records.len())).unwrap();
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

    /// The symbol a record for one eye stores a 10-bit `value` as.
    pub fn eye_symbol(profile: Profile, value: u16) -> u16 {
        match value {
            _ if profile.is_exact() => value,
            0..=84 => value,
            896.. => value - 406,
            _ => 84 + (value - 84 + 1) / 2,
        }
    }

    /// One sensor's frame of `frame` (the samples a reader gets back) and
    /// its record, stored under `profile`: the stripes hold the two
    /// pictures of [`frame_tile_picture`] in the order of
    /// [`frame_tile_which`], cut where the frame ends.
    #[allow(clippy::too_many_arguments)]
    pub fn eye_record(profile: Profile, frame: EyeFrame, eye: usize, sequence: u32, hardware: u64, exposure_ns: u64, shutter_ns: u64) -> (Vec<u16>, Vec<u8>) {
        let (w, h, coded, rows) = (frame.width as usize, frame.height as usize, frame.coded as usize, EyeFrame::STRIPE_ROWS);
        let (mode, quantiser) = profile.eye_record().expect("a profile of a record for one eye");
        let levels = profile.levels();
        // A picture as the symbols of a stripe: the frame's columns, zeros
        // after them, and zeros in the rows after the first `kept`.
        let symbols = |which: u32, kept: usize| -> Vec<u16> {
            let mut p = frame_tile_picture(coded, rows, 10, which + 2 * eye as u32);
            for (y, row) in p.chunks_mut(coded).enumerate() {
                row[if y < kept { w } else { 0 }..].fill(0);
            }
            p.iter().map(|&v| eye_symbol(profile, v)).collect()
        };
        let whole: Vec<(Vec<u16>, Vec<u8>)> = (0..2).map(|which| {
            let s = symbols(which, rows);
            let coded_stripe = encode_tile_split(&s, coded);
            (s, coded_stripe)
        }).collect();
        let stripes = frame.stripes();
        let (mut samples, mut body, mut lens) = (Vec::with_capacity(w * h), Vec::new(), Vec::with_capacity(stripes));
        for t in 0..stripes {
            let (which, kept) = (frame_tile_which(eye, t), rows.min(h - t * rows));
            let cut;
            let (s, stripe) = if kept == rows {
                (&whole[which as usize].0, &whole[which as usize].1)
            } else {
                let s = symbols(which, kept);
                let coded_stripe = encode_tile_split(&s, coded);
                cut = (s, coded_stripe);
                (&cut.0, &cut.1)
            };
            lens.push(stripe.len() as u32 - 4);
            body.extend_from_slice(stripe);
            for row in s.chunks(coded).take(kept) {
                samples.extend(row[..w].iter().map(|&v| levels[v as usize]));
            }
        }
        let len = BLOCK + pad(body.len() as u64);
        let mut out = vec![0u8; BLOCK as usize];
        out[..8].copy_from_slice(profile.signature());
        for (o, v) in [
            (8, 2), (12, len as u32), (16, body.len() as u32), (20, eye as u32), (24, 1000 * eye as u32 + sequence), (28, sequence),
            (32, w as u32), (36, h as u32), (40, coded as u32), (44, stripes as u32),
            (48, quantiser[0]), (52, quantiser[1]), (56, quantiser[2]), (60, crc32c(&body)), (88, mode),
        ] {
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (o, v) in [(96, hardware), (104, exposure_ns), (112, shutter_ns)] {
            out[o..o + 8].copy_from_slice(&v.to_le_bytes());
        }
        for (i, n) in lens.iter().enumerate() {
            out[128 + 4 * i..132 + 4 * i].copy_from_slice(&n.to_le_bytes());
        }
        out.extend_from_slice(&body);
        out.resize(len as usize, 0);
        (samples, out)
    }

    /// Write a finished take of `frame` with a record for each eye holding
    /// `records` (in the order given) into `dir`: `pairs` of them inside
    /// `window`, `eye_frames` frames of each sensor.
    pub fn write_eye_take(dir: &Path, frame: EyeFrame, fps: u32, window: (u32, u32), pairs: usize, eye_frames: [usize; 2], records: &[Vec<u8>]) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes: Vec<u8> = records.concat();
        std::fs::write(dir.join(CAPTURE_FILE), &bytes).unwrap();
        std::fs::write(dir.join(CLIP_FILE), format!(
            r#"{{"layout":"paired-bayer-eye-records-v2","fps_setting":{fps},"size":[{},{}],"sensor_bits":10,"black_level":64,"white_level":1023,"cfa_pattern":[1,2,0,1],"exposure_file":"exposure.csv","sensor_metadata":"sensor-metadata.json","clock_start":{{"counter_ticks":1920000,"counter_hz":19200000,"mono_before_ns":1,"mono_after_ns":2,"boottime_ns":1}}}}"#,
            frame.width, frame.height)).unwrap();
        std::fs::write(dir.join(SEAL_FILE), format!(
            r#"{{"fps_setting":{fps},"layout":"paired-bayer-eye-records-v2","request_window":[{},{}],"received_pairs":{pairs},"pairs":{pairs},"bytes":{},"eye_frames":{eye_frames:?},"eye_hardware_gaps":[0,0],"unpaired_eye_frames":[{},{}],"exposure_status":"ok"}}"#,
            window.0, window.1, bytes.len(), eye_frames[0] - pairs, eye_frames[1] - pairs)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    /// The profile of a signature in a 3840×3840 paired take set to `fps`.
    fn of_pair(signature: &[u8], fps: u32) -> Option<Profile> {
        Profile::from_signature(signature, Format::paired(fps))
    }

    /// The profile of a signature in a take of one lens set to `fps`.
    fn of_one_lens(signature: &[u8], fps: u32) -> Option<Profile> {
        Profile::from_signature(signature, Format { layout: Layout::Single { channel: 0 }, fps })
    }

    /// The frame rate setting a 3840×3840 paired take's profile belongs to.
    fn rate(profile: Profile) -> u32 {
        [30, 50, 60].into_iter().find(|&fps| profile.records(Layout::Paired, fps)).expect("a paired profile")
    }

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
        // One lens: a symbol is the 10-bit sample, in either block layout.
        let t = Profile::Single10.levels();
        assert!(t.len() == 1024 && t.iter().enumerate().all(|(i, v)| *v as usize == i));
        assert_eq!(Profile::Single10Split.levels(), t);
    }

    #[test]
    fn a_signature_belongs_to_one_frame_rate() {
        assert_eq!(of_pair(b"L1024___", 30), Some(Profile::Exact30));
        assert_eq!(of_pair(b"X6QBLK0B", 60), Some(Profile::Raised60));
        assert_eq!(of_pair(b"X6QBLK0B", 50), None);
        assert_eq!(of_pair(b"X6M50AB2", 60), None);
        assert_eq!(of_pair(b"UNKNOWN!", 30), None);
        // Two signatures at 30 fps, one for each layout.
        assert_eq!(of_one_lens(b"X6S8B01\0", 30), Some(Profile::Single10));
        assert_eq!(of_one_lens(b"X6S8B01\0", 60), None);
        assert!(Profile::Single10.layout_is(Layout::Single { channel: 1 }) && !Profile::Single10.layout_is(Layout::Paired));
        // One lens has a second signature, for its later block layout.
        assert_eq!(of_one_lens(b"X6S8B02\0", 30), Some(Profile::Single10Split));
        assert_eq!(of_one_lens(b"X6S8B02\0", 50), None);
        assert_eq!(of_one_lens(b"X6S8B03\0", 30), None);
        assert!(Profile::Single10Split.layout_is(Layout::Single { channel: 0 }) && !Profile::Single10Split.layout_is(Layout::Paired));
        // A kept pair list names a profile by its place: the earlier ones stay where they were.
        assert_eq!(Profile::ALL[..6], [Profile::Exact30, Profile::Mid50, Profile::Raised50, Profile::Coarse60, Profile::Raised60, Profile::Single10]);
        assert!(Profile::Exact30.layout_is(Layout::Paired) && !Profile::Exact30.layout_is(Layout::Single { channel: 0 }));
    }

    #[test]
    fn every_profile_decodes_what_was_stored() {
        for (n, profile) in Profile::ALL.into_iter().filter(|p| p.layout_is(Layout::Paired)).enumerate() {
            let samples = test_samples(7 + n as u32);
            let record = encode_record(profile, 100 + n as u32, [1_000_000, 999_971], &samples);
            let mut out = vec![0u16; 2 * EYE_SAMPLES];
            let got = decode_record(&record, Format::paired(rate(profile)), Channels::Both, &mut out, 0).unwrap();
            assert_eq!((got.request, got.exposure_ns, got.len as usize), (100 + n as u32, [1_000_000, 999_971], record.len()));
            let want: Vec<u16> = samples.iter().map(|&v| stored_value(profile, v)).collect();
            assert_eq!(fnv(&out), fnv(&want), "{profile:?}");
            if profile.is_exact() {
                assert_eq!(fnv(&out), fnv(&samples), "30 fps keeps every value");
            }
            // One thread, and one channel only, give the same samples.
            let mut one = vec![0u16; 2 * EYE_SAMPLES];
            decode_record(&record, Format::paired(rate(profile)), Channels::Both, &mut one, 1).unwrap();
            assert_eq!(fnv(&one), fnv(&out));
            let mut right = vec![7u16; 2 * EYE_SAMPLES];
            decode_record(&record, Format::paired(rate(profile)), Channels::One(1), &mut right, 0).unwrap();
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
        let a = encode_single_record(Profile::Single10, 41, 900, 7_000_000_000, 8_000_000, &samples);
        let b = encode_single_record(Profile::Single10, 43, 902, 7_066_733_333, 4_000_000, &samples);
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

    #[test]
    fn the_later_block_layout_reads_as_it_is_written() {
        // Four rows of 32 samples with the widths 5 | 11, 0 | 3, 7 = 7 and
        // 2 | 6: written from the layout's description by another program,
        // which read them back with a transcription of the camera side's
        // reference reader.
        let stream: [u8; 86] = [
            181, 4, 16, 156, 222, 216, 141, 38, 33, 115, 8, 192, 118, 140, 186, 128, 162, 88, 153, 246, 144, 75, 211, 19,
            36, 16, 209, 165, 138, 153, 159, 181, 52, 48, 48, 72, 216, 178, 6, 15, 119, 252, 106, 88, 128, 84, 135, 58,
            46, 11, 132, 73, 138, 2, 65, 226, 110, 118, 146, 68, 234, 42, 176, 202, 241, 233, 250, 63, 144, 98, 34, 40,
            9, 24, 38, 20, 32, 144, 1, 68, 186, 4, 188, 99, 54, 144,
        ];
        let want: [u16; 128] = [
            2, 8, 15, 19, 20, 3, 13, 1, 19, 1, 24, 8, 15, 14, 7, 6,
            868, 62, 888, 508, 892, 847, 490, 219, 852, 791, 37, 1017, 516, 863, 435, 653,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 2, 3, 2, 3, 1, 0, 4, 2, 2, 4, 4, 0, 5, 0,
            63, 57, 83, 38, 94, 17, 83, 32, 39, 21, 41, 4, 30, 7, 53, 53,
            505, 95, 688, 323, 810, 552, 628, 351, 702, 629, 411, 759, 403, 796, 422, 685,
            1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0,
            20, 10, 27, 15, 14, 11, 37, 8, 45, 27, 30, 29, 29, 8, 18, 22,
        ];
        let mut got = [0u16; 128];
        unpack_split(&stream, &mut got, 32, 11, 1024).unwrap();
        assert_eq!(got, want);
        // Cut short, a byte too many, a width beyond the widest, a symbol beyond the table: refused.
        assert!(unpack_split(&stream[..85], &mut got, 32, 11, 1024).is_err());
        assert!(unpack_split(&[&stream[..], &[0]].concat(), &mut got, 32, 11, 1024).is_err());
        let mut wide = stream;
        wide[0] = 0xc5;
        assert!(unpack_split(&wide, &mut got, 32, 11, 1024).is_err());
        assert!(unpack_split(&stream, &mut got, 32, 11, 512).is_err());
        // The earlier layout reads another stream in these bytes.
        assert!(unpack_symbols(&stream, &mut [0u16; 256], 64, 11, 1024).is_err());
    }

    #[test]
    fn one_lens_records_decode_in_either_block_layout() {
        let dir = temp_dir("split");
        let single = Format { layout: Layout::Single { channel: 0 }, fps: 30 };
        let (plain, banded) = (single_test_samples(5), banded_test_samples());
        // One capture may hold both: a record in the earlier layout, then two in the later one.
        let a = encode_single_record(Profile::Single10, 7, 100, 1_000_000_000, 6_000_000, &plain);
        let b = encode_single_record(Profile::Single10Split, 8, 101, 1_033_366_667, 6_000_000, &banded);
        let c = encode_single_record(Profile::Single10Split, 9, 102, 1_066_733_333, 6_000_000, &plain);
        write_single_take(&dir, 0, (100, 102), &[a.clone(), b.clone(), c.clone()]);
        let take = read_take(&dir).unwrap().unwrap();
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(pairs.iter().map(|p| (p.request, p.profile)).collect::<Vec<_>>(),
            vec![(7, Profile::Single10), (8, Profile::Single10Split), (9, Profile::Single10Split)]);
        let mut file = File::open(take.capture_path()).unwrap();
        let (mut record, mut out) = (Vec::new(), vec![0u16; SINGLE_SIDE * SINGLE_SIDE]);
        for (entry, want) in pairs.iter().zip([&plain, &banded, &plain]) {
            decode_pair(&mut file, entry, take.format(), Channels::Both, &mut record, &mut out, 0).unwrap();
            assert_eq!(fnv(&out), fnv(want), "request {}", entry.request);
        }
        // On one thread as on many.
        let mut one = vec![0u16; SINGLE_SIDE * SINGLE_SIDE];
        let got = decode_record(&b, single, Channels::Both, &mut one, 1).unwrap();
        assert_eq!((fnv(&one), got.request, got.profile, got.len as usize), (fnv(&banded), 8, Profile::Single10Split, b.len()));
        // A record under the other layout's signature is refused, either way round.
        for (bytes, other) in [(&a, Profile::Single10Split), (&c, Profile::Single10)] {
            let mut renamed = bytes.clone();
            renamed[..8].copy_from_slice(other.signature());
            assert!(decode_record(&renamed, single, Channels::Both, &mut out, 0).is_err(), "{other:?}");
        }
        // So are a width beyond the widest and bytes after a tile's stream.
        let mut wide = b.clone();
        wide[BLOCK as usize + 4] = 0xcc;
        assert!(decode_record(&wide, single, Channels::Both, &mut out, 0).is_err());
        let mut longer = b.clone();
        let first = u32::from_le_bytes(b[48..52].try_into().unwrap());
        longer[48..52].copy_from_slice(&(first + 1).to_le_bytes());
        assert!(decode_record(&longer, single, Channels::Both, &mut out, 0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A last record the camera did not finish (the top of its frame, then
    /// what the buffer held of an earlier one) is told from a whole one.
    #[test]
    fn a_last_record_left_unfinished_is_told() {
        let dir = temp_dir("unfinished");
        let layout = Layout::Single { channel: 0 };
        let (early, between, late) = (single_test_samples(21), single_test_samples(22), single_test_samples(23));
        let record = |request: u32, samples: &[u16]| encode_single_record(Profile::Single10Split, request, 100 + request as u64, 1_000_000_000 + request as u64 * 33_366_667, 4_000_000, samples);
        let (a, b) = (record(10, &early), record(11, &between));
        // Whole: three frames, each its own.
        write_single_take(&dir, 0, (110, 112), &[a.clone(), b.clone(), record(12, &late)]);
        let take = read_take(&dir).unwrap().unwrap();
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(leftover_in_last_record(&take, &pairs).unwrap(), None);
        // Unfinished: the last frame's first nine tiles, then request 10's.
        let cut = 9 * layout.tile_samples();
        let mixed: Vec<u16> = late[..cut].iter().chain(&early[cut..]).copied().collect();
        let unfinished = record(12, &mixed);
        write_single_take(&dir, 0, (110, 112), &[a.clone(), b.clone(), unfinished.clone()]);
        let take = read_take(&dir).unwrap().unwrap();
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(leftover_in_last_record(&take, &pairs).unwrap(), Some(Leftover { of_request: 10, tiles: 15 }));
        // The records before it are whole: only the last one is looked at.
        assert_eq!(leftover_in_last_record(&take, &pairs[..2]).unwrap(), None);
        // One record, or none: nothing to hold it against.
        assert_eq!(leftover_in_last_record(&take, &pairs[..1]).unwrap(), None);
        assert_eq!(leftover_in_last_record(&take, &[]).unwrap(), None);
        // With the checksums in the headers the answer is the same.
        let with_crcs = |mut r: Vec<u8>, samples: &[u16]| {
            for (i, tile) in samples.chunks(layout.tile_samples()).enumerate() {
                r[CRC_AT + 4 * i..CRC_AT + 4 * i + 4].copy_from_slice(&packed_crc(tile).to_le_bytes());
            }
            r[CRC_FLAG_AT..CRC_FLAG_AT + 4].copy_from_slice(&1u32.to_le_bytes());
            r
        };
        write_single_take(&dir, 0, (110, 112), &[with_crcs(a, &early), with_crcs(b, &between), with_crcs(unfinished, &mixed)]);
        let take = read_take(&dir).unwrap().unwrap();
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(leftover_in_last_record(&take, &pairs).unwrap(), Some(Leftover { of_request: 10, tiles: 15 }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record of a real take of one lens against the checksums it carries
    /// (a take's first record has them, so a capture file itself will do):
    /// skipped unless `VR180_ONE_LENS_RECORD` names a file that starts at a
    /// record's header.
    #[test]
    fn a_real_record_of_one_lens_matches_its_checksums() {
        let Some(path) = std::env::var_os("VR180_ONE_LENS_RECORD") else { return };
        let single = Format { layout: Layout::Single { channel: 0 }, fps: 30 };
        let mut file = File::open(&path).unwrap();
        let mut record = vec![0u8; BLOCK as usize];
        file.read_exact(&mut record).unwrap();
        let len = Header::parse(&record, single, 0).unwrap().record_len() as usize;
        record.resize(len, 0);
        file.read_exact(&mut record[BLOCK as usize..]).unwrap();
        let mut out = vec![0u16; SINGLE_SIDE * SINGLE_SIDE];
        let got = decode_record(&record, single, Channels::Both, &mut out, 0).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(record[o..o + 4].try_into().unwrap());
        assert_eq!(u32_at(240), 1, "request {} carries no checksums", got.request);
        for (i, tile) in out.chunks(single.layout.tile_samples()).enumerate() {
            assert_eq!(packed_crc(tile), u32_at(144 + 4 * i), "tile {i} of request {} ({:?})", got.request, got.profile);
        }
        println!("request {} ({:?}): 24 tiles match their checksums", got.request, got.profile);
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

    /// The frame of [`PAIRED_FRAMES`] with this side.
    fn frame_of(side: u16) -> PairedFrame {
        PAIRED_FRAMES.into_iter().find(|f| f.side == side).expect("a frame of that side")
    }

    /// Paired takes of the other frame sizes: each reads as its own layout
    /// with the sample format it states, its records decode to what was
    /// stored tile for tile (the 4912 frame out of its larger canvas), and a
    /// signature belongs to the frames of its own depth and frame rates.
    #[test]
    fn a_paired_take_of_another_frame_size_reads_and_decodes() {
        for (side, fps, profile, levels) in [(5504u16, 30u32, Profile::Frame10, (64u16, 1023u16)), (4912, 30, Profile::Frame14, (1024, 16368)), (3968, 60, Profile::Frame10, (64, 1023))] {
            let frame = frame_of(side);
            let layout = Layout::PairedOf(frame);
            let format = Format { layout, fps };
            assert_eq!((layout.width(), layout.height(), layout.channels(), layout.tiles(), layout.bits()), (side as usize, side as usize, 2, 32, frame.bits));
            assert_eq!(layout.tile_rows() * 16, frame.coded as usize);
            assert_eq!(Profile::from_signature(profile.signature(), format), Some(profile));
            let (samples, record) = frame_record(profile, frame, 300, [5_000_000, 5_000_711]);
            assert_eq!(samples.len(), 2 * side as usize * side as usize);
            let mut out = vec![0u16; layout.samples()];
            let got = decode_record(&record, format, Channels::Both, &mut out, 0).unwrap();
            assert_eq!((got.request, got.profile, got.exposure_ns, got.len as usize), (300, profile, [5_000_000, 5_000_711], record.len()));
            assert!(out == samples, "{side}: the frame is not what was stored");
            // The two tile pictures differ, so a tile in the wrong place would show.
            let rows = layout.tile_rows() * side as usize;
            assert_ne!(fnv(&samples[..rows]), fnv(&samples[2 * rows..3 * rows]), "{side}: tiles 0 and 2 hold different pictures");
            // One thread, and one channel only, give the same samples.
            let mut one = vec![0u16; layout.samples()];
            decode_record(&record, format, Channels::Both, &mut one, 1).unwrap();
            assert!(one == out, "{side}: one thread");
            let mut second = vec![9u16; layout.samples()];
            decode_record(&record, format, Channels::One(1), &mut second, 0).unwrap();
            let per = layout.eye_samples();
            assert!(second[..per].iter().all(|&v| v == 9) && second[per..] == out[per..], "{side}: channel 1 alone");

            // As a take: the layout, the format it states, its pairs.
            let dir = temp_dir(&format!("frame{side}"));
            let (_, earlier) = frame_record(profile, frame, 299, [4_000_000, 4_000_500]);
            write_frame_take(&dir, frame, fps, levels, (299, 301), &[record.clone(), earlier]);
            assert_eq!(take_layout(&dir), Some(layout));
            let take = read_take(&dir).unwrap().unwrap();
            assert_eq!((take.layout, take.width, take.height, take.fps_setting), (layout, side as u32, side as u32, fps));
            assert_eq!((take.cfa, take.black_level, take.white_level, take.stated_levels), (CFA, levels.0, levels.1, None));
            assert_eq!(take.sample_max(), if frame.bits == 14 { 16383 } else { 1023 });
            assert!(take.window_numbers_records() && take.missing_pairs() == 1);
            // Records land in the capture in any order: pairs come back by request.
            let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
            assert_eq!(pairs.iter().map(|p| p.request).collect::<Vec<_>>(), vec![299, 300]);
            assert!(pairs[0].offset > pairs[1].offset && pairs[0].profile == profile);
            assert_eq!(missing_requests(&take, &pairs), vec![301]);
            let mut file = File::open(take.capture_path()).unwrap();
            let mut scratch = Vec::new();
            decode_pair(&mut file, &pairs[1], take.format(), Channels::Both, &mut scratch, &mut one, 0).unwrap();
            assert!(one == samples, "{side}: read from the take");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// What does not belong to a frame is refused: a signature of the other
    /// depth, a frame rate the frame is not recorded at, header fields of
    /// another layout, bytes after a 10-bit tile's stream, a 14-bit tile's
    /// tail cut short.
    #[test]
    fn a_record_of_another_frame_size_is_checked() {
        let (wide, crop, small) = (frame_of(5504), frame_of(4912), frame_of(3968));
        let at = |frame: PairedFrame, fps: u32| Format { layout: Layout::PairedOf(frame), fps };
        // Signatures: the 10-bit one for the 10-bit frames at their rates, the 14-bit one for the 14-bit frame.
        for (signature, format, want) in [
            (b"X6D5B02\0", at(wide, 30), Some(Profile::Frame10)), (b"X6D5B02\0", at(small, 50), Some(Profile::Frame10)),
            (b"X6D5B02\0", at(small, 60), Some(Profile::Frame10)), (b"X6D4B01\0", at(crop, 30), Some(Profile::Frame14)),
            (b"X6D5B02\0", at(crop, 30), None), (b"X6D4B01\0", at(wide, 30), None), (b"X6D5B02\0", at(wide, 60), None),
            (b"X6D5B02\0", at(small, 30), None), (b"X6D5B02\0", Format::paired(30), None), (b"L1024___", at(crop, 30), None),
            (b"X6S8B02\0", at(wide, 30), None), (b"X6D5B03\0", at(wide, 30), None),
        ] {
            assert_eq!(Profile::from_signature(signature, format), want, "{:?} in {format:?}", std::str::from_utf8(signature));
        }
        assert!(Profile::Frame10.layout_is(Layout::PairedOf(small)) && !Profile::Frame10.layout_is(Layout::Paired) && !Profile::Frame14.layout_is(Layout::PairedOf(small)));

        let (_, good) = frame_record(Profile::Frame10, small, 12, [10, 20]);
        let format = at(small, 60);
        let mut out = vec![0u16; format.layout.samples()];
        assert!(decode_record(&good, format, Channels::Both, &mut out, 0).is_ok());
        let with = |o: usize, v: u32| { let mut r = good.clone(); r[o..o + 4].copy_from_slice(&v.to_le_bytes()); r };
        // 24 tiles an eye, the threshold of the 3840 layout, another side, another header length.
        for (what, record) in [("tiles", with(24, 24)), ("threshold", with(28, 13472)), ("width", with(16, 3840)), ("height", with(20, 3840)), ("header", with(8, 2048))] {
            assert!(decode_record(&record, format, Channels::Both, &mut out, 0).is_err(), "{what}");
        }
        // The other layouts do not take it, nor it theirs.
        assert!(decode_record(&good, Format::paired(60), Channels::Both, &mut vec![0u16; 2 * EYE_SAMPLES], 0).is_err());
        assert!(decode_record(&good, at(wide, 30), Channels::Both, &mut vec![0u16; at(wide, 30).layout.samples()], 0).is_err());
        // A 10-bit tile is its predictor stream and nothing else: a byte more in its stated length is left over.
        let first = u32::from_le_bytes(good[48..52].try_into().unwrap());
        assert!(decode_record(&with(48, first + 1), format, Channels::Both, &mut out, 0).is_err());

        // A 14-bit tile: its tail cut short, and a tail that says more groups carry low bits than it holds.
        let (stored, good) = frame_record(Profile::Frame14, crop, 13, [10, 20]);
        let format = at(crop, 30);
        let mut out = vec![0u16; format.layout.samples()];
        decode_record(&good, format, Channels::Both, &mut out, 0).unwrap();
        assert!(out == stored);
        assert!(stored.iter().any(|v| *v >= 1024 && v & 15 != 0), "the picture has low bits above 1024");
        let first = u32::from_le_bytes(good[48..52].try_into().unwrap());
        let mut short = good.clone();
        short[48..52].copy_from_slice(&(first - 64).to_le_bytes());
        assert!(decode_record(&short, format, Channels::Both, &mut out, 0).is_err());
        let stream = u32::from_le_bytes(good[BLOCK as usize..BLOCK as usize + 4].try_into().unwrap()) as usize;
        let mut marked = good.clone();
        let bitmap = BLOCK as usize + 4 + stream;
        marked[bitmap..bitmap + 8].copy_from_slice(&[0xff; 8]);
        assert!(marked != good && decode_record(&marked, format, Channels::Both, &mut out, 0).is_err());

        // A take: a frame rate the frame is not recorded at, a canvas or a depth that is not the frame's.
        let dir = temp_dir("frame-checks");
        write_frame_take(&dir, small, 30, (64, 1023), (12, 12), &[good.clone()]);
        assert!(read_take(&dir).is_err(), "3968 at 30 fps");
        write_frame_take(&dir, wide, 60, (64, 1023), (12, 12), &[good.clone()]);
        assert!(read_take(&dir).is_err(), "5504 at 60 fps");
        write_frame_take(&dir, PairedFrame { coded: 4912, ..crop }, 30, (1024, 16368), (13, 13), &[good.clone()]);
        assert!(read_take(&dir).is_err(), "a 4912 frame coded on a 4912 canvas");
        write_frame_take(&dir, PairedFrame { bits: 10, ..crop }, 30, (64, 1023), (13, 13), &[good.clone()]);
        assert!(read_take(&dir).is_err(), "a 4912 frame of 10 bits");
        // A frame size this module does not know is named.
        std::fs::write(dir.join(CLIP_FILE), r#"{"layout":"paired-bayer-v1","fps_setting":30,"size":[4000,4000]}"#).unwrap();
        assert!(read_take(&dir).unwrap_err().to_string().contains("4000"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A 14-bit frame's levels: the sensor's own where the take states the
    /// 3840×3840 layout's, the take's where it states any other.
    #[test]
    fn a_14_bit_frame_is_read_by_the_sensors_levels() {
        let (crop, wide) = (frame_of(4912), frame_of(5504));
        assert_eq!(sensor_levels(Layout::PairedOf(crop), (64, 15408)), Some((1024, 16368)));
        assert_eq!(sensor_levels(Layout::PairedOf(crop), (1024, 16368)), None);
        assert_eq!(sensor_levels(Layout::PairedOf(crop), (256, 16000)), None);
        assert_eq!(sensor_levels(Layout::PairedOf(wide), (64, 1023)), None);
        assert_eq!(sensor_levels(Layout::Paired, (64, 15408)), None);
        let (_, record) = frame_record(Profile::Frame14, crop, 5, [10, 20]);
        let dir = temp_dir("frame-levels");
        write_frame_take(&dir, crop, 30, (64, 15408), (5, 5), &[record.clone()]);
        let take = read_take(&dir).unwrap().unwrap();
        assert_eq!((take.black_level, take.white_level, take.stated_levels), (1024, 16368, Some((64, 15408))));
        write_frame_take(&dir, crop, 30, (1000, 16000), (5, 5), &[record]);
        let take = read_take(&dir).unwrap().unwrap();
        assert_eq!((take.black_level, take.white_level, take.stated_levels), (1000, 16000, None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The sensor record places every frame: where the sensor was read,
    /// through its binning, its kept part, its scale and the frame's window.
    #[test]
    fn the_sensor_record_places_every_frame() {
        let dir = temp_dir("windows");
        std::fs::create_dir_all(&dir).unwrap();
        let eye = |roi: [u32; 4], bin: u32, kept: [u32; 4], out: [u32; 2], line: f64, span: &str, window: Option<[u32; 4]>| format!(
            r#"{{"analog_roi_register_xyxy_inclusive":{roi:?},"binning_xy":[{bin},{bin}],"digital_crop_after_binning_xywh":{kept:?},"bayer_output_wh":{out:?},"line_period_ns":{line},"rolling_shutter_first_to_last_row_ns":{span}{}}}"#,
            window.map_or(String::new(), |w| format!(r#","recorded_bayer_window_xywh":{w:?}"#)));
        let take_of = |layout: Layout| Take {
            dir: dir.clone(), layout, fps_setting: 30, width: layout.width() as u32, height: layout.height() as u32, cfa: CFA, black_level: 64, white_level: 1023, stated_levels: None,
            clock_start: None, audio_file: None, audio_timing: None, sensor_metadata: Some("sensor-metadata.json".into()), exposure_file: None, white_balance_file: None,
            seal: Seal { fps_setting: 30, request_window: (1, 1), received_pairs: 1, pairs: 1, bytes: 0, eye_frames: None, eye_gaps: None }, capture_bytes: 0,
        };
        let placed = |take: &Take, cam0: String, cam1: Option<String>| -> Option<[EyeWindow; 2]> {
            let other = cam1.map_or(String::new(), |e| format!(r#","CAM1":{e}"#));
            std::fs::write(dir.join("sensor-metadata.json"), format!(r#"{{"schema":"raw-sensor-metadata-v1","eyes":{{"CAM0":{cam0}{other}}}}}"#)).unwrap();
            read_sensor_windows(take)
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
        // 3840×3840: the sensor binned 2×2, the middle 3840 columns of its 4096.
        let binned = || eye([0, 256, 8191, 7935], 2, [128, 0, 3840, 3840], [3840, 3840], 3791.6666628749995, "14556208", None);
        let w = placed(&take_of(Layout::Paired), binned(), Some(binned())).unwrap();
        assert_eq!((w[0].origin, w[0].scale, w[0].rows_read), ((256.0, 256.0), (0.5, 0.5), 3840.0));
        assert!(close(w[1].first_to_last_row_ns(), 14_556_208.0));
        // One lens at 7680×7680: the same window, every pixel.
        let full = eye([0, 256, 8191, 7935], 1, [256, 0, 7680, 7680], [7680, 7680], 4000.000009472, "30716000", None);
        let w = placed(&take_of(Layout::Single { channel: 0 }), full, None).unwrap();
        assert_eq!((w[0].origin, w[0].scale, w[0]), ((256.0, 256.0), (1.0, 1.0), w[1]));
        // 5504×5504: 7912 rows scaled by 16/23, and the first 7912 of the 7920 columns kept. No stated span: the rows read, a line apart.
        let scaled = || eye([0, 128, 8191, 8063], 1, [136, 12, 7920, 7912], [5504, 5504], 4000.000055296001, "null", None);
        let w = placed(&take_of(Layout::PairedOf(frame_of(5504))), scaled(), Some(scaled())).unwrap();
        assert!(close(w[0].origin.0, 136.0) && close(w[0].origin.1, 140.0) && close(w[0].scale.0, 16.0 / 23.0) && close(w[0].scale.1, 16.0 / 23.0), "{:?}", w[0]);
        assert!(close(w[0].rows_read, 7912.0) && (w[0].first_to_last_row_ns() - 7911.0 * 4000.000055296001).abs() < 1.0);
        assert!(close(read_sensor_readout(&take_of(Layout::PairedOf(frame_of(5504)))).unwrap()[0].first_to_last_row_ns, w[0].first_to_last_row_ns()));
        // 4912×4912: the sensor's own pixels, the middle of it.
        let crop = || eye([0, 1632, 8191, 6559], 1, [1640, 8, 4912, 4912], [4912, 4912], 4333.333329, "21281000", Some([0, 0, 4912, 4912]));
        let w = placed(&take_of(Layout::PairedOf(frame_of(4912))), crop(), Some(crop())).unwrap();
        assert_eq!((w[0].origin, w[0].scale), ((1640.0, 1640.0), (1.0, 1.0)));
        assert!(close(w[0].first_to_last_row_ns(), 21_281_000.0));
        // 3968×3968: a window of a 4912×3968 output.
        let window = || eye([0, 2112, 8191, 6079], 1, [1640, 0, 4912, 3968], [4912, 3968], 4041.6666551074995, "16033292", Some([472, 0, 3968, 3968]));
        let take = take_of(Layout::PairedOf(frame_of(3968)));
        let w = placed(&take, window(), Some(window())).unwrap();
        assert_eq!((w[0].origin, w[0].scale, w[0].rows_read), ((2112.0, 2112.0), (1.0, 1.0), 3968.0));
        assert!(close(w[0].first_to_last_row_ns(), 16_033_292.0));
        // A record of another frame places nothing, and one eye missing of a pair neither.
        assert!(placed(&take, crop(), Some(crop())).is_none());
        assert!(placed(&take, window(), None).is_none());

        // Frames that are a record for each eye. One the sensor scales says
        // so in its registers, which give the scale.
        let scaled_eye = |kept: [u32; 4], out: u32, m: u32| format!(
            r#"{{"analog_roi_register_xyxy_inclusive":[0,128,8191,8063],"binning_xy":[1,1],"digital_crop_after_binning_xywh":{kept:?},"bayer_output_wh":[{out},{out}],"line_period_ns":4000.000055296001,"rolling_shutter_first_to_last_row_ns":null,"scaling_registers":{{"0x401":2,"0x404":0,"0x405":{m}}}}}"#);
        // 5760×5760: 7920 of the sensor's pixels each way, scaled by 16/22: all of the kept part.
        let take = take_of(Layout::EyeRecords(EYE_FRAMES[0]));
        let w = placed(&take, scaled_eye([136, 8, 7920, 7920], 5760, 22), Some(scaled_eye([136, 8, 7920, 7920], 5760, 22))).unwrap();
        assert!(close(w[0].origin.0, 136.0) && close(w[0].origin.1, 136.0) && close(w[0].scale.0, 16.0 / 22.0) && close(w[0].scale.1, 16.0 / 22.0), "{:?}", w[0]);
        assert!(close(w[0].rows_read, 7920.0) && (w[0].first_to_last_row_ns() - 7919.0 * 4000.000055296001).abs() < 1.0);
        // 6000×6000 scaled by 16/21 is 7875 pixels each way of a kept 7880:
        // its first columns and rows. The scale is the scaler's, not what
        // 6000 rows of 7880 would give.
        let take = take_of(Layout::EyeRecords(EYE_FRAMES[1]));
        let w = placed(&take, scaled_eye([156, 28, 7880, 7880], 6000, 21), Some(scaled_eye([156, 28, 7880, 7880], 6000, 21))).unwrap();
        assert!(close(w[0].origin.0, 156.0) && close(w[0].origin.1, 156.0) && close(w[0].scale.0, 16.0 / 21.0) && close(w[0].scale.1, 16.0 / 21.0), "{:?}", w[0]);
        assert!(close(w[0].rows_read, 7875.0) && (w[0].first_to_last_row_ns() - 7874.0 * 4000.000055296001).abs() < 1.0);
        // A scaler that is off states no scale: the rows give it.
        let unscaled = scaled_eye([156, 28, 7880, 7880], 6000, 21).replace(r#""0x401":2"#, r#""0x401":0"#);
        let w = placed(&take, unscaled.clone(), Some(unscaled)).unwrap();
        assert!(close(w[0].scale.1, 6000.0 / 7880.0) && close(w[0].origin.1, 156.0), "{:?}", w[0]);
        // The middle of the sensor at its own pixels: 5760×5760, 6000×6000, and 4608×3968.
        let native = |roi: [u32; 4], kept: [u32; 4], line: f64, span: &str| eye(roi, 1, kept, [kept[2], kept[3]], line, span, Some([0, 0, kept[2], kept[3]]));
        for (frame, roi, kept, line, span, origin) in [
            (EYE_FRAMES[0], [0, 1216, 8191, 6975], [1216, 0, 5760, 5760], 4333.333329, 24_955_666.641711, (1216.0, 1216.0)),
            (EYE_FRAMES[1], [0, 1088, 8191, 7103], [1096, 8, 6000, 6000], 4333.333329, 25_995_666.640671, (1096.0, 1096.0)),
            (EYE_FRAMES[2], [0, 2112, 8191, 6079], [1792, 0, 4608, 3968], 4041.6666551074995, 16_033_291.620811451, (1792.0, 2112.0)),
        ] {
            let take = take_of(Layout::EyeRecords(frame));
            let record = || native(roi, kept, line, &format!("{span}"));
            let w = placed(&take, record(), Some(record())).unwrap();
            assert_eq!((w[0].origin, w[0].scale, w[0].rows_read, w[0]), (origin, (1.0, 1.0), frame.height as f64, w[1]), "{frame:?}");
            assert!((w[0].first_to_last_row_ns() - span).abs() < 1.0, "{frame:?}");
            let r = read_sensor_readout(&take).unwrap();
            assert!(close(r[0].first_to_last_row_ns, w[0].first_to_last_row_ns()));
        }
        // The wide frame's record does not place the square one.
        assert!(placed(&take_of(Layout::EyeRecords(EYE_FRAMES[0])), native([0, 2112, 8191, 6079], [1792, 0, 4608, 3968], 4041.6666551074995, "16033292"), None).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_checksum_is_crc_32c() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        // Every length around the eight bytes a step takes, against the bit-by-bit sum.
        let bytes: Vec<u8> = (0..300u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for n in [1, 7, 8, 9, 15, 16, 17, 255, 300] {
            let mut c = !0u32;
            for &b in &bytes[..n] {
                c ^= b as u32;
                for _ in 0..8 {
                    c = (c >> 1) ^ if c & 1 == 1 { 0x82f6_3b78 } else { 0 };
                }
            }
            assert_eq!(crc32c(&bytes[..n]), !c, "{n} bytes");
        }
    }

    /// The format of a take of `frame` with a record for each eye.
    fn eyes_at(frame: EyeFrame, fps: u32) -> Format {
        Format { layout: Layout::EyeRecords(frame), fps }
    }

    /// The five signatures of a record for one eye, each with its frame and
    /// frame rate; the values stored to within one code value.
    #[test]
    fn a_record_for_one_eye_has_its_signature_and_levels() {
        let [square, larger, wide] = EYE_FRAMES;
        assert_eq!((square.stripes(), larger.stripes(), wide.stripes()), (180, 188, 124));
        for (signature, format, want) in [
            (b"X6F5O02\0", eyes_at(square, 30), Some(Profile::EyeScaled)), (b"X6C5O02\0", eyes_at(square, 30), Some(Profile::EyeNative)),
            (b"X6F6Q02\0", eyes_at(larger, 30), Some(Profile::EyeScaledNear)), (b"X6C6Q02\0", eyes_at(larger, 30), Some(Profile::EyeNativeNear)),
            (b"X6C9Q02\0", eyes_at(wide, 60), Some(Profile::EyeWideNear)),
            (b"X6F5O02\0", eyes_at(larger, 30), None), (b"X6F6Q02\0", eyes_at(square, 30), None), (b"X6C9Q02\0", eyes_at(wide, 30), None),
            (b"X6C9Q02\0", eyes_at(square, 60), None), (b"X6F5O02\0", eyes_at(square, 60), None), (b"X6F5O03\0", eyes_at(square, 30), None),
            (b"X6F5O02\0", Format::paired(30), None), (b"X6D5B02\0", eyes_at(square, 30), None), (b"L1024___", eyes_at(square, 30), None),
        ] {
            assert_eq!(Profile::from_signature(signature, format), want, "{:?} in {format:?}", std::str::from_utf8(signature));
        }
        let layout = Layout::EyeRecords(larger);
        assert_eq!((layout.width(), layout.height(), layout.coded_width(), layout.channels(), layout.bits()), (6000, 6000, 6016, 2, 10));
        assert_eq!((layout.tiles_per_eye(), layout.tile_rows(), layout.tile_samples(), layout.eye_samples()), (188, 32, 6016 * 32, 36_000_000));
        let layout = Layout::EyeRecords(wide);
        assert_eq!((layout.width(), layout.height(), layout.coded_width(), layout.samples()), (4608, 3968, 4608, 2 * 4608 * 3968));
        // Every value kept: a symbol is the sample.
        let exact = Profile::EyeScaled.levels();
        assert!(exact.len() == 1024 && exact.iter().enumerate().all(|(i, v)| *v as usize == i) && Profile::EyeNative.levels() == exact);
        assert!(Profile::EyeScaled.is_exact() && Profile::EyeNative.is_exact() && !Profile::EyeWideNear.is_exact());
        // To within one code value: 618 symbols, every value up to 84 and from 896, every other one between.
        let near = Profile::EyeScaledNear.levels();
        assert_eq!((near.len(), near[84], near[85], near[86], near[489], near[490], near[617]), (618, 84, 86, 88, 894, 896, 1023));
        assert!(Profile::EyeNativeNear.levels() == near && Profile::EyeWideNear.levels() == near);
        for value in 0..1024u16 {
            let back = near[eye_symbol(Profile::EyeScaledNear, value) as usize];
            let exact = value <= 84 || value >= 896 || value % 2 == 0;
            assert_eq!(back, if exact { value } else { value + 1 }, "{value}");
        }
        // The profiles a kept pair list already names keep their places.
        assert_eq!(Profile::ALL[7..9], [Profile::Frame10, Profile::Frame14]);
        assert_eq!(Profile::ALL[9..], [Profile::EyeScaled, Profile::EyeScaledNear, Profile::EyeNative, Profile::EyeNativeNear, Profile::EyeWideNear]);
    }

    /// A take with a record for each eye: the records lie in any order, one
    /// sensor starts a frame ahead of the other and loses a frame midway;
    /// the pairs come out by the first sensor's frame numbers with both
    /// records' places, and decode to what was stored, the 6000 frame out
    /// of its wider canvas and the 4608×3968 one as the rectangle it is.
    #[test]
    fn a_take_with_a_record_for_each_eye_reads_and_decodes() {
        for (profile, frame, fps) in [
            (Profile::EyeScaled, EYE_FRAMES[0], 30u32), (Profile::EyeScaledNear, EYE_FRAMES[1], 30), (Profile::EyeNative, EYE_FRAMES[0], 30),
            (Profile::EyeNativeNear, EYE_FRAMES[1], 30), (Profile::EyeWideNear, EYE_FRAMES[2], 60),
        ] {
            let layout = Layout::EyeRecords(frame);
            let format = Format { layout, fps };
            let step = 1_000_000_000 / fps as u64;
            // Sensor 0's frames 500, 501, 503 (502 was never written); sensor 1's 739…743, its first a frame ahead of the other sensor's first.
            let t = |k: u64| 9_000_000_000 + k * step;
            let frames: Vec<(usize, u64, u64, u64)> = vec![
                (1, 739, t(0) - step + 29_000, 4_000_000), (0, 500, t(0), 4_000_100), (1, 740, t(0) + 29_000, 4_000_200),
                (1, 741, t(1) + 29_000, 4_000_300), (0, 501, t(1), 4_000_400), (1, 742, t(2) + 29_000, 4_000_500),
                (1, 743, t(3) + 29_000, 4_000_600), (0, 503, t(3), 4_000_700),
            ];
            let built: Vec<(Vec<u16>, Vec<u8>)> = frames.iter().enumerate()
                .map(|(sequence, &(eye, hardware, stamp, shutter))| eye_record(profile, frame, eye, sequence as u32, hardware, stamp, shutter)).collect();
            let records: Vec<Vec<u8>> = built.iter().map(|b| b.1.clone()).collect();
            let offsets: Vec<u64> = records.iter().scan(0u64, |at, r| { let o = *at; *at += r.len() as u64; Some(o) }).collect();
            let dir = temp_dir(&format!("eyes{profile:?}"));
            write_eye_take(&dir, frame, fps, (500, 503), 3, [3, 5], &records);
            assert_eq!(take_layout(&dir), Some(layout));
            let take = read_take(&dir).unwrap().unwrap();
            assert_eq!((take.layout, take.width, take.height, take.fps_setting), (layout, frame.width as u32, frame.height as u32, fps));
            assert_eq!((take.cfa, take.black_level, take.white_level, take.sample_max()), (CFA, 64, 1023, 1023));
            assert_eq!((take.seal.eye_frames, take.seal.eye_gaps, take.pairs(), take.missing_pairs()), (Some([3, 5]), Some([0, 0]), 3, 1));
            let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
            assert_eq!(pairs.iter().map(|p| p.request).collect::<Vec<_>>(), vec![500, 501, 503]);
            assert_eq!(missing_requests(&take, &pairs), vec![502]);
            // Each pair: sensor 0's record and the record of sensor 1 that started 29 µs after it.
            for (pair, (first, second)) in pairs.iter().zip([(1usize, 2usize), (4, 3), (7, 6)]) {
                assert_eq!((pair.offset, pair.len, pair.other), (offsets[first], records[first].len() as u64, Some((offsets[second], records[second].len() as u64))));
                assert_eq!((pair.exposure_ns, pair.shutter_ns, pair.profile), ([frames[first].2, frames[second].2], [frames[first].3, frames[second].3], profile));
                let mut file = File::open(take.capture_path()).unwrap();
                let (mut scratch, mut out) = (Vec::new(), vec![0u16; layout.samples()]);
                decode_pair(&mut file, pair, take.format(), Channels::Both, &mut scratch, &mut out, 0).unwrap();
                let per = layout.eye_samples();
                assert!(out[..per] == built[first].0[..] && out[per..] == built[second].0[..], "{profile:?}: pair {} is not what was stored", pair.request);
                // One thread, and one channel only, give the same samples.
                let mut one = vec![0u16; layout.samples()];
                let got = decode_record(&scratch, format, Channels::Both, &mut one, 1).unwrap();
                assert!(one == out && got == PairEntry { offset: 0, other: Some((pair.len, records[second].len() as u64)), ..*pair }, "{profile:?}: one thread");
                let mut second_only = vec![5u16; layout.samples()];
                decode_record(&scratch, format, Channels::One(1), &mut second_only, 0).unwrap();
                assert!(second_only[..per].iter().all(|&v| v == 5) && second_only[per..] == out[per..], "{profile:?}: channel 1 alone");
            }
            // The two sensors' pictures differ, and so do neighbouring stripes'.
            let rows = frame.width as usize * 32;
            assert!(built[1].0 != built[2].0 && built[1].0[..rows] != built[1].0[2 * rows..3 * rows]);
            assert!(built[1].0.iter().all(|v| *v <= 1023));
            // A stopped walk says so.
            assert!(scan_pairs(&take, |_, _| false).unwrap().is_none());
            // Counts that are not the closing record's are refused: of pairs, and of a sensor's frames.
            write_eye_take(&dir, frame, fps, (500, 503), 2, [3, 5], &records);
            assert!(scan_pairs(&read_take(&dir).unwrap().unwrap(), |_, _| true).unwrap_err().to_string().contains("pairs"));
            write_eye_take(&dir, frame, fps, (500, 503), 3, [3, 4], &records);
            assert!(scan_pairs(&read_take(&dir).unwrap().unwrap(), |_, _| true).unwrap_err().to_string().contains("frames of its two sensors"));
            // A pair outside the window, and a capture cut inside its last record.
            write_eye_take(&dir, frame, fps, (500, 502), 3, [3, 5], &records);
            assert!(scan_pairs(&read_take(&dir).unwrap().unwrap(), |_, _| true).unwrap_err().to_string().contains("outside"));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// What two sensors' frames pair as: the nearest start within a
    /// millisecond, each frame once.
    #[test]
    fn frames_pair_by_when_their_exposures_start() {
        let frame = |hardware: u64, exposure_ns: u64| EyeRecord { hardware, exposure_ns, shutter_ns: 1_000, offset: hardware * 4096, len: 4096 };
        let ms = 1_000_000u64;
        // Sensor 1 starts two frames early and ends one early; its numbers are 300 ahead.
        let first: Vec<EyeRecord> = (10..15).map(|n| frame(n, n * 33 * ms)).collect();
        let second: Vec<EyeRecord> = (8..14).map(|n| frame(300 + n, n * 33 * ms - 31_000)).collect();
        let pairs = pair_eye_records([first.clone(), second.clone()], Profile::EyeNative).unwrap();
        assert_eq!(pairs.iter().map(|p| (p.request, p.other.unwrap().0 / 4096)).collect::<Vec<_>>(), vec![(10, 310), (11, 311), (12, 312), (13, 313)]);
        assert_eq!(pairs[0].exposure_ns, [330 * ms, 330 * ms - 31_000]);
        // In any order of the lists.
        let mut shuffled = [first.clone(), second.clone()];
        shuffled[0].reverse();
        shuffled[1].swap(0, 4);
        assert_eq!(pair_eye_records(shuffled, Profile::EyeNative).unwrap(), pairs);
        // A millisecond apart is a pair, more is none.
        let near = pair_eye_records([vec![frame(1, 50 * ms)], vec![frame(9, 51 * ms)]], Profile::EyeNative).unwrap();
        assert_eq!(near.len(), 1);
        assert!(pair_eye_records([vec![frame(1, 50 * ms)], vec![frame(9, 51 * ms + 1)]], Profile::EyeNative).unwrap().is_empty());
        // The nearer of two candidates is taken, before or after.
        let two = vec![frame(9, 50 * ms - 400_000), frame(10, 50 * ms + 300_000)];
        assert_eq!(pair_eye_records([vec![frame(1, 50 * ms)], two], Profile::EyeNative).unwrap()[0].other, Some((10 * 4096, 4096)));
        // A frame number twice, a frame that does not start after the one before, two frames for one partner: refused.
        assert!(pair_eye_records([vec![frame(1, 50 * ms), frame(1, 83 * ms)], vec![]], Profile::EyeNative).unwrap_err().to_string().contains("twice"));
        assert!(pair_eye_records([vec![frame(1, 50 * ms), frame(2, 50 * ms)], vec![]], Profile::EyeNative).unwrap_err().to_string().contains("does not start after"));
        let close_two = vec![frame(1, 50 * ms), frame(2, 50 * ms + 600_000)];
        assert!(pair_eye_records([close_two, vec![frame(9, 50 * ms + 300_000)]], Profile::EyeNative).unwrap_err().to_string().contains("two frames"));
        // A frame number beyond what a pair goes by.
        assert!(pair_eye_records([vec![frame(1 << 40, 50 * ms)], vec![frame(9, 50 * ms)]], Profile::EyeNative).is_err());
    }

    /// What does not belong in a record for one eye is refused: header
    /// fields of another frame or coder, lengths that do not add up, a
    /// checksum that is not the stripes', a pair that is not one record of
    /// each sensor.
    #[test]
    fn a_record_for_one_eye_is_checked() {
        let frame = EYE_FRAMES[2];
        let format = eyes_at(frame, 60);
        let (profile, layout) = (Profile::EyeWideNear, format.layout);
        let (_, first) = eye_record(profile, frame, 0, 0, 70, 5_000_000, 3_000_000);
        let (_, second) = eye_record(profile, frame, 1, 1, 90, 5_020_000, 3_000_000);
        let pair = [first.clone(), second.clone()].concat();
        let mut out = vec![0u16; layout.samples()];
        let got = decode_record(&pair, format, Channels::Both, &mut out, 0).unwrap();
        assert_eq!((got.request, got.exposure_ns, got.shutter_ns), (70, [5_000_000, 5_020_000], [3_000_000; 2]));
        // The first record with one field changed, ahead of the second.
        let with = |o: usize, v: u32| { let mut r = first.clone(); r[o..o + 4].copy_from_slice(&v.to_le_bytes()); [r, second.clone()].concat() };
        for (what, at, value) in [
            ("version", 8, 1), ("width", 32, 4612), ("height", 36, 3840), ("coded width", 40, 4640), ("stripes", 44, 125),
            ("step", 48, 1), ("lower threshold", 52, 0), ("upper threshold", 56, 1023), ("predictor", 80, 3), ("block coder", 84, 4),
            ("readout mode", 88, 61), ("sensor", 20, 2), ("record length", 12, first.len() as u32 + 1), ("record length", 12, 4096),
            ("stripes' length", 16, u32::from_le_bytes(first[16..20].try_into().unwrap()) + 4), ("a stripe's length", 128, 1 << 30), ("a stripe's length", 128, 0),
        ] {
            assert!(decode_record(&with(at, value), format, Channels::Both, &mut out, 0).is_err(), "{what}");
        }
        // The header's length of a stripe and the stripe's own must agree (the two changed alike keep the sum; the stream then fails).
        let stripe0 = u32::from_le_bytes(first[128..132].try_into().unwrap());
        let stripe1 = u32::from_le_bytes(first[132..136].try_into().unwrap());
        let mut moved = first.clone();
        moved[128..132].copy_from_slice(&(stripe0 + 8).to_le_bytes());
        moved[132..136].copy_from_slice(&(stripe1 - 8).to_le_bytes());
        let e = decode_record(&[moved, second.clone()].concat(), format, Channels::Both, &mut out, 0).unwrap_err();
        assert!(e.to_string().contains("not the length"), "{e}");
        // A checksum that is not the stripes': told as damage, on one thread or many, and not looked at for a channel that is not wanted.
        for threads in [0, 1] {
            let e = decode_record(&with(60, 0x1234_5678), format, Channels::Both, &mut out, threads).unwrap_err();
            assert!(e.to_string().contains("checksum"), "{e}");
        }
        assert!(decode_record(&with(60, 0x1234_5678), format, Channels::One(1), &mut out, 0).is_ok());
        // A byte of the stripes changed: the checksum or the stream tells.
        let mut flipped = pair.clone();
        flipped[BLOCK as usize + 40_000] ^= 0x10;
        assert!(decode_record(&flipped, format, Channels::Both, &mut out, 0).is_err());
        // Cut short, in the first record or the second.
        assert!(decode_record(&pair[..first.len() - 1], format, Channels::Both, &mut out, 0).is_err());
        assert!(decode_record(&pair[..pair.len() - 1], format, Channels::Both, &mut out, 0).is_err());
        // Two records of one sensor, or the sensors the other way round, are no pair.
        assert!(decode_record(&[first.clone(), first.clone()].concat(), format, Channels::Both, &mut out, 0).is_err());
        assert!(decode_record(&[second.clone(), first.clone()].concat(), format, Channels::Both, &mut out, 0).is_err());
        // Another layout does not take such a record, nor this one another's.
        assert!(decode_record(&pair, Format::paired(60), Channels::Both, &mut vec![0u16; 2 * EYE_SAMPLES], 0).is_err());
        assert!(decode_record(&pair, eyes_at(EYE_FRAMES[0], 30), Channels::Both, &mut vec![0u16; Layout::EyeRecords(EYE_FRAMES[0]).samples()], 0).is_err());
        let (_, paired) = frame_record(Profile::Frame10, frame_of(3968), 12, [10, 20]);
        assert!(decode_record(&paired, format, Channels::Both, &mut out, 0).is_err());

        // A take: two kinds of record in one capture, a record out of its place, a stamp of zero, a capture cut short.
        let dir = temp_dir("eye-checks");
        let scan = |records: &[Vec<u8>], pairs: usize, eyes: [usize; 2]| {
            write_eye_take(&dir, EYE_FRAMES[0], 30, (1, 100), pairs, eyes, records);
            scan_pairs(&read_take(&dir).unwrap().unwrap(), |_, _| true)
        };
        let square = |profile: Profile, eye: usize, sequence: u32, stamp: u64| eye_record(profile, EYE_FRAMES[0], eye, sequence, 40 + sequence as u64, stamp, 1_000).1;
        let (a, b) = (square(Profile::EyeScaled, 0, 0, 7_000_000), square(Profile::EyeScaled, 1, 1, 7_010_000));
        assert_eq!(scan(&[a.clone(), b.clone()], 1, [1, 1]).unwrap().unwrap().len(), 1);
        let other_kind = square(Profile::EyeNative, 1, 1, 7_010_000);
        assert!(scan(&[a.clone(), other_kind], 1, [1, 1]).unwrap_err().to_string().contains("another kind"));
        assert!(scan(&[b.clone(), a.clone()], 1, [1, 1]).unwrap_err().to_string().contains("out of its place"));
        assert!(scan(&[a.clone(), square(Profile::EyeScaled, 1, 1, 0)], 1, [1, 1]).unwrap_err().to_string().contains("no exposure time"));
        write_eye_take(&dir, EYE_FRAMES[0], 30, (1, 100), 1, [1, 1], &[a.clone(), b[..b.len() - 4096].to_vec()]);
        assert!(scan_pairs(&read_take(&dir).unwrap().unwrap(), |_, _| true).unwrap_err().to_string().contains("truncated"));
        // What the take says of its frames: a size this module does not know, a frame rate the frame is not recorded at, another depth, no size.
        let clip = |size: &str, fps: u32, more: &str| std::fs::write(dir.join(CLIP_FILE), format!(r#"{{"layout":"paired-bayer-eye-records-v2","fps_setting":{fps}{size}{more}}}"#)).unwrap();
        clip(r#","size":[5000,5000]"#, 30, "");
        assert!(take_dir(&dir).is_some() && read_take(&dir).unwrap_err().to_string().contains("5000"));
        clip(r#","size":[4608,3968]"#, 30, "");
        assert!(read_take(&dir).unwrap_err().to_string().contains("frame rate"));
        clip(r#","size":[5760,5760]"#, 30, r#","sensor_bits":14"#);
        assert!(read_take(&dir).unwrap_err().to_string().contains("depth"));
        clip("", 30, "");
        assert!(take_dir(&dir).is_none() && read_take(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The take's white balance record: each sensor's neutral from the
    /// gains the camera had found, and nothing for a sensor the record does
    /// not vouch for.
    #[test]
    fn the_white_balance_record_reads() {
        let dir = temp_dir("white");
        let (_, record) = frame_record(Profile::Frame10, frame_of(3968), 5, [10, 20]);
        write_frame_take(&dir, frame_of(3968), 60, (64, 1023), (5, 5), &[record]);
        let read = |text: &str| {
            std::fs::write(dir.join(WHITE_BALANCE_FILE), text).unwrap();
            read_white_balance(&read_take(&dir).unwrap().unwrap())
        };
        // No record: nothing.
        assert_eq!(read_white_balance(&read_take(&dir).unwrap().unwrap()), [None; 2]);
        let eye = |status: &str, gains: &str, neutral: &str| format!(r#"{{"status":"{status}","request":137,"rgb_gains":{gains},"as_shot_neutral":{neutral},"color_temperature_kelvin":3531}}"#);
        let file = |cam0: &str, cam1: &str| format!(r#"{{"schema":"sample-white-balance-v1","status":"ok","channel_order":["R","G","B"],"eyes":{{"CAM0":{cam0},"CAM1":{cam1}}}}}"#);
        let good = eye("ok", "[1.638916015625,1.0,2.543212890625]", "[0.6101593922240429,1.0,0.3932034174906403]");
        let cooler = eye("ok", "[2.0,1.0,1.6]", "[0.5,1.0,0.625]");
        let got = read(&file(&good, &cooler));
        let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12);
        assert!(close(got[0].unwrap(), [1.0 / 1.638916015625, 1.0, 1.0 / 2.543212890625]) && close(got[1].unwrap(), [0.5, 1.0, 0.625]), "{got:?}");
        // A sensor whose status is not ok, whose gains are no numbers or out of range, or whose neutral is not the gains': nothing for it.
        assert_eq!(read(&file(&eye("unavailable", "[2.0,1.0,1.6]", "[0.5,1.0,0.625]"), &cooler)), [None, Some([0.5, 1.0, 0.625])]);
        assert_eq!(read(&file(&cooler, &eye("ok", "[2.0,1.0]", "[0.5,1.0,0.625]")))[1], None);
        assert_eq!(read(&file(&cooler, &eye("ok", "[2.0,1.0,0.0]", "[0.5,1.0,0.625]")))[1], None);
        assert_eq!(read(&file(&cooler, &eye("ok", "[2.0,1.0,100.0]", "[0.5,1.0,0.01]")))[1], None);
        assert_eq!(read(&file(&cooler, &eye("ok", "[2.0,1.0,1.6]", "[0.5,1.0,0.7]")))[1], None);
        // The neutral may be left out: the gains give it.
        assert_eq!(read(&file(&cooler, r#"{"status":"ok","rgb_gains":[2.0,1.0,1.6]}"#))[1], Some([0.5, 1.0, 0.625]));
        // Another kind of record, or no JSON: nothing.
        assert_eq!(read(&file(&good, &cooler).replace("white-balance-v1", "white-balance-v2")), [None; 2]);
        assert_eq!(read("not json"), [None; 2]);
        // The file the take names is the one read.
        std::fs::write(dir.join("wb.json"), file(&cooler, &cooler)).unwrap();
        let mut take = read_take(&dir).unwrap().unwrap();
        assert_eq!(take.white_balance_file, None);
        take.white_balance_file = Some("wb.json".into());
        assert_eq!(read_white_balance(&take), [Some([0.5, 1.0, 0.625]); 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Records written by an independent packer against the samples an
    /// independent reader took from them (skipped unless
    /// `VR180_EYE_RECORDS` names a folder of them): for each of the five
    /// signatures a file `<signature>.x6b` of one record of each sensor,
    /// and `<signature>-CAM0.u16`, `<signature>-CAM1.u16`, the frames as
    /// 16-bit samples.
    #[test]
    fn independent_records_for_one_eye_decode_to_the_same_samples() {
        let Some(dir) = std::env::var_os("VR180_EYE_RECORDS").map(PathBuf::from) else { return };
        for (profile, frame, fps) in [
            (Profile::EyeScaled, EYE_FRAMES[0], 30u32), (Profile::EyeScaledNear, EYE_FRAMES[1], 30), (Profile::EyeNative, EYE_FRAMES[0], 30),
            (Profile::EyeNativeNear, EYE_FRAMES[1], 30), (Profile::EyeWideNear, EYE_FRAMES[2], 60),
        ] {
            let name = String::from_utf8_lossy(&profile.signature()[..7]).into_owned();
            let format = eyes_at(frame, fps);
            let pair = std::fs::read(dir.join(format!("{name}.x6b"))).unwrap();
            let mut out = vec![0u16; format.layout.samples()];
            let started = std::time::Instant::now();
            let got = decode_record(&pair, format, Channels::Both, &mut out, 0).unwrap();
            let took = started.elapsed();
            assert_eq!(got.profile, profile);
            for (channel, eye) in out.chunks(format.layout.eye_samples()).enumerate() {
                let want: Vec<u16> = std::fs::read(dir.join(format!("{name}-CAM{channel}.u16"))).unwrap()
                    .chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                assert!(eye == &want[..], "{name}: sensor {channel} is not the independent reader's frame");
            }
            println!("{name} ({profile:?}): frame {} of both sensors as the independent reader has them, {:.1} MB in {took:.1?}", got.request, pair.len() as f64 / 1e6);
        }
    }
}
