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

use crate::{Error, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Frame side of one eye.
pub const EYE_SIDE: usize = 3840;
/// Rows of one tile.
pub const TILE_ROWS: usize = 160;
/// Tiles of one eye.
pub const TILES_PER_EYE: usize = 24;
/// Samples of one tile.
pub const TILE_SAMPLES: usize = EYE_SIDE * TILE_ROWS;
/// Samples of one eye.
pub const EYE_SAMPLES: usize = EYE_SIDE * EYE_SIDE;
/// Header length, and the boundary every record and tile is padded to.
pub const BLOCK: u64 = 4096;
/// The header's threshold field.
const THRESHOLD: u32 = 13472;
/// Bayer pattern of the samples, row-major 2×2 (0 = R, 1 = G, 2 = B).
pub const CFA: [u8; 4] = [1, 2, 0, 1];
/// Sensor black level.
pub const BLACK_LEVEL: u16 = 64;
/// Sensor white level (saturated samples read exactly this).
pub const WHITE_LEVEL: u16 = 15408;

pub const CLIP_FILE: &str = "clip.json";
pub const SEAL_FILE: &str = "seal.json";
pub const CAPTURE_FILE: &str = "capture.x6b";
pub const CALIBRATION_FILE: &str = "calibration.json";
pub const GYRO_FILE: &str = "gyro.csv";
pub const GYRO_INFO_FILE: &str = "gyro-info.json";
/// The index of a folder that holds a take's frames as DNG files (one line
/// per pair) in place of the capture.
pub const FRAMES_FILE: &str = "frames.jsonl";
const LAYOUT: &str = "paired-bayer-v1";

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
}

impl Profile {
    pub const ALL: [Profile; 5] = [Self::Exact30, Self::Mid50, Self::Raised50, Self::Coarse60, Self::Raised60];

    /// The record signature.
    pub fn signature(self) -> &'static [u8; 8] {
        match self {
            Self::Exact30 => b"L1024___",
            Self::Mid50 => b"X6M50ABC",
            Self::Raised50 => b"X6M50AB2",
            Self::Coarse60 => b"X6QBLK05",
            Self::Raised60 => b"X6QBLK0B",
        }
    }

    /// The frame rate setting this signature belongs to.
    pub fn fps(self) -> u32 {
        match self {
            Self::Exact30 => 30,
            Self::Mid50 | Self::Raised50 => 50,
            Self::Coarse60 | Self::Raised60 => 60,
        }
    }

    /// The profile of a signature recorded at `fps`; `None` for an unknown
    /// signature or one that belongs to another frame rate.
    pub fn from_signature(signature: &[u8], fps: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.signature()[..] == *signature && p.fps() == fps)
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
        }
    }

    /// Symbol → 14-bit level, in symbol order.
    pub fn levels(self) -> &'static [u16] {
        static TABLES: OnceLock<[Vec<u16>; 5]> = OnceLock::new();
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
        self == Self::Exact30
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
    /// First and last request number the take may hold (both included).
    pub request_window: (u32, u32),
    /// Pairs the camera received (may exceed `pairs` when the card fell behind).
    pub received_pairs: u64,
    /// Complete pairs in the capture.
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
    pub fps_setting: u32,
    pub width: u32,
    pub height: u32,
    pub clock_start: Option<ClockAnchor>,
    /// Sound file inside the folder (`clip.json` → `audio_file`).
    pub audio_file: Option<String>,
    /// Its timing record (`clip.json` → `audio_timing`).
    pub audio_timing: Option<String>,
    pub seal: Seal,
    pub capture_bytes: u64,
}

impl Take {
    pub fn capture_path(&self) -> PathBuf {
        self.dir.join(CAPTURE_FILE)
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
/// the folder has no `clip.json` with the paired Bayer layout.
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
    (v.get("layout").and_then(|l| l.as_str()) == Some(LAYOUT)).then_some(dir)
}

/// Read a take's description. `Err` for a folder that is no take at all;
/// `Ok(Err(problem))` for a take that cannot be opened as it is.
pub fn read_take(dir: &Path) -> Result<std::result::Result<Take, TakeProblem>> {
    let clip: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(CLIP_FILE))?)?;
    if clip.get("layout").and_then(|l| l.as_str()) != Some(LAYOUT) {
        return Err(err(format!("{}: not a {LAYOUT} take", dir.display())));
    }
    let fps_setting = clip.get("fps_setting").and_then(|v| v.as_u64()).ok_or_else(|| err("clip.json has no fps_setting"))? as u32;
    let size: Vec<u64> = clip.get("size").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
    if size != [EYE_SIDE as u64, EYE_SIDE as u64] {
        return Err(err(format!("unsupported frame size {size:?}")));
    }
    if !Profile::ALL.iter().any(|p| p.fps() == fps_setting) {
        return Err(err(format!("unsupported frame rate setting {fps_setting}")));
    }
    let text = |k: &str| clip.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(str::to_owned);
    let capture = dir.join(CAPTURE_FILE);
    let Ok(meta) = std::fs::metadata(&capture) else {
        return Ok(Err(if dir.join(FRAMES_FILE).is_file() { TakeProblem::FramesOnly } else { TakeProblem::NoCapture }));
    };
    let Ok(seal_bytes) = std::fs::read(dir.join(SEAL_FILE)) else { return Ok(Err(TakeProblem::NotSealed)) };
    let seal: serde_json::Value = serde_json::from_slice(&seal_bytes)?;
    let u = |k: &str| seal.get(k).and_then(|v| v.as_u64());
    let window: Vec<u64> = seal.get("request_window").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
    let (Some(seal_fps), Some(pairs), Some(bytes), &[lo, hi]) = (u("fps_setting"), u("pairs"), u("bytes"), window.as_slice()) else {
        return Err(err("seal.json is incomplete"));
    };
    let received = u("received_pairs").unwrap_or(pairs);
    if seal_fps as u32 != fps_setting {
        return Err(err(format!("clip.json says {fps_setting} fps, seal.json {seal_fps}")));
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
        fps_setting,
        width: EYE_SIDE as u32,
        height: EYE_SIDE as u32,
        clock_start: clip.get("clock_start").and_then(ClockAnchor::from_json),
        audio_file: text("audio_file"),
        audio_timing: text("audio_timing"),
        seal: Seal { fps_setting, request_window: (lo as u32, hi as u32), received_pairs: received, pairs, bytes },
        capture_bytes: meta.len(),
    }))
}

/// One pair of the capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairEntry {
    /// The camera's request number (the pair's identity and order).
    pub request: u32,
    /// Byte offset of the record's header.
    pub offset: u64,
    /// Length of the whole record, header included.
    pub len: u64,
    pub profile: Profile,
    /// Exposure start of each channel, nanoseconds on the take's counter.
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
    tile_len: [u32; 2 * TILES_PER_EYE],
}

impl Header {
    fn parse(h: &[u8], fps: u32, at: u64) -> Result<Self> {
        if h.len() < BLOCK as usize {
            return Err(err(format!("truncated header at byte {at}")));
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let u64_at = |o: usize| u64::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3], h[o + 4], h[o + 5], h[o + 6], h[o + 7]]);
        let profile = Profile::from_signature(&h[..8], fps).ok_or_else(|| {
            let shown: String = h[..8].iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect();
            err(format!("unsupported signature \"{shown}\" for {fps} fps at byte {at}"))
        })?;
        if (u32_at(8), u32_at(16), u32_at(20), u32_at(24), u32_at(28))
            != (BLOCK as u32, EYE_SIDE as u32, EYE_SIDE as u32, TILES_PER_EYE as u32, THRESHOLD)
        {
            return Err(err(format!("invalid header at byte {at}")));
        }
        let mut tile_len = [0u32; 2 * TILES_PER_EYE];
        for (i, len) in tile_len.iter_mut().enumerate() {
            *len = u32_at(48 + 4 * i);
            if *len <= 4 || *len as usize > TILE_SAMPLES * 4 {
                return Err(err(format!("invalid tile size at byte {at}")));
            }
        }
        Ok(Self { profile, request: u32_at(12), exposure_ns: [u64_at(32), u64_at(40)], tile_len })
    }

    /// Length of the record, header and padding included.
    fn record_len(&self) -> u64 {
        BLOCK + self.tile_len.iter().map(|&n| pad(n as u64)).sum::<u64>()
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
/// signature against the frame rate, the header fields, every record inside
/// the file, request numbers inside the window and used once, positive
/// stamps, and the number of pairs against the seal.
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
        let h = Header::parse(&head, take.fps_setting, at)?;
        if h.request < lo || h.request > hi {
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

/// Request numbers of the window the capture does not hold, in order.
pub fn missing_requests(take: &Take, pairs: &[PairEntry]) -> Vec<u32> {
    let (lo, hi) = take.seal.request_window;
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
    One(usize),
}

/// Read one record whole (`entry.len` bytes) into `buf`.
pub fn read_record(file: &mut File, entry: &PairEntry, buf: &mut Vec<u8>) -> Result<()> {
    buf.resize(entry.len as usize, 0);
    file.seek(SeekFrom::Start(entry.offset))?;
    file.read_exact(buf).map_err(|_| err(format!("truncated pair at byte {}", entry.offset)))?;
    Ok(())
}

/// Decode a record read by [`read_record`] into `out`: channel 0's
/// 3840×3840 samples, then channel 1's (`2 × EYE_SAMPLES` values, 14-bit).
/// Tiles decode on `threads` threads (0 = one per core, at most one per tile).
pub fn decode_record(record: &[u8], fps: u32, channels: Channels, out: &mut [u16], threads: usize) -> Result<PairEntry> {
    if out.len() != 2 * EYE_SAMPLES {
        return Err(err("output buffer is not two eyes"));
    }
    let h = Header::parse(record, fps, 0)?;
    if (record.len() as u64) < h.record_len() {
        return Err(err("truncated pair"));
    }
    // Tile payloads, in file order.
    let mut tiles: Vec<(usize, &[u8])> = Vec::with_capacity(2 * TILES_PER_EYE);
    let mut at = BLOCK as usize;
    for (i, &n) in h.tile_len.iter().enumerate() {
        tiles.push((i, &record[at..at + n as usize]));
        at += pad(n as u64) as usize;
    }
    let wanted = |tile: usize| match channels {
        Channels::Both => true,
        Channels::One(c) => tile / TILES_PER_EYE == c,
    };
    let jobs: Vec<(&[u8], &mut [u16])> = tiles.iter()
        .zip(out.chunks_mut(TILE_SAMPLES))
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
            decode_tile(bytes, profile, dst)?;
        }
    } else {
        let queue = std::sync::Mutex::new(jobs.into_iter());
        let failed: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let job = queue.lock().ok().and_then(|mut q| q.next());
                    let Some((bytes, dst)) = job else { break };
                    if let Err(e) = decode_tile(bytes, profile, dst) {
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
/// 3840×160 samples.
fn decode_tile(bytes: &[u8], profile: Profile, dst: &mut [u16]) -> Result<()> {
    debug_assert_eq!(dst.len(), TILE_SAMPLES);
    let main = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if main > bytes.len() - 4 {
        return Err(err("invalid predictor length"));
    }
    let levels = profile.levels();
    unpack_symbols(&bytes[4..4 + main], dst, profile.max_width(), levels.len() as i32)?;
    apply_levels(&bytes[4 + main..], levels, dst)
}

/// The predictor stream of one tile → symbols in `dst`.
fn unpack_symbols(stream: &[u8], dst: &mut [u16], max_width: usize, count: i32) -> Result<()> {
    const W: usize = EYE_SIDE;
    let mut p = 0usize;
    for y in 0..TILE_ROWS {
        // The row above the one being written, two rows up (same Bayer site).
        let (done, rest) = dst.split_at_mut(y * W);
        let row = &mut rest[..W];
        let up: Option<&[u16]> = (y >= 2).then(|| &done[(y - 2) * W..(y - 1) * W]);
        for x in (0..W).step_by(64) {
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
                    let mut w = 0u64;
                    for k in 0..whole {
                        w |= (block[k * 8 + lane] as u64) << (8 * k);
                    }
                    if b & 1 == 1 {
                        w |= (((block[whole * 8 + lane / 2] >> (4 * (lane & 1))) & 15) as u64) << (8 * whole);
                    }
                    *word = w;
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
pub fn decode_pair(file: &mut File, entry: &PairEntry, fps: u32, channels: Channels, record: &mut Vec<u8>, out: &mut [u16], threads: usize) -> Result<()> {
    read_record(file, entry, record)?;
    let got = decode_record(record, fps, channels, out, threads)?;
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
    let text = std::fs::read_to_string(dir.join(GYRO_FILE))?;
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

    fn encode_tile(profile: Profile, samples: &[u16]) -> Vec<u8> {
        const W: usize = EYE_SIDE;
        let coded: Vec<(u16, u8)> = samples.iter().map(|&v| symbol_of(profile, v)).collect();
        let mut main = Vec::new();
        for y in 0..TILE_ROWS {
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
        let tiles: Vec<Vec<u8>> = samples.chunks(TILE_SAMPLES).map(|t| encode_tile(profile, t)).collect();
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
    }

    #[test]
    fn a_signature_belongs_to_one_frame_rate() {
        assert_eq!(Profile::from_signature(b"L1024___", 30), Some(Profile::Exact30));
        assert_eq!(Profile::from_signature(b"X6QBLK0B", 60), Some(Profile::Raised60));
        assert_eq!(Profile::from_signature(b"X6QBLK0B", 50), None);
        assert_eq!(Profile::from_signature(b"X6M50AB2", 60), None);
        assert_eq!(Profile::from_signature(b"UNKNOWN!", 30), None);
    }

    #[test]
    fn every_profile_decodes_what_was_stored() {
        for (n, profile) in Profile::ALL.into_iter().enumerate() {
            let samples = test_samples(7 + n as u32);
            let record = encode_record(profile, 100 + n as u32, [1_000_000, 999_971], &samples);
            let mut out = vec![0u16; 2 * EYE_SAMPLES];
            let got = decode_record(&record, profile.fps(), Channels::Both, &mut out, 0).unwrap();
            assert_eq!((got.request, got.exposure_ns, got.len as usize), (100 + n as u32, [1_000_000, 999_971], record.len()));
            let want: Vec<u16> = samples.iter().map(|&v| stored_value(profile, v)).collect();
            assert_eq!(fnv(&out), fnv(&want), "{profile:?}");
            if profile.is_exact() {
                assert_eq!(fnv(&out), fnv(&samples), "30 fps keeps every value");
            }
            // One thread, and one channel only, give the same samples.
            let mut one = vec![0u16; 2 * EYE_SAMPLES];
            decode_record(&record, profile.fps(), Channels::Both, &mut one, 1).unwrap();
            assert_eq!(fnv(&one), fnv(&out));
            let mut right = vec![7u16; 2 * EYE_SAMPLES];
            decode_record(&record, profile.fps(), Channels::One(1), &mut right, 0).unwrap();
            assert!(right[..EYE_SAMPLES].iter().all(|&v| v == 7));
            assert_eq!(fnv(&right[EYE_SAMPLES..]), fnv(&out[EYE_SAMPLES..]));
        }
    }

    #[test]
    fn a_damaged_record_is_refused() {
        let profile = Profile::Coarse60;
        let record = encode_record(profile, 5, [10, 20], &test_samples(1));
        let mut out = vec![0u16; 2 * EYE_SAMPLES];
        let mut decode = |r: &[u8], fps: u32| decode_record(r, fps, Channels::Both, &mut out, 0);
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
        decode_pair(&mut file, &pairs[0], 60, Channels::Both, &mut record, &mut out, 0).unwrap();
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
        take.audio_timing = Some("audio-info.json".into());
        std::fs::write(dir.join("audio-info.json"), r#"{"version":2,"file":"audio.aac","source":"ambisonic","channels":4,"codec_delay_samples":2048,"trailing_padding_samples":384,"sample_rate_hz":48000,"sample_frames":1142400,"first_sample_counter_ns":372569353830,"anchors":[{"sample_index":960,"counter_ns":372589353830}]}"#).unwrap();
        let t = read_audio_timing(&take).unwrap();
        assert_eq!((t.file.as_str(), t.channels, t.sample_rate_hz, t.sample_frames, t.codec_delay_samples), ("audio.aac", 4, 48000, 1142400, 2048));
        assert_eq!((t.first_sample_ns, t.anchors.len(), t.ambisonic), (372569353830, 1, true));
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
            decode_pair(&mut file, entry, take.fps_setting, Channels::Both, &mut record, &mut out, 0).unwrap();
            assert_eq!(fnv(&out), want, "{take_name} request {request}");
        }
    }
}
