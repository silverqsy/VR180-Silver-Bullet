//! A raw take's frames as the sensor recorded them: one Bayer DNG per eye
//! per frame, nothing of this app's rendering in them.
//!
//! [`export_sensor_dng_sequence`] writes a take (or a stretch of it) into a
//! folder:
//!
//! ```text
//! <folder>/CAM0/000000094.dng     channel 0 of request 94
//! <folder>/CAM1/000000094.dng     channel 1 of the same request
//! <folder>/frames.jsonl           one line per pair: request, exposure starts, files
//! <folder>/calibration.json …     the take's own small files, copied
//! ```
//!
//! A file is named by the camera's request number (nine digits), the pair's
//! identity in the take, so a frame keeps its name whatever stretch is
//! written and a request the camera never saved leaves its number out.
//!
//! ## The file
//!
//! A little-endian TIFF with one IFD: 3840×3840, one 14-bit sample per
//! site, `PhotometricInterpretation` 32803 (colour filter array), the
//! take's pattern, black and white levels, tiles of 3840×160 (the capture's
//! own) as lossless JPEG ([`crate::ljpeg::encode_tile_cfa_into`]), and the
//! camera's colour description (`ColorMatrix1` under D65, `AsShotNeutral`).
//! The sample values are the decoded ones, untouched: codes above the white
//! level stay as they are. `ImageDescription` holds the channel, the
//! request number and the exposure start.
//!
//! The colour description is the take's reference one (a raw take records
//! no white balance of its own), the same this app develops with, so a raw
//! converter opening a file starts from the colours the app shows.
//!
//! A take of one lens writes its one channel's folder, and its files are
//! that take's frames: 7680×7680, 10-bit samples, tiles of 7680×320.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use vr180_fisheye::x6b::{self, Channels, PairEntry, EYE_SIDE, TILES_PER_EYE, TILE_ROWS};

use crate::fisheye_export::ExportProgress;
use crate::raw_color::RawCamera;
use crate::{Error, Result};

/// Bits of a paired take's sample.
pub const BITS: u8 = 14;

/// The mosaic a file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mosaic {
    /// Side of the frame.
    pub side: usize,
    /// Rows of one of its tiles (the capture's own).
    pub tile_rows: usize,
    /// Bits of a sample.
    pub bits: u8,
    pub cfa: [u8; 4],
    pub black_level: u16,
    pub white_level: u16,
}

impl Mosaic {
    /// One channel of a paired take.
    pub const PAIRED: Mosaic = Mosaic {
        side: EYE_SIDE, tile_rows: TILE_ROWS, bits: BITS,
        cfa: x6b::CFA, black_level: x6b::BLACK_LEVEL, white_level: x6b::WHITE_LEVEL,
    };

    /// One channel of `take`: the bits its largest sample needs.
    pub fn of(take: &x6b::Take) -> Self {
        let top = take.sample_max();
        Mosaic {
            side: take.layout.side(), tile_rows: take.layout.tile_rows(),
            bits: (16 - top.leading_zeros()) as u8,
            cfa: take.cfa, black_level: take.black_level, white_level: take.white_level,
        }
    }

    fn samples(&self) -> usize {
        self.side * self.side
    }
}
/// `UniqueCameraModel` of the files: a name of its own, so a raw converter
/// takes the colour description from the file.
pub const UNIQUE_CAMERA_MODEL: &str = "Insta360 X6 Raw Video";
/// The index of a written folder.
pub const FRAMES_FILE: &str = x6b::FRAMES_FILE;
/// Workers at most: a pair in flight holds about 100 MB.
const MAX_WORKERS: usize = 12;

/// Folder of one channel's files.
pub fn channel_dir(channel: usize) -> String {
    format!("CAM{channel}")
}

/// File of one channel of one request, relative to the folder written.
pub fn frame_file(channel: usize, request: u32) -> String {
    format!("{}/{request:09}.dng", channel_dir(channel))
}

/// A value as a fraction with a denominator of at most a million (the
/// closest one, by continued fractions).
fn fraction(v: f64) -> (i64, i64) {
    let (mut p0, mut q0, mut p1, mut q1) = (0i64, 1i64, 1i64, 0i64);
    let mut x = v;
    for _ in 0..32 {
        let a = x.floor();
        let (p2, q2) = (a as i64 * p1 + p0, a as i64 * q1 + q0);
        if q2 > 1_000_000 {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
        let frac = x - a;
        if frac.abs() < 1e-12 {
            break;
        }
        x = 1.0 / frac;
    }
    if q1 == 0 { (v.round() as i64, 1) } else { (p1, q1) }
}

const BYTE: u16 = 1;
const ASCII: u16 = 2;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const SRATIONAL: u16 = 10;

/// The fields of the one IFD, by tag: type, count, value bytes.
#[derive(Default)]
struct Fields(BTreeMap<u16, (u16, u32, Vec<u8>)>);

impl Fields {
    fn shorts(&mut self, tag: u16, v: &[u16]) {
        self.0.insert(tag, (SHORT, v.len() as u32, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    fn longs(&mut self, tag: u16, v: &[u32]) {
        self.0.insert(tag, (LONG, v.len() as u32, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    fn bytes(&mut self, tag: u16, v: &[u8]) {
        self.0.insert(tag, (BYTE, v.len() as u32, v.to_vec()));
    }
    fn text(&mut self, tag: u16, v: &str) {
        let mut data = v.as_bytes().to_vec();
        data.push(0);
        self.0.insert(tag, (ASCII, data.len() as u32, data));
    }
    fn fractions(&mut self, tag: u16, signed: bool, v: &[f64]) {
        let mut data = Vec::with_capacity(v.len() * 8);
        for &x in v {
            let (n, d) = fraction(x);
            if signed {
                data.extend_from_slice(&(n as i32).to_le_bytes());
            } else {
                data.extend_from_slice(&(n.max(0) as u32).to_le_bytes());
            }
            data.extend_from_slice(&(d as u32).to_le_bytes());
        }
        self.0.insert(tag, (if signed { SRATIONAL } else { RATIONAL }, v.len() as u32, data));
    }

    /// The file's head: the TIFF header, the IFD and the values that do not
    /// fit an entry, padded to a multiple of four bytes.
    fn head(&self) -> Vec<u8> {
        let n = self.0.len();
        let values_at = 8 + 2 + 12 * n + 4;
        let mut out = Vec::with_capacity(values_at + 512);
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        let mut values: Vec<u8> = Vec::new();
        for (&tag, (kind, count, data)) in &self.0 {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            if data.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..data.len()].copy_from_slice(data);
                out.extend_from_slice(&inline);
            } else {
                out.extend_from_slice(&((values_at + values.len()) as u32).to_le_bytes());
                values.extend_from_slice(data);
                values.resize(values.len().next_multiple_of(4), 0);
            }
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(out.len(), values_at);
        out.extend_from_slice(&values);
        out
    }
}

/// What a file says of its frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTag {
    /// Channel of the pair (0 or 1).
    pub channel: usize,
    pub request: u32,
    /// Exposure start, nanoseconds on the take's counter.
    pub exposure_ns: u64,
}

impl FrameTag {
    /// The `ImageDescription` of the file.
    pub fn description(&self) -> String {
        format!("CAM{} request={} exposure_start_ns={}", self.channel, self.request, self.exposure_ns)
    }

    /// Read a description back.
    pub fn parse(description: &str) -> Option<Self> {
        let mut words = description.split_whitespace();
        let channel = words.next()?.strip_prefix("CAM")?.parse().ok()?;
        let request = words.next()?.strip_prefix("request=")?.parse().ok()?;
        let exposure_ns = words.next()?.strip_prefix("exposure_start_ns=")?.parse().ok()?;
        Some(Self { channel, request, exposure_ns })
    }
}

/// Scratch space of one writer, reused from frame to frame.
#[derive(Default)]
pub struct EyeScratch {
    tile: Vec<u8>,
    body: Vec<u8>,
}

/// One channel's frame (`mosaic`'s samples) as a DNG file: the head and
/// the tile data, to be written one after the other.
pub fn eye_dng<'a>(samples: &[u16], mosaic: &Mosaic, tag: FrameTag, camera: &RawCamera, scratch: &'a mut EyeScratch) -> (Vec<u8>, &'a [u8]) {
    assert_eq!(samples.len(), mosaic.samples(), "one eye's samples");
    let fields = |offsets: &[u32], counts: &[u32]| {
        let mut f = Fields::default();
        f.longs(254, &[0]);
        f.longs(256, &[mosaic.side as u32]);
        f.longs(257, &[mosaic.side as u32]);
        f.shorts(258, &[mosaic.bits as u16]);
        f.shorts(259, &[7]);
        f.shorts(262, &[32803]);
        f.text(270, &tag.description());
        f.text(271, "Arashi Vision");
        f.text(272, "Insta360 X6");
        f.shorts(274, &[1]);
        f.shorts(277, &[1]);
        f.shorts(284, &[1]);
        f.text(305, concat!("VR180 Silver Bullet ", env!("CARGO_PKG_VERSION")));
        f.longs(322, &[mosaic.side as u32]);
        f.longs(323, &[mosaic.tile_rows as u32]);
        f.longs(324, offsets);
        f.longs(325, counts);
        f.shorts(33421, &[2, 2]);
        f.bytes(33422, &mosaic.cfa);
        f.bytes(50706, &[1, 4, 0, 0]);
        f.bytes(50707, &[1, 1, 0, 0]);
        f.text(50708, UNIQUE_CAMERA_MODEL);
        f.shorts(50714, &[mosaic.black_level]);
        f.longs(50717, &[mosaic.white_level as u32]);
        let m = camera.color_matrix;
        f.fractions(50721, true, &[m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]);
        f.fractions(50728, false, &camera.as_shot_neutral);
        f.shorts(50778, &[21]);
        f
    };
    // The head's length does not depend on the tiles' sizes.
    let head_len = fields(&[0; TILES_PER_EYE], &[0; TILES_PER_EYE]).head().len();
    scratch.body.clear();
    let mut offsets = [0u32; TILES_PER_EYE];
    let mut counts = [0u32; TILES_PER_EYE];
    for (i, tile) in samples.chunks_exact(mosaic.side * mosaic.tile_rows).enumerate() {
        crate::ljpeg::encode_tile_cfa_into(tile, mosaic.side, mosaic.tile_rows, mosaic.bits, &mut scratch.tile);
        offsets[i] = (head_len + scratch.body.len()) as u32;
        counts[i] = scratch.tile.len() as u32;
        scratch.body.extend_from_slice(&scratch.tile);
        scratch.body.resize(scratch.body.len().next_multiple_of(4), 0);
    }
    let head = fields(&offsets, &counts).head();
    debug_assert_eq!(head.len(), head_len);
    (head, &scratch.body)
}

/// One pair of a written folder: a line of its index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenPair {
    pub request: u32,
    pub exposure_ns: [u64; 2],
    /// The record's signature in the capture.
    pub signature: String,
    /// Each file written: its channel and its length (a paired take's two
    /// channels; the one channel of a take of one lens).
    pub files: Vec<(usize, u64)>,
}

impl WrittenPair {
    /// The pair's line of `frames.jsonl`.
    pub fn line(&self) -> String {
        serde_json::json!({
            "request": self.request,
            "exposures_ns": self.exposure_ns,
            "signature": self.signature,
            "files": self.files.iter().map(|f| frame_file(f.0, self.request)).collect::<Vec<_>>(),
            "bytes": self.files.iter().map(|f| f.1).collect::<Vec<_>>(),
        }).to_string()
    }

    fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.1).sum()
    }
}

/// Write `bytes` parts to `path` through a `.part` file beside it, so a file
/// of that name is always whole.
fn write_whole(path: &Path, parts: &[&[u8]]) -> std::io::Result<u64> {
    let part = path.with_extension("dng.part");
    let mut file = File::create(&part)?;
    let mut len = 0u64;
    for p in parts {
        file.write_all(p)?;
        len += p.len() as u64;
    }
    drop(file);
    std::fs::rename(&part, path)?;
    Ok(len)
}

/// The frames of `index` between `t_in` and `t_out` (seconds from the first
/// frame; either end open when `None`), the way the other exports pick them.
pub fn frames_in(index: &crate::x6_raw::TakeIndex, t_in: Option<f64>, t_out: Option<f64>) -> std::ops::Range<usize> {
    let n = index.frames();
    let dt = index.dt_s();
    let t_in = t_in.unwrap_or(0.0).max(0.0);
    let t_out = t_out.map(|t| t.max(t_in + 1e-3)).unwrap_or(f64::INFINITY);
    let first = (0..n).find(|&i| index.pts_s(i) >= t_in - 0.5 * dt).unwrap_or(n);
    let end = (first..n).find(|&i| index.pts_s(i) >= t_out).unwrap_or(n);
    first..end
}

/// Copy the take's own small files (everything in its folder but the
/// capture and the camera's closing record of it) into `out`.
fn copy_take_files(take_dir: &Path, out: &Path) -> std::io::Result<Vec<String>> {
    let mut copied = Vec::new();
    for entry in std::fs::read_dir(take_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || !entry.file_type()?.is_file() {
            continue;
        }
        if name.eq_ignore_ascii_case(x6b::CAPTURE_FILE) || name.eq_ignore_ascii_case(x6b::SEAL_FILE) || name.eq_ignore_ascii_case(FRAMES_FILE) {
            continue;
        }
        std::fs::copy(entry.path(), out.join(&name))?;
        copied.push(name);
    }
    copied.sort();
    Ok(copied)
}

/// Write the frames of the take at `source` between `t_in` and `t_out`
/// (seconds on the take's timeline; `None` = that end of the take) into the
/// folder `out` (created if needed) as the sensor recorded them: see the
/// module's description. Returns the number of pairs written; a cancelled
/// run keeps the pairs it finished and lists them.
pub fn export_sensor_dng_sequence(
    source: &Path,
    out: &Path,
    t_in: Option<f64>,
    t_out: Option<f64>,
    mut progress: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<u64> {
    let index = crate::x6_raw::open(source)?;
    let frames = frames_in(&index, t_in, t_out);
    let total = frames.len();
    // The sensor channel of each of a record's channels: a paired take's
    // two, the one a take of one lens holds.
    let channels: Vec<usize> = match index.take.layout.single_channel() {
        Some(channel) => vec![channel],
        None => vec![0, 1],
    };
    for &channel in &channels {
        std::fs::create_dir_all(out.join(channel_dir(channel)))?;
    }
    let camera = RawCamera::X6;
    let (format, mosaic) = (index.take.format(), Mosaic::of(&index.take));
    let capture = index.take.capture_path();
    // Two cores stay free for the app's own threads.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let workers = cores.saturating_sub(2).clamp(1, MAX_WORKERS).min(total.max(1));
    let stop = AtomicBool::new(false);
    let started = std::time::Instant::now();
    let (tx, rx) = std::sync::mpsc::channel::<Result<WrittenPair>>();
    // One thread reads the capture front to back (a network share or a
    // spinning disk serves one stream best); the workers unpack, pack and
    // write. The queue between them is short, and record buffers go round.
    let (work_tx, work_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<u8>)>(workers);
    let work_rx = Mutex::new(work_rx);
    let (free_tx, free_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let mut written: Vec<WrittenPair> = Vec::with_capacity(total);
    let mut failed: Option<Error> = None;

    std::thread::scope(|scope| {
        {
            let tx = tx.clone();
            let (index, capture, stop, cancel, frames) = (&index, &capture, &stop, &cancel, &frames);
            scope.spawn(move || {
                let fail = |e: Error| {
                    stop.store(true, Ordering::SeqCst);
                    let _ = tx.send(Err(e));
                };
                let mut file = match File::open(capture) {
                    Ok(f) => f,
                    Err(e) => return fail(e.into()),
                };
                for i in frames.clone() {
                    if stop.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst) {
                        break;
                    }
                    let mut record = free_rx.try_recv().unwrap_or_default();
                    if let Err(e) = x6b::read_record(&mut file, &index.pairs[i], &mut record) {
                        return fail(Error::Ffmpeg(e.to_string()));
                    }
                    if work_tx.send((i, record)).is_err() {
                        break;
                    }
                }
            });
        }
        for _ in 0..workers {
            let (tx, free_tx) = (tx.clone(), free_tx.clone());
            let (index, work_rx, stop, cancel, channels, mosaic) = (&index, &work_rx, &stop, &cancel, &channels, &mosaic);
            scope.spawn(move || {
                let one = |record: &[u8], samples: &mut [u16], scratch: &mut EyeScratch, entry: &PairEntry| -> Result<WrittenPair> {
                    let got = x6b::decode_record(record, format, Channels::Both, samples, 1).map_err(|e| Error::Ffmpeg(e.to_string()))?;
                    if got.request != entry.request {
                        return Err(Error::Ffmpeg(format!("byte {} of the capture holds request {}, {} expected", entry.offset, got.request, entry.request)));
                    }
                    let mut files = Vec::with_capacity(channels.len());
                    for (&channel, eye) in channels.iter().zip(samples.chunks(mosaic.samples())) {
                        let tag = FrameTag { channel, request: entry.request, exposure_ns: entry.exposure_ns[channel] };
                        let (head, body) = eye_dng(eye, mosaic, tag, &camera, scratch);
                        files.push((channel, write_whole(&out.join(frame_file(channel, entry.request)), &[&head, body])?));
                    }
                    let signature = String::from_utf8_lossy(entry.profile.signature()).trim_end_matches('\0').to_owned();
                    Ok(WrittenPair { request: entry.request, exposure_ns: entry.exposure_ns, signature, files })
                };
                let mut samples = vec![0u16; format.layout.samples()];
                let mut scratch = EyeScratch::default();
                loop {
                    let job = work_rx.lock().ok().and_then(|rx| rx.recv().ok());
                    let Some((i, record)) = job else { break };
                    // A stopped run still empties the queue, so the reader is never left waiting on it.
                    if stop.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst) {
                        continue;
                    }
                    let result = one(&record, &mut samples, &mut scratch, &index.pairs[i]);
                    let _ = free_tx.send(record);
                    if result.is_err() {
                        stop.store(true, Ordering::SeqCst);
                    }
                    if tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        drop(free_tx);
        for result in rx {
            match result {
                Ok(pair) => {
                    written.push(pair);
                    progress(ExportProgress {
                        frame_idx: written.len() as u64,
                        total_frames: total as u64,
                        fps_avg: written.len() as f32 / started.elapsed().as_secs_f32().max(1e-3),
                    });
                }
                Err(e) => {
                    failed.get_or_insert(e);
                }
            }
        }
    });

    // The index lists what is whole on disk, whether the run ended or not.
    written.sort_by_key(|p| p.request);
    let mut lines = String::with_capacity(written.len() * 220);
    for pair in &written {
        lines.push_str(&pair.line());
        lines.push('\n');
    }
    std::fs::write(out.join(FRAMES_FILE), lines)?;
    if let Some(e) = failed {
        return Err(e);
    }
    let copied = copy_take_files(&index.take.dir, out)?;
    let bytes: u64 = written.iter().map(|p| p.bytes()).sum();
    if cancel.load(Ordering::SeqCst) && written.len() < total {
        tracing::info!("sensor frames: cancelled after {} of {total} pairs", written.len());
    }
    tracing::info!(
        "sensor frames: {} pairs ({:.1} GB) in {} on {workers} threads in {:.1?} ({:.1} pairs/s); beside them: {}",
        written.len(), bytes as f64 / 1e9, out.display(), started.elapsed(),
        written.len() as f64 / started.elapsed().as_secs_f64().max(1e-3), copied.join(", "));
    Ok(written.len() as u64)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// A written file read back: its fields and its samples.
    pub(crate) struct ReadBack {
        pub fields: BTreeMap<u16, (u16, u32, Vec<u8>)>,
        pub samples: Vec<u16>,
    }

    impl ReadBack {
        pub(crate) fn longs(&self, tag: u16) -> Vec<u32> {
            let (kind, count, data) = &self.fields[&tag];
            match *kind {
                SHORT => data.chunks_exact(2).take(*count as usize).map(|b| u16::from_le_bytes([b[0], b[1]]) as u32).collect(),
                LONG => data.chunks_exact(4).take(*count as usize).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                BYTE => data.iter().take(*count as usize).map(|&b| b as u32).collect(),
                k => panic!("tag {tag} is of type {k}"),
            }
        }
        pub(crate) fn text(&self, tag: u16) -> String {
            let (kind, _, data) = &self.fields[&tag];
            assert_eq!(*kind, ASCII);
            String::from_utf8_lossy(&data[..data.len() - 1]).into_owned()
        }
        pub(crate) fn fractions(&self, tag: u16) -> Vec<f64> {
            let (kind, count, data) = &self.fields[&tag];
            data.chunks_exact(8).take(*count as usize).map(|b| {
                let d = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as f64;
                match *kind {
                    SRATIONAL => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / d,
                    RATIONAL => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / d,
                    k => panic!("tag {tag} is of type {k}"),
                }
            }).collect()
        }
    }

    /// Read a file this module wrote, checking its structure on the way.
    pub(crate) fn read_back(file: &[u8]) -> ReadBack {
        let u16_at = |o: usize| u16::from_le_bytes([file[o], file[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes([file[o], file[o + 1], file[o + 2], file[o + 3]]);
        assert_eq!(&file[..4], b"II\x2a\x00");
        let ifd = u32_at(4) as usize;
        let n = u16_at(ifd) as usize;
        let mut fields = BTreeMap::new();
        let mut last = 0u16;
        for i in 0..n {
            let e = ifd + 2 + 12 * i;
            let (tag, kind, count) = (u16_at(e), u16_at(e + 2), u32_at(e + 4));
            assert!(tag > last, "tags in rising order");
            last = tag;
            let unit = match kind { BYTE | ASCII => 1, SHORT => 2, LONG => 4, RATIONAL | SRATIONAL => 8, k => panic!("type {k}") };
            let len = unit * count as usize;
            let data = if len <= 4 { file[e + 8..e + 8 + len].to_vec() } else {
                let at = u32_at(e + 8) as usize;
                assert_eq!(at % 2, 0, "values on a word boundary");
                file[at..at + len].to_vec()
            };
            fields.insert(tag, (kind, count, data));
        }
        assert_eq!(u32_at(ifd + 2 + 12 * n), 0, "one IFD");
        let mut back = ReadBack { fields, samples: Vec::new() };
        let (w, h, tw, th) = (back.longs(256)[0] as usize, back.longs(257)[0] as usize, back.longs(322)[0] as usize, back.longs(323)[0] as usize);
        assert_eq!((tw, h % th), (w, 0));
        let (offsets, counts) = (back.longs(324), back.longs(325));
        assert_eq!((offsets.len(), counts.len()), (h / th, h / th));
        let mut end = 0usize;
        for (&o, &c) in offsets.iter().zip(&counts) {
            assert!(o as usize >= end, "tiles in order, apart");
            end = (o + c) as usize;
            let tile = crate::ljpeg::test_support::ljpeg_decode(&file[o as usize..end]);
            assert_eq!((tile.height, tile.width, tile.ncomp), (th, tw / 2, 2));
            back.samples.extend_from_slice(&tile.samples);
        }
        assert!(end <= file.len() && file.len() - end < 4, "nothing after the last tile");
        back
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    /// A frame's worth of sensor-like samples: a scene, noise, a clipped
    /// patch, codes above the white level.
    fn mosaic(seed: u64) -> Vec<u16> {
        let mut s = seed.max(1);
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        (0..x6b::EYE_SAMPLES).map(|i| {
            let (x, y) = (i % EYE_SIDE, i / EYE_SIDE);
            let site = [1.0f64, 0.6, 1.7, 1.0][(y % 2) * 2 + x % 2];
            let scene = 64.0 + 2600.0 * site * (1.0 + ((x as f64) * 0.004).sin() * ((y as f64) * 0.003).cos());
            let v = scene + (next() % 64) as f64;
            if (1000..1200).contains(&x) && (800..900).contains(&y) { 15408 + (next() % 900) as u16 } else { v.clamp(0.0, 16383.0) as u16 }
        }).collect()
    }

    #[test]
    fn fractions_are_the_closest_small_ones() {
        assert_eq!(fraction(0.71875), (23, 32));
        assert_eq!(fraction(-0.1484375), (-19, 128));
        assert_eq!(fraction(1.0), (1, 1));
        assert_eq!(fraction(0.0), (0, 1));
        assert_eq!(fraction(0.5654296875), (579, 1024));
        let (n, d) = fraction(std::f64::consts::PI);
        assert!(d <= 1_000_000 && (n as f64 / d as f64 - std::f64::consts::PI).abs() < 1e-11, "{n}/{d}");
        let (n, d) = fraction(-0.333_333_333_333);
        assert_eq!((n, d), (-1, 3));
    }

    #[test]
    fn a_frame_reads_back_as_recorded() {
        let samples = mosaic(7);
        let tag = FrameTag { channel: 1, request: 94, exposure_ns: 416_279_938_533 };
        let mut scratch = EyeScratch::default();
        let (head, body) = eye_dng(&samples, &Mosaic::PAIRED, tag, &RawCamera::X6, &mut scratch);
        let file = [head.as_slice(), body].concat();
        let back = read_back(&file);
        // Every sample, the ones above the white level too.
        assert!(back.samples == samples, "the samples differ");
        assert!(samples.iter().any(|&v| v > x6b::WHITE_LEVEL));
        // The picture's description.
        assert_eq!(back.longs(254), [0]);
        assert_eq!((back.longs(256), back.longs(257)), (vec![3840], vec![3840]));
        assert_eq!((back.longs(258), back.longs(259), back.longs(262)), (vec![14], vec![7], vec![32803]));
        assert_eq!((back.longs(277), back.longs(284), back.longs(274)), (vec![1], vec![1], vec![1]));
        assert_eq!((back.longs(322), back.longs(323)), (vec![3840], vec![160]));
        assert_eq!((back.longs(33421), back.longs(33422)), (vec![2, 2], vec![1, 2, 0, 1]));
        assert_eq!((back.longs(50706), back.longs(50707)), (vec![1, 4, 0, 0], vec![1, 1, 0, 0]));
        assert_eq!((back.longs(50714), back.longs(50717), back.longs(50778)), (vec![64], vec![15408], vec![21]));
        // The camera, and the frame's place in the take.
        assert_eq!((back.text(271), back.text(272), back.text(50708)), ("Arashi Vision".into(), "Insta360 X6".into(), UNIQUE_CAMERA_MODEL.to_string()));
        assert_eq!(FrameTag::parse(&back.text(270)), Some(tag));
        assert_eq!(back.text(270), "CAM1 request=94 exposure_start_ns=416279938533");
        // The colours the app develops with.
        let m = RawCamera::X6.color_matrix;
        assert_eq!(back.fractions(50721), [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]);
        assert_eq!(back.fractions(50728), RawCamera::X6.as_shot_neutral);
        // Packed, it is well under the loose 2 bytes a sample.
        assert!(file.len() < x6b::EYE_SAMPLES * 2 * 3 / 4, "{} bytes", file.len());
        // The scratch space serves the next frame.
        let other = mosaic(8);
        let (head, body) = eye_dng(&other, &Mosaic::PAIRED, FrameTag { channel: 0, ..tag }, &RawCamera::X6, &mut scratch);
        let back = read_back(&[head.as_slice(), body].concat());
        assert!(back.samples == other);
        assert_eq!(back.text(270), "CAM0 request=94 exposure_start_ns=416279938533");
    }

    #[test]
    fn names_and_index_lines() {
        assert_eq!(frame_file(0, 94), "CAM0/000000094.dng");
        assert_eq!(frame_file(1, 123_456_789), "CAM1/123456789.dng");
        let pair = WrittenPair { request: 94, exposure_ns: [416_279_938_533, 416_279_938_901], signature: "L1024___".into(), files: vec![(0, 11), (1, 12)] };
        let v: serde_json::Value = serde_json::from_str(&pair.line()).unwrap();
        assert_eq!(v["request"], 94);
        assert_eq!(v["exposures_ns"][1], 416_279_938_901u64);
        assert_eq!(v["files"][0], "CAM0/000000094.dng");
        assert_eq!(v["files"][1], "CAM1/000000094.dng");
        assert_eq!((v["signature"].as_str(), v["bytes"][1].as_u64()), (Some("L1024___"), Some(12)));
        assert!(!pair.line().contains('\n'));
        // One lens's frame: its one file.
        let one = WrittenPair { request: 7, exposure_ns: [5, 5], signature: "X6S8B01".into(), files: vec![(1, 9)] };
        let v: serde_json::Value = serde_json::from_str(&one.line()).unwrap();
        assert_eq!((v["files"].as_array().map(|a| a.len()), v["files"][0].as_str(), v["bytes"][0].as_u64()), (Some(1), Some("CAM1/000000007.dng"), Some(9)));
        assert_eq!(FrameTag::parse("CAM0 request=7 exposure_start_ns=12"), Some(FrameTag { channel: 0, request: 7, exposure_ns: 12 }));
        assert_eq!(FrameTag::parse("frame 12 at 0.4 s"), None);
    }

    /// A stretch of a sample take (skipped when the volume is not mounted):
    /// the files hold the capture's own samples, the folder its index and
    /// the take's small files.
    #[test]
    fn a_stretch_of_a_sample_take() {
        let take = Path::new("/Volumes/Database/x6/1115 raw video/RAWVIDEO-20261116-011809");
        if !take.exists() { return; }
        let out = std::env::temp_dir().join(format!("vr180-sensor-dng-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let index = crate::x6_raw::open(take).unwrap();
        let dt = index.dt_s();
        // Frames 3, 4 and 5 of the take.
        let seen = std::sync::Mutex::new(Vec::new());
        let n = export_sensor_dng_sequence(take, &out, Some(3.0 * dt), Some(5.5 * dt), |p| seen.lock().unwrap().push((p.frame_idx, p.total_frames)), Arc::default()).unwrap();
        assert_eq!(n, 3);
        assert_eq!(seen.into_inner().unwrap(), [(1, 3), (2, 3), (3, 3)]);
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(out.join(FRAMES_FILE)).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        let mut iter = crate::x6_raw::X6RawIter::new(take, false, false, 1, None).unwrap();
        for (k, line) in lines.iter().enumerate() {
            let entry = index.pairs[3 + k];
            assert_eq!(line["request"], entry.request);
            assert_eq!(line["exposures_ns"][0], entry.exposure_ns[0]);
            let samples = iter.decode_frame(3 + k).unwrap().to_vec();
            for channel in 0..2 {
                let path = out.join(line["files"][channel].as_str().unwrap());
                let file = std::fs::read(&path).unwrap();
                assert_eq!(file.len() as u64, line["bytes"][channel].as_u64().unwrap());
                let back = read_back(&file);
                assert!(back.samples == samples[channel * x6b::EYE_SAMPLES..(channel + 1) * x6b::EYE_SAMPLES], "request {} channel {channel}", entry.request);
                assert_eq!(FrameTag::parse(&back.text(270)), Some(FrameTag { channel, request: entry.request, exposure_ns: entry.exposure_ns[channel] }));
            }
        }
        // The take's small files travel; the capture and its closing record do not.
        for name in ["clip.json", "calibration.json", "gyro.csv", "gyro-info.json"] {
            assert!(out.join(name).is_file(), "{name}");
        }
        assert!(!out.join("capture.x6b").exists() && !out.join("seal.json").exists());
        assert!(std::fs::read_dir(out.join("CAM0")).unwrap().all(|e| e.unwrap().file_name().to_string_lossy().ends_with(".dng")));
        let _ = std::fs::remove_dir_all(&out);
    }
}
