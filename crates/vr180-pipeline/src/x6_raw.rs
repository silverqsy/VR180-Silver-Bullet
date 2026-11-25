//! Insta360 X6 raw video takes in the pipeline.
//!
//! A take is a folder (see [`vr180_fisheye::x6b`]); every consumer names it
//! by that folder. This module turns it into the things the rest of the
//! pipeline asks any clip for:
//!
//! - [`open`] — the take's description and its list of pairs. Listing the
//!   pairs reads one header per pair across the whole capture (seconds on a
//!   network volume), so the result is kept for the session and, when a
//!   cache folder is set ([`set_cache_dir`]), on disk.
//! - [`probe`] / [`timeline`] — frame size, rate, duration and the time of
//!   every frame, from the exposure stamps: a pair that was never saved
//!   leaves its gap in the timeline.
//! - [`build_imu`] — the lens calibration (the factory string, the frame
//!   placed where the take's sensor record says the sensor was read, or
//!   like the camera's own 8K video frame for a take without one) and the
//!   gyro as per-frame poses.
//! - [`X6RawIter`] — the frames. With a GPU device a pair is decoded,
//!   uploaded as two mosaics and demosaiced there into camera-linear 16-bit
//!   eyes; without one it is demosaiced on the CPU.
//! - [`audio_source`] — the sound file and where the picture starts in it.
//!
//! The two capture channels follow the camera's video streams: channel 0 is
//! the lens the stock video writes as stream 0 (calibration entry 2, the
//! RIGHT eye on a VR180-modded body), channel 1 the other.
//!
//! A take of one lens ([`x6b::Layout::Single`]) is a single-eye source
//! ([`crate::source_kind::is_single_eye`]): its frames are one 7680×7680
//! eye, a pair's left with no right, under that lens's calibration entry.
//!
//! A paired take of another frame size ([`x6b::Layout::PairedOf`]) is a
//! paired take in everything but its frame: 5504×5504 holds more of the
//! sensor than the video frame, scaled; 4912×4912 and 3968×3968 hold the
//! middle of it at its own pixels, a crop of the lens's view.
//!
//! A take with a record for each eye ([`x6b::Layout::EyeRecords`]) is a
//! paired take too: its pairs are the two sensors' frames whose exposures
//! started together, each pair two records of the capture. 5760×5760 and
//! 6000×6000 hold either the sensor's whole view, scaled, or the middle of
//! it at its own pixels; 4608×3968 is a wider than tall part of the middle.
//!
//! The frames are the sensor's linear values (black level subtracted, white
//! level at 1.0, no white balance): they need the raw development stage of
//! the colour stack before anything display-referred. A take that records
//! the white balance the camera had found ([`white_balance`]) is developed
//! for that white unless another is set ([`camera_of`]).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::insta360::{Insta360Meta, InsvImuSample, X6ViewChoice, X6_DEFAULT_WINDOW};
use vr180_fisheye::x6b::{self, Channels, Layout, PairEntry, Take, EYE_SAMPLES, EYE_SIDE};
use vr180_fisheye::DjiOsvImu;

use crate::decode::VideoProbe;
use crate::fisheye_decode::{FisheyePair, FisheyePairIter, GpuPair};
use crate::frame_timeline::FrameTimeline;
use crate::gpu::Device;
use crate::raw_color::RawCamera;
use crate::raw_gpu::BayerLayout;
use crate::{Error, Result};

fn take_err(e: impl std::fmt::Display) -> Error {
    Error::Ffmpeg(e.to_string())
}

/// `FisheyePair::bit_depth` of a pair whose two buffers are MOSAICS (one
/// little-endian `u16` per site, `eye_w × eye_h` sites) and not pictures:
/// what [`X6RawIter::mosaic_pairs`] hands a consumer that demosaics on its
/// own device ([`demosaic_pair`]). A take of one lens has one mosaic, the
/// left, and no right.
pub const MOSAIC_BIT_DEPTH: u8 = 14;

/// Demosaic a mosaic pair (see [`MOSAIC_BIT_DEPTH`]) of a take whose
/// mosaic is `bayer` ([`bayer_at`]) on `dev` at full size: the two
/// camera-linear eyes, left and right (one eye twice, for a pair of one).
pub fn demosaic_pair(dev: &Device, pair: &FisheyePair, bayer: &BayerLayout) -> Result<(wgpu::Texture, wgpu::Texture)> {
    let (w, h) = (pair.eye_w, pair.eye_h);
    let want = w as usize * h as usize * 2;
    let one_eye = pair.right.is_empty();
    if pair.bit_depth != MOSAIC_BIT_DEPTH || pair.left.len() != want || (!one_eye && pair.right.len() != want) {
        return Err(take_err("not a mosaic pair"));
    }
    let eye = |bytes: &[u8]| -> Result<wgpu::Texture> {
        let mosaic = dev.new_bayer_texture(w, h);
        dev.write_bayer_bytes(&mosaic, bytes);
        let out = dev.new_linear_eye_texture(w, h);
        dev.demosaic_bayer(&mosaic, bayer, &out)?;
        Ok(out)
    };
    let left = eye(&pair.left)?;
    if one_eye {
        return Ok((left.clone(), left));
    }
    Ok((left, eye(&pair.right)?))
}

/// The mosaic of a paired take's frames.
pub const BAYER: BayerLayout = BayerLayout {
    cfa: x6b::CFA,
    black: x6b::BLACK_LEVEL as f32,
    white: x6b::WHITE_LEVEL as f32,
    // `RawCamera::X6`'s as-shot neutral (a test in `bayer` holds them equal).
    neutral: [0.565_429_7, 1.0, 0.401_367_2],
};

/// The mosaic of `take`'s frames: its pattern and levels (a take of one
/// lens states its own), the sensor's neutral.
pub fn bayer_of(take: &Take) -> BayerLayout {
    BayerLayout { cfa: take.cfa, black: take.black_level as f32, white: take.white_level as f32, neutral: BAYER.neutral }
}

/// [`bayer_of`] the take `path` names; a paired take's for a path that
/// names none.
pub fn bayer_at(path: &Path) -> BayerLayout {
    describe(path).map_or(BAYER, |take| bayer_of(&take))
}

/// The white balance the camera had found when `take` began, as the take
/// records it ([`x6b::read_white_balance`]): each sensor's neutral, channel
/// 0 then channel 1; `None` for a sensor the take records none for. A take
/// of one lens has its one sensor's in both places.
pub fn white_balance(take: &Take) -> [Option<[f64; 3]>; 2] {
    let eyes = x6b::read_white_balance(take);
    match take.layout.single_channel() {
        Some(channel) => [eyes[channel.min(1)]; 2],
        None => eyes,
    }
}

/// The camera each of `take`'s channels is written as, frame for frame
/// ([`crate::sensor_dng`]): the sensor's colours with the white balance the
/// take records for that sensor as its as-shot neutral, the reference
/// neutral for a sensor it records none for.
pub fn eye_cameras(take: &Take) -> [RawCamera; 2] {
    white_balance(take).map(|neutral| match neutral {
        Some(n) => RawCamera { as_shot_neutral: n, ..RawCamera::X6 },
        None => RawCamera::X6,
    })
}

/// The camera `take`'s frames are developed as: the sensor's colours with
/// the white balance the take records as its as-shot neutral. The two eyes
/// of a pair are developed alike: where the take records a white balance
/// for each sensor, the one halfway between the two (channel by channel, as
/// ratios); where for one sensor only, that one. A take that records none
/// keeps the reference neutral.
pub fn camera_of(take: &Take) -> RawCamera {
    let eyes = white_balance(take);
    let neutral = match (eyes[0], eyes[1]) {
        (Some(a), Some(b)) => Some([0, 1, 2].map(|i| (a[i] * b[i]).sqrt())),
        (a, b) => a.or(b),
    };
    match neutral {
        Some(n) => RawCamera { as_shot_neutral: [n[0] / n[1], 1.0, n[2] / n[1]], ..RawCamera::X6 },
        None => RawCamera::X6,
    }
}

/// [`camera_of`] the take `path` names; the reference camera for a path
/// that names none. A take is looked at once (its small files); later calls
/// touch no file.
pub fn camera_at(path: &Path) -> RawCamera {
    static KNOWN: OnceLock<Mutex<HashMap<PathBuf, RawCamera>>> = OnceLock::new();
    let known = KNOWN.get_or_init(Default::default);
    if let Some(camera) = known.lock().ok().and_then(|k| k.get(path).copied()) {
        return camera;
    }
    let Ok(take) = describe(path) else { return RawCamera::X6 };
    let camera = camera_of(&take);
    if camera == RawCamera::X6 {
        tracing::info!("raw take {}: it records no white balance: developed from the sensor's reference neutral", take.dir.display());
    } else {
        let eyes = white_balance(&take);
        let (kelvin, tint) = camera.as_shot_temperature();
        tracing::info!(
            "raw take {}: developed for the white balance it records: neutral {:.4?} (about {kelvin:.0} K, tint {tint:+.0}; its sensors: {:.4?} and {:.4?})",
            take.dir.display(), camera.as_shot_neutral, eyes[0], eyes[1]);
    }
    if let Ok(mut k) = known.lock() {
        k.insert(path.to_path_buf(), camera);
    }
    camera
}

/// The take `path` names holds one lens's frames: a single-eye source. A
/// take is looked at once (its `clip.json`); later calls touch no file. A
/// path that names no take (yet: a folder still being copied) is asked
/// again the next time.
pub fn single_eye(path: &Path) -> bool {
    static KNOWN: OnceLock<Mutex<HashMap<PathBuf, bool>>> = OnceLock::new();
    let known = KNOWN.get_or_init(Default::default);
    if let Some(one) = known.lock().ok().and_then(|k| k.get(path).copied()) {
        return one;
    }
    let Some(layout) = x6b::take_layout(path) else { return false };
    let one = matches!(layout, Layout::Single { .. });
    if let Ok(mut k) = known.lock() {
        k.insert(path.to_path_buf(), one);
    }
    one
}

/// Whether the take `path` names was shot with the camera upside down
/// ([`crate::insv_imu::mounted_upside_down`]), from the first seconds of
/// its accelerometer record. `None`: it has none, or the record says
/// nothing (a camera on its side).
pub fn mounted_upside_down(path: &Path) -> Option<bool> {
    use std::io::Read;
    let dir = x6b::take_dir(path)?;
    let mut head = Vec::new();
    File::open(dir.join(x6b::ACCEL_FILE)).ok()?.take(1 << 18).read_to_end(&mut head).ok()?;
    const G: f32 = 9.80665;
    // counter_ticks,x,y,z (m/s²); the last line of the head may be cut short and is left out.
    let text = String::from_utf8_lossy(&head);
    let mut lines: Vec<&str> = text.lines().skip(1).collect();
    lines.pop();
    crate::insv_imu::mounted_upside_down(lines.iter().filter_map(|line| {
        let mut fields = line.split(',').skip(1).map(|f| f.trim().parse::<f32>());
        Some([fields.next()?.ok()? / G, fields.next()?.ok()? / G, fields.next()?.ok()? / G])
    }))
}

/// What the take `path` names says of itself as an archive copy
/// ([`crate::raw_archive`]); `None` for a camera's take. A take is looked at
/// once (its `clip.json`); later calls touch no file.
pub fn archive_note(path: &Path) -> Option<x6b::ArchiveNote> {
    static KNOWN: OnceLock<Mutex<HashMap<PathBuf, Option<x6b::ArchiveNote>>>> = OnceLock::new();
    let known = KNOWN.get_or_init(Default::default);
    if let Some(note) = known.lock().ok().and_then(|k| k.get(path).cloned()) {
        return note;
    }
    let dir = x6b::take_dir(path)?;
    let note = x6b::read_archive_note(&dir);
    if let Ok(mut k) = known.lock() {
        k.insert(path.to_path_buf(), note.clone());
    }
    note
}

/// A take, opened: its description and its pairs in request order.
#[derive(Debug)]
pub struct TakeIndex {
    pub take: Take,
    pub pairs: Vec<PairEntry>,
    /// Requests of the take's window that were never saved.
    pub missing: Vec<u32>,
    /// Time of every pair in seconds from the first one.
    pub timeline: FrameTimeline,
    /// Each pair's integration times (channel 0, channel 1), nanoseconds,
    /// where the take lists them ([`x6b::read_exposures`]).
    pub shutter_ns: Vec<[Option<u64>; 2]>,
    /// Each pair's time on the take's counter, nanoseconds: see [`Self::new`].
    time_ns: Vec<u64>,
    pts_s: Vec<f64>,
}

impl TakeIndex {
    /// A pair's time is the middle of its two eyes' stamps, the start of
    /// their first rows' exposures — unless the take lists every pair's
    /// integration time, in which case that time is added: the moment the
    /// first rows are read out. The sensor reads frames out at an even pace
    /// whatever the exposure does, while the START of an exposure moves when
    /// its length changes: on the stamps alone a frame after a shutter
    /// change sits early or late, and after a change of more than half a
    /// frame period it looks as if a frame were missing before it.
    fn new(take: Take, pairs: Vec<PairEntry>) -> Self {
        let listed = x6b::read_exposures(&take);
        // A record's own word on its integration time comes first, then the take's list.
        let shutter_ns: Vec<[Option<u64>; 2]> = pairs.iter().map(|p| {
            let list = listed.get(&p.request).copied().unwrap_or([None; 2]);
            [0, 1].map(|channel| Some(p.shutter_ns[channel]).filter(|ns| *ns > 0 && *ns < 1_000_000_000).or(list[channel]))
        }).collect();
        let read_out: Option<Vec<u64>> = pairs.iter().zip(&shutter_ns)
            .map(|(p, s)| Some(p.time_ns() + s[0].or(s[1])? / 2 + s[1].or(s[0])? / 2))
            .collect();
        let time_ns = read_out.filter(|t| !t.is_empty()).unwrap_or_else(|| pairs.iter().map(|p| p.time_ns()).collect());
        let t0 = time_ns.first().copied().unwrap_or(0);
        let pts_s: Vec<f64> = time_ns.iter().map(|t| t.saturating_sub(t0) as f64 * 1e-9).collect();
        let dt = 1.0 / take.nominal_fps();
        let missing = x6b::missing_requests(&take, &pairs);
        Self { timeline: FrameTimeline::from_pts(pts_s.clone(), dt), take, pairs, missing, shutter_ns, time_ns, pts_s }
    }

    /// Frame `i`'s integration time in seconds: channel `channel`'s, or the
    /// other eye's when that one has none; `None` when the take lists
    /// neither.
    pub fn shutter_s(&self, i: usize, channel: usize) -> Option<f64> {
        let eyes = self.shutter_ns.get(i)?;
        eyes[channel.min(1)].or(eyes[1 - channel.min(1)]).map(|ns| ns as f64 * 1e-9)
    }

    /// Frames in the take.
    pub fn frames(&self) -> usize {
        self.pairs.len()
    }

    /// Nominal frame period in seconds.
    pub fn dt_s(&self) -> f64 {
        1.0 / self.take.nominal_fps()
    }

    /// Time of frame `i` in seconds from the first frame.
    pub fn pts_s(&self, i: usize) -> f64 {
        self.pts_s.get(i).copied().unwrap_or_else(|| self.pts_s.last().copied().unwrap_or(0.0))
    }

    /// Time of the first frame on the take's counter, nanoseconds.
    pub fn t0_ns(&self) -> u64 {
        self.time_ns.first().copied().unwrap_or(0)
    }

    /// From the first frame to the end of the last one, seconds.
    pub fn duration_s(&self) -> f64 {
        self.pts_s.last().copied().unwrap_or(0.0) + self.dt_s()
    }

    /// The frame shown at `t_s`: the first one at or after `t_s − dt/2`.
    pub fn frame_at(&self, t_s: f64) -> usize {
        let cut = t_s - self.dt_s() * 0.5;
        self.pts_s.partition_point(|&p| p < cut).min(self.pts_s.len().saturating_sub(1))
    }

    /// Mean bytes per second of the capture, both eyes together.
    pub fn bytes_per_second(&self) -> f64 {
        self.take.capture_bytes as f64 / self.duration_s().max(1e-6)
    }
}

// ── Opening a take (session cache + disk cache) ────────────────────────

#[derive(Clone, PartialEq, Eq, Hash)]
struct TakeKey {
    dir: PathBuf,
    bytes: u64,
    modified: Option<std::time::SystemTime>,
}

type Opened = std::result::Result<Arc<TakeIndex>, String>;

fn sessions() -> &'static Mutex<HashMap<TakeKey, Arc<OnceLock<Opened>>>> {
    static OPEN: OnceLock<Mutex<HashMap<TakeKey, Arc<OnceLock<Opened>>>>> = OnceLock::new();
    OPEN.get_or_init(Default::default)
}

fn cache_dir_slot() -> &'static Mutex<Option<PathBuf>> {
    static DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    DIR.get_or_init(Default::default)
}

fn progress_slot() -> &'static Mutex<HashMap<PathBuf, f32>> {
    static P: OnceLock<Mutex<HashMap<PathBuf, f32>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// Folder the pair lists of opened takes are kept in between sessions
/// (`None` = this session only).
pub fn set_cache_dir(dir: Option<PathBuf>) {
    *cache_dir_slot().lock().unwrap() = dir;
}

/// How far the pair listing of a take being opened has come (0..1), `None`
/// when none is running for it.
pub fn open_progress(path: &Path) -> Option<f32> {
    let dir = x6b::take_dir(path)?;
    progress_slot().lock().unwrap().get(&dir).copied()
}

/// `path` names a raw take: the folder, or one of its own files.
pub fn is_take(path: &Path) -> bool {
    x6b::take_dir(path).is_some()
}

/// The take folder `path` names.
pub fn take_dir(path: &Path) -> Option<PathBuf> {
    x6b::take_dir(path)
}

/// The take's description without listing its pairs (two small files).
pub fn describe(path: &Path) -> Result<Take> {
    let dir = x6b::take_dir(path).ok_or_else(|| take_err(format!("{}: not a raw take", path.display())))?;
    x6b::read_take(&dir).map_err(take_err)?.map_err(|why| take_err(format!("{}: {why}", dir.display())))
}

/// Read and decode the pair at `entry`, whatever kind of record holds it:
/// the camera's ([`x6b::decode_pair`]) or an archive copy's
/// ([`crate::raw_archive`]). `record` is scratch space reused across calls.
pub fn decode_pair(file: &mut File, entry: &PairEntry, format: x6b::Format, channels: Channels, record: &mut Vec<u8>, out: &mut [u16], threads: usize) -> Result<()> {
    if entry.profile != x6b::Profile::Archive {
        return x6b::decode_pair(file, entry, format, channels, record, out, threads).map_err(take_err);
    }
    x6b::read_record(file, entry, record).map_err(take_err)?;
    let got = crate::raw_archive::decode_record(record, format.layout, channels, out, threads)?;
    if got.request != entry.request || got.exposure_ns != entry.exposure_ns {
        return Err(take_err(format!("byte {} holds request {}, {} expected", entry.offset, got.request, entry.request)));
    }
    Ok(())
}

/// Decode a pair read by [`x6b::read_record`], whatever kind of record it
/// is ([`decode_pair`]).
pub fn decode_record(record: &[u8], format: x6b::Format, channels: Channels, out: &mut [u16], threads: usize) -> Result<PairEntry> {
    if x6b::is_archive_record(record) {
        return crate::raw_archive::decode_record(record, format.layout, channels, out, threads);
    }
    x6b::decode_record(record, format, channels, out, threads).map_err(take_err)
}

/// A kept pair list starts with one of these: a row per pair of a capture
/// whose pairs are one record each, of one with a record for each eye (the
/// second record's place and both integration times after the rest), or of
/// an archive copy (both integration times after the rest).
const INDEX_MAGIC: &[u8; 8] = b"X6BIDX01";
const INDEX_MAGIC_EYES: &[u8; 8] = b"X6BIDX02";
const INDEX_MAGIC_ARCHIVE: &[u8; 8] = b"X6BIDX03";
const INDEX_ROW: usize = 37;
const INDEX_ROW_EYES: usize = INDEX_ROW + 32;
const INDEX_ROW_ARCHIVE: usize = INDEX_ROW + 16;

/// The start and the row length of the kept pair list of `take`, whose
/// records are an archive copy's (`archive`) or the camera's.
fn index_form(take: &Take, archive: bool) -> (&'static [u8; 8], usize) {
    match take.layout {
        _ if archive => (INDEX_MAGIC_ARCHIVE, INDEX_ROW_ARCHIVE),
        Layout::EyeRecords(_) => (INDEX_MAGIC_EYES, INDEX_ROW_EYES),
        _ => (INDEX_MAGIC, INDEX_ROW),
    }
}

fn index_file(take: &Take, modified: Option<std::time::SystemTime>) -> Option<PathBuf> {
    let dir = cache_dir_slot().lock().unwrap().clone()?;
    let stamp = modified.and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
    let mut h = 0xcbf29ce484222325u64;
    for b in take.dir.to_string_lossy().bytes().chain(take.capture_bytes.to_le_bytes()).chain(stamp.to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    Some(dir.join(format!("{h:016x}.pairs")))
}

fn load_index(file: &Path, take: &Take) -> Option<Vec<PairEntry>> {
    let bytes = std::fs::read(file).ok()?;
    let archive = bytes.starts_with(INDEX_MAGIC_ARCHIVE);
    let (magic, row) = index_form(take, archive);
    let rest = bytes.strip_prefix(magic)?;
    if rest.len() < 16 {
        return None;
    }
    let (head, rows) = rest.split_at(16);
    let capture = u64::from_le_bytes(head[..8].try_into().ok()?);
    let count = u64::from_le_bytes(head[8..].try_into().ok()?);
    if capture != take.capture_bytes || count != take.seal.pairs || rows.len() != count as usize * row {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    for r in rows.chunks_exact(row) {
        let u64_at = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        let (other, shutter_ns) = if row == INDEX_ROW_EYES {
            (Some((u64_at(37), u64_at(45))), [u64_at(53), u64_at(61)])
        } else if row == INDEX_ROW_ARCHIVE {
            (None, [u64_at(37), u64_at(45)])
        } else {
            (None, [0; 2])
        };
        out.push(PairEntry {
            request: u32::from_le_bytes(r[..4].try_into().unwrap()),
            profile: *x6b::Profile::ALL.get(r[4] as usize)?,
            offset: u64_at(5),
            len: u64_at(13),
            exposure_ns: [u64_at(21), u64_at(29)],
            other,
            shutter_ns,
        });
    }
    // The same checks the walk makes on what the rows claim.
    let (lo, hi) = take.seal.request_window;
    let in_window = |p: &PairEntry| !take.window_numbers_records() || (p.request >= lo && p.request <= hi);
    let inside = |p: &PairEntry| {
        let fits = |offset: u64, len: u64| offset.checked_add(len).is_some_and(|end| end <= take.capture_bytes);
        fits(p.offset, p.len) && p.other.map_or(true, |(offset, len)| fits(offset, len))
    };
    let ok = out.windows(2).all(|w| w[0].request < w[1].request)
        && out.iter().all(|p| in_window(p) && p.profile.records(take.layout, take.fps_setting) && inside(p) && (p.profile == x6b::Profile::Archive) == archive);
    ok.then_some(out)
}

fn store_index(file: &Path, take: &Take, pairs: &[PairEntry]) {
    let (magic, row) = index_form(take, pairs.first().is_some_and(|p| p.profile == x6b::Profile::Archive));
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&take.capture_bytes.to_le_bytes());
    bytes.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    for p in pairs {
        bytes.extend_from_slice(&p.request.to_le_bytes());
        bytes.push(x6b::Profile::ALL.iter().position(|q| *q == p.profile).unwrap_or(0) as u8);
        bytes.extend_from_slice(&p.offset.to_le_bytes());
        bytes.extend_from_slice(&p.len.to_le_bytes());
        bytes.extend_from_slice(&p.exposure_ns[0].to_le_bytes());
        bytes.extend_from_slice(&p.exposure_ns[1].to_le_bytes());
        if row == INDEX_ROW_EYES {
            let (offset, len) = p.other.unwrap_or((0, 0));
            for v in [offset, len, p.shutter_ns[0], p.shutter_ns[1]] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        } else if row == INDEX_ROW_ARCHIVE {
            for v in p.shutter_ns {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    if let Some(parent) = file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(file, bytes) {
        tracing::warn!("raw take: could not keep the pair list at {}: {e}", file.display());
    }
}

/// A take's frames: the capture's records in request order, without a last
/// one the camera did not finish ([`x6b::leftover_in_last_record`]: part of
/// it is an earlier frame). A record that cannot be looked at stays.
fn whole_frames(take: &Take, mut pairs: Vec<PairEntry>) -> Vec<PairEntry> {
    match x6b::leftover_in_last_record(take, &pairs) {
        Ok(Some(left)) => {
            if let Some(last) = pairs.pop() {
                tracing::warn!(
                    "raw take {}: the camera did not finish its last record (request {}: {} of its {} tiles still hold request {}): left out, {} frames",
                    take.dir.display(), last.request, left.tiles, take.layout.tiles(), left.of_request, pairs.len());
            }
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("raw take {}: its last record could not be looked at ({e}): kept", take.dir.display()),
    }
    pairs
}

/// Open the take `path` names (the folder or one of its files). The first
/// call lists the pairs; later calls, from any thread, share the result.
pub fn open(path: &Path) -> Result<Arc<TakeIndex>> {
    let take = describe(path)?;
    let modified = std::fs::metadata(take.capture_path()).ok().and_then(|m| m.modified().ok());
    let key = TakeKey { dir: take.dir.clone(), bytes: take.capture_bytes, modified };
    let slot = sessions().lock().unwrap().entry(key).or_default().clone();
    slot.get_or_init(|| {
        let on_disk = index_file(&take, modified);
        if let Some(pairs) = on_disk.as_deref().and_then(|f| load_index(f, &take)) {
            tracing::info!("raw take {}: {} pairs (kept list)", take.dir.display(), pairs.len());
            return Ok(Arc::new(TakeIndex::new(take.clone(), whole_frames(&take, pairs))));
        }
        let t0 = std::time::Instant::now();
        let dir = take.dir.clone();
        let walked = x6b::scan_pairs(&take, |at, size| {
            progress_slot().lock().unwrap().insert(dir.clone(), at as f32 / size.max(1) as f32);
            true
        });
        progress_slot().lock().unwrap().remove(&dir);
        match walked {
            Ok(Some(pairs)) => {
                if let Some((black, white)) = take.stated_levels {
                    tracing::warn!("raw take {}: it states a black level of {black} and a white level of {white}, another frame's; its samples are read by the sensor's own, {} and {}",
                        take.dir.display(), take.black_level, take.white_level);
                }
                // The list is kept as the capture holds it.
                if let Some(f) = on_disk {
                    store_index(&f, &take, &pairs);
                }
                let index = TakeIndex::new(take.clone(), whole_frames(&take, pairs));
                tracing::info!(
                    "raw take {}: {} {} at {} fps listed in {:.1?} ({} missing, {:.0} MB/s {})",
                    take.dir.display(), index.frames(), if take.layout.channels() == 1 { "frames of one lens" } else { "pairs" },
                    take.fps_setting, t0.elapsed(), index.missing.len(), index.bytes_per_second() / 1e6,
                    if take.layout.channels() == 1 { "its one eye" } else { "both eyes" });
                if let Some(frames) = take.seal.eye_frames {
                    let alone = frames.map(|n| n.saturating_sub(take.seal.pairs));
                    let skipped = take.seal.eye_gaps.unwrap_or([0; 2]);
                    tracing::info!(
                        "raw take {}: a record for each eye, {} and {} frames of the two sensors; {} and {} have no partner and are left out; the sensors numbered {} and {} frames the capture does not hold",
                        take.dir.display(), frames[0], frames[1], alone[0], alone[1], skipped[0], skipped[1]);
                }
                Ok(Arc::new(index))
            }
            Ok(None) => Err("stopped".into()),
            Err(e) => Err(e.to_string()),
        }
    }).clone().map_err(take_err)
}

/// Frame size, rate and duration of a take. Uses the opened take when one
/// is at hand; otherwise the camera's closing record (no walk): the frame
/// count of the window at the nominal rate.
pub fn probe(path: &Path) -> Result<VideoProbe> {
    let take = describe(path)?;
    let known = {
        let modified = std::fs::metadata(take.capture_path()).ok().and_then(|m| m.modified().ok());
        let key = TakeKey { dir: take.dir.clone(), bytes: take.capture_bytes, modified };
        sessions().lock().unwrap().get(&key).and_then(|s| s.get().cloned()).and_then(|r| r.ok())
    };
    let (lo, hi) = take.seal.request_window;
    let duration_sec = match &known {
        Some(index) => index.duration_s(),
        None => (hi - lo + 1) as f64 / take.nominal_fps(),
    };
    let rate = (take.capture_bytes as f64 * 8.0 / duration_sec.max(1e-6) / take.layout.channels() as f64) as u64;
    Ok(VideoProbe {
        width: take.width,
        height: take.height,
        fps: take.nominal_fps() as f32,
        duration_sec,
        hlg: false,
        video_bitrates_bps: [rate, rate],
    })
}

/// The time of every frame of the take (opens it).
pub fn timeline(path: &Path) -> Result<FrameTimeline> {
    Ok(open(path)?.timeline.clone())
}

// ── Sound ──────────────────────────────────────────────────────────────

/// The take's sound file and the time, in seconds of that file's decoded
/// stream, at which the first frame's exposure starts (the codec's priming
/// samples included; negative when the picture starts first). `None` when
/// the take has no sound.
pub fn audio_source(path: &Path) -> Option<(PathBuf, f64)> {
    let take = describe(path).ok()?;
    let timing = x6b::read_audio_timing(&take)?;
    let file = take.dir.join(&timing.file);
    if !file.is_file() {
        return None;
    }
    // The first frame's time needs the pair list; without it (a take that
    // was never opened) the first request's stamp is read alone.
    let t0 = match open(path) {
        Ok(index) => index.t0_ns(),
        Err(_) => return None,
    };
    let lead = (t0 as i128 - timing.first_sample_ns as i128) as f64 * 1e-9;
    let priming = timing.codec_delay_samples as f64 / timing.sample_rate_hz.max(1) as f64;
    Some((file, lead + priming))
}

// ── Lens and motion ────────────────────────────────────────────────────

/// Where the factory calibration's frame (the 7744² of each sensor half)
/// starts on the sensor, in the register coordinates a take's sensor record
/// is written in: the 7680² window of the camera's own 8K video lies 32
/// pixels inside that frame and starts at 256 on the sensor.
const CALIBRATION_ORIGIN: (f64, f64) = (224.0, 224.0);

/// The place of a take's frame in the calibration's frame, from its sensor
/// record ([`x6b::read_sensor_windows`]); `None` for a take without one.
/// The two sensors are read alike; channel 0's window stands for both.
pub fn stated_frame(take: &Take) -> Option<vr180_fisheye::insta360::StatedFrame> {
    let eyes = x6b::read_sensor_windows(take)?;
    if eyes[0].origin != eyes[1].origin || eyes[0].scale != eyes[1].scale {
        tracing::warn!("raw take {}: its two sensors are read from different windows ({:?} and {:?}): the first is used for both",
            take.dir.display(), eyes[0].origin, eyes[1].origin);
    }
    let w = eyes[0];
    Some(vr180_fisheye::insta360::StatedFrame {
        width: take.width,
        height: take.height,
        origin: (w.origin.0 - CALIBRATION_ORIGIN.0, w.origin.1 - CALIBRATION_ORIGIN.1),
        scale: w.scale,
    })
}

/// What the camera's metadata would say about the take's frames, as far as
/// the take states it: the factory lens string, the frame placed where its
/// sensor record says the sensor was read ([`stated_frame`]), channel 0 =
/// calibration entry 2. No motion data. A take without a sensor record is
/// placed like the camera's own 8K video frame (the 7680² window of each
/// sensor half, binned 2×2).
///
/// A take of one lens: that lens's entry alone, as the camera's own
/// single-lens video states it, and the same window at the sensor's own
/// resolution.
pub fn lens_meta(take: &Take) -> Insta360Meta {
    let calib_string = x6b::read_calibration_string(&take.dir).unwrap_or_default();
    let mut lenses = vr180_fisheye::insta360::parse_calib_string(&calib_string).unwrap_or_default();
    if lenses.len() < 2 {
        tracing::warn!("raw take {}: no factory lens string, the generic X6 profile is used", take.dir.display());
    }
    let single = take.layout.single_channel();
    if let Some(channel) = single {
        // Channel 0 is the string's second entry, as in a paired take.
        lenses = match lenses.len() {
            n if n >= 2 => vec![lenses.swap_remove(1 - channel.min(1))],
            _ => Vec::new(),
        };
    }
    Insta360Meta {
        model: "Insta360 X6".into(),
        calib_string,
        lenses,
        window_crop: Some(X6_DEFAULT_WINDOW),
        dimension: Some((take.width, take.height)),
        frame_rate: Some(take.fps_setting),
        track_order_reversed: Some(true),
        single_lens: single.is_some(),
        stated_frame: stated_frame(take),
        ..Default::default()
    }
}

/// How far from a lens's principal point its image reaches on the sensor,
/// in the pixels of the factory calibration's frame, with room to spare.
///
/// The lens's picture does not fill a frame that holds its whole view: it
/// fades out towards a rim, and past the rim no light of the scene falls,
/// only stray light and the sensor's own noise. Measured on takes of three
/// frame sizes (the brightest 16 × 16 block of every ring around the
/// principal point, over frames spread through a take): the last ring that
/// holds light ends 4048 to 4080 pixels out, on both lenses alike, where
/// the lens model puts a ray of more than 110° from the axis. This is that,
/// and a little over one percent more.
pub const IMAGE_CIRCLE: f64 = 4128.0;

/// The middle of a lens's half of the calibration's frame: where its
/// principal point is taken to be when the take has no lens record.
const SENSOR_MIDDLE: f64 = 3872.0;

/// The part of a take's frames its lenses put a picture on: what lies
/// within [`IMAGE_CIRCLE`] of each lens's principal point. A frame that
/// holds the lens's whole view has four corners outside it; a frame that
/// is the middle of the sensor has nothing outside.
#[derive(Debug, Clone, PartialEq)]
pub struct PictureArea {
    width: usize,
    height: usize,
    /// For each channel, each row's samples inside: `from..to`.
    rows: Vec<Vec<(u32, u32)>>,
    /// For each channel, its circle in the frame's own samples: the centre,
    /// and how far it reaches across and down.
    circles: Vec<[f64; 4]>,
}

impl PictureArea {
    /// The area of the take's frames: each channel's lens through the place
    /// the take states for its frame ([`lens_meta`]).
    pub fn of(take: &Take) -> Self {
        let (width, height) = (take.layout.width(), take.layout.height());
        let meta = lens_meta(take);
        let geom = meta.stream_geometry(width as u32, height as u32, X6ViewChoice::Auto);
        let circles: Vec<[f64; 4]> = (0..take.layout.channels()).map(|channel| {
            // Channel 0 is the lens string's second entry; a take of one lens has its one.
            let entry = if meta.lenses.len() >= 2 { meta.lenses.get(1 - channel.min(1)) } else { meta.lenses.first() };
            let (cx, cy) = match entry {
                Some(entry) => geom.principal_point(entry),
                None => ((SENSOR_MIDDLE - geom.origin_x) * geom.scale_x, (SENSOR_MIDDLE - geom.origin_y) * geom.scale_y),
            };
            [cx, cy, IMAGE_CIRCLE * geom.scale_x, IMAGE_CIRCLE * geom.scale_y]
        }).collect();
        Self::within(width, height, circles)
    }

    /// The area inside the given circles (centre, reach across, reach down;
    /// one a channel) of frames `width` by `height`.
    pub fn within(width: usize, height: usize, circles: Vec<[f64; 4]>) -> Self {
        let rows = circles.iter().map(|&[cx, cy, rx, ry]| {
            (0..height).map(|y| {
                let dy = (y as f64 - cy) / ry.max(1e-9);
                if dy.abs() >= 1.0 {
                    return (0, 0);
                }
                let reach = rx * (1.0 - dy * dy).sqrt();
                let from = (cx - reach).ceil().clamp(0.0, width as f64) as u32;
                let to = ((cx + reach).floor() + 1.0).clamp(from as f64, width as f64) as u32;
                (from, to)
            }).collect()
        }).collect();
        Self { width, height, rows, circles }
    }

    /// Samples of one frame (all its channels) that lie outside.
    pub fn outside(&self) -> usize {
        self.rows.iter().flatten().map(|&(from, to)| self.width - (to - from) as usize).sum()
    }

    /// The same as a share of the frame's samples.
    pub fn share_outside(&self) -> f64 {
        self.outside() as f64 / (self.rows.len() * self.width * self.height).max(1) as f64
    }

    /// Each channel's circle in the frame's samples: the centre, and how
    /// far it reaches across and down.
    pub fn circles(&self) -> &[[f64; 4]] {
        &self.circles
    }

    /// The sample at (`x`, `y`) of `channel` lies inside.
    pub fn holds(&self, channel: usize, x: usize, y: usize) -> bool {
        self.rows.get(channel).and_then(|rows| rows.get(y)).is_some_and(|&(from, to)| (from as usize..to as usize).contains(&x))
    }

    /// Set every sample outside to `value`, in a frame whose channels follow
    /// one another.
    pub fn blacken(&self, frame: &mut [u16], value: u16) {
        let per = self.width * self.height;
        for (rows, mosaic) in self.rows.iter().zip(frame.chunks_exact_mut(per)) {
            for (&(from, to), row) in rows.iter().zip(mosaic.chunks_exact_mut(self.width)) {
                row[..from as usize].fill(value);
                row[to as usize..].fill(value);
            }
        }
    }
}

/// How much later than a frame's content time its gyro samples are
/// stamped, ms. A take states none: this is what the camera states in every
/// one of its own video files (`gyro_timestamp`: the same on every body,
/// firmware and recording mode looked at), taken to hold for a take's gyro
/// as well. [`build_imu`] adds it as a camera file's poses have it added
/// (less [`crate::insv_imu::INSV_STUDIO_SAMPLE_LAG_MS`]).
pub const GYRO_OFFSET_MS: f64 = 1.1;

/// Timing of a take's frames, as far as the take states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureTiming {
    /// Sensor readout (first row to last), ms. `None` = not known: no
    /// per-row rolling-shutter correction.
    pub readout_ms: Option<f32>,
    /// From a frame's exposure stamp to the time its centre row is sampled
    /// at, ms.
    pub stamp_to_content_ms: f64,
}

/// The timing of a take's frames, from its sensor record
/// ([`x6b::read_sensor_readout`]): the stamp is the start of the first
/// row's exposure and each row starts a line period after the one above, so
/// the centre row starts half the readout after the stamp and the rows span
/// the readout around it, top row first. A take without the record states
/// nothing: the pose is sampled at the stamp and rows are not corrected.
///
/// Measured on six takes with the camera in the hand (features tracked in
/// the sensor picture against the gyro through the lens model): the fitted
/// readout is the record's (14.26–14.54 ms for its 14.556, 15.38 for
/// 15.356), top row first, and the centre row's content sits 9.9–11.4 ms
/// after the stamp, that is 2.7–3.6 ms beyond half the readout. That rest
/// is half the shutter time plus whatever the gyro's stamps lag by; the
/// take states neither, so neither is added here.
///
/// Takes of a later camera build list each frame's integration time
/// (`exposure.csv`, [`x6b::read_exposures`]). [`build_imu`] adds half of it,
/// frame by frame, to the time above: the centre row's mid-exposure, which
/// is the content time. What then remains is the gyro's own lag, which no
/// take states: [`GYRO_OFFSET_MS`] stands in for it.
pub fn capture_timing(take: &Take) -> CaptureTiming {
    let readout_ms = x6b::read_sensor_readout(take).map(|eyes| (eyes[0].first_to_last_row_ns * 1e-6) as f32);
    CaptureTiming { readout_ms, stamp_to_content_ms: readout_ms.map_or(0.0, |r| r as f64 * 0.5) }
}

/// The accelerometer (m/s²) at each gyro sample's time, and the gyro
/// samples it speaks for: the two streams share a clock, and nearly every
/// stamp. `frames` is the time of the take's first and last frame. Empty,
/// with no samples, when the take has too little of it to be of use.
///
/// The camera opens the two streams one after the other (26 ms apart on a
/// take of 2026-11-17; on some of the next night's the accelerometer came
/// 1.65 s after the gyro, half a second into the frames). A gyro sample up
/// to half a second outside the accelerometer's span takes its nearest
/// reading: gravity is used slowly (the bias estimate, the trim), so that
/// changes nothing. One further out has none (zeros: the pose builder
/// leaves its trim alone there). Where every gyro sample has a reading the
/// accelerometer speaks for all of them; else for those inside its own
/// span, and the gyro's bias is estimated from that stretch.
fn accel_at_gyro(gyro: &[x6b::GyroSample], accel: &[x6b::GyroSample], frames: (u64, u64)) -> (Vec<[f32; 3]>, std::ops::Range<usize>) {
    if accel.len() < 16 || gyro.is_empty() {
        return (Vec::new(), 0..0);
    }
    let slack = 500_000_000;
    let (first, last) = (accel[0].t_ns, accel[accel.len() - 1].t_ns);
    let inside = |from: u64, to: u64| gyro.partition_point(|g| g.t_ns < from)..gyro.partition_point(|g| g.t_ns <= to);
    let read = inside(first.saturating_sub(slack), last + slack);
    let covered = if read.len() == gyro.len() { read.clone() } else { inside(first, last) };
    // Of use when it has something to say about the frames.
    if covered.len() < 16 || gyro[covered.start].t_ns > frames.1 || gyro[covered.end - 1].t_ns < frames.0 {
        return (Vec::new(), 0..0);
    }
    let mut j = 0usize;
    let at = gyro.iter().enumerate().map(|(i, g)| {
        if !read.contains(&i) {
            return [0.0; 3];
        }
        while j + 2 < accel.len() && accel[j + 1].t_ns <= g.t_ns {
            j += 1;
        }
        let (a, b) = (accel[j], accel[j + 1]);
        let span = b.t_ns.saturating_sub(a.t_ns).max(1) as f64;
        let f = ((g.t_ns as f64 - a.t_ns as f64) / span).clamp(0.0, 1.0) as f32;
        [0, 1, 2].map(|k| a.xyz[k] + (b.xyz[k] - a.xyz[k]) * f)
    }).collect();
    (at, covered)
}

/// The gyro's rates and the accelerometer at them as the samples of ONE
/// rate, with that rate's step in seconds: what the 6-axis pass takes (it
/// steps once a sample, each the same length of time).
///
/// A stream whose steps are all alike is handed over as it is, with its
/// mean step. A take's stream may also run slower for a stretch (a camera
/// build's gyro starts seconds ahead of the frames at a fifth of its rate,
/// and falls back to that after the last): read as one rate, the turning
/// of those stretches would count for a fifth of what it was, and the bias
/// estimate follows the error. Such a stream is put on an even grid of its
/// usual step (the median), each value taken between the two samples
/// around it.
fn evenly_sampled(gyro: &[x6b::GyroSample], accel: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, f32) {
    let n = gyro.len();
    let span_s = (gyro[n - 1].t_ns - gyro[0].t_ns) as f64 * 1e-9;
    let mut steps: Vec<u64> = gyro.windows(2).map(|w| w[1].t_ns.saturating_sub(w[0].t_ns)).collect();
    let longest = steps.iter().copied().max().unwrap_or(0);
    let middle = steps.len() / 2;
    let usual = *steps.select_nth_unstable(middle).1;
    if usual == 0 || longest * 2 < usual * 3 {
        return (gyro.iter().map(|g| g.xyz).collect(), accel.to_vec(), (span_s / (n - 1) as f64) as f32);
    }
    let dt = usual as f64 * 1e-9;
    let count = (span_s / dt).floor() as usize + 1;
    let (mut rates, mut acc) = (Vec::with_capacity(count), Vec::with_capacity(count));
    let t0 = gyro[0].t_ns as f64;
    let mut j = 0usize;
    for k in 0..count {
        let t = t0 + k as f64 * usual as f64;
        while j + 2 < n && gyro[j + 1].t_ns as f64 <= t {
            j += 1;
        }
        let (a, b) = (gyro[j].t_ns as f64, gyro[j + 1].t_ns as f64);
        let f = ((t - a) / (b - a).max(1.0)).clamp(0.0, 1.0) as f32;
        rates.push([0, 1, 2].map(|i| gyro[j].xyz[i] + (gyro[j + 1].xyz[i] - gyro[j].xyz[i]) * f));
        acc.push([0, 1, 2].map(|i| accel[j][i] + (accel[j + 1][i] - accel[j][i]) * f));
    }
    (rates, acc, dt as f32)
}

/// Gyro bias in rad/s from the stillest half second of the stream, when one
/// is still enough to tell (else zero: a constant rate error only drifts
/// the heading, which the stabilizer follows).
fn rest_bias(samples: &[InsvImuSample]) -> [f32; 3] {
    const WINDOW: usize = 512;
    const STILL_DPS: f64 = 0.4;
    let mut best: Option<(f64, [f64; 3])> = None;
    for w in samples.chunks(WINDOW).filter(|w| w.len() == WINDOW) {
        let mut mean = [0f64; 3];
        for s in w { for i in 0..3 { mean[i] += s.gyr_dps[i] as f64; } }
        for m in &mut mean { *m /= WINDOW as f64; }
        let var: f64 = w.iter().map(|s| (0..3).map(|i| (s.gyr_dps[i] as f64 - mean[i]).powi(2)).sum::<f64>()).sum::<f64>() / WINDOW as f64;
        if best.as_ref().map_or(true, |b| var < b.0) {
            best = Some((var, mean));
        }
    }
    match best {
        Some((var, mean)) if var.sqrt() < STILL_DPS && mean.iter().all(|m| m.abs() < 3.0) => {
            let r = std::f64::consts::PI / 180.0;
            [(mean[0] * r) as f32, (mean[1] * r) as f32, (mean[2] * r) as f32]
        }
        _ => [0.0; 3],
    }
}

/// Lens calibration and per-frame poses of a take, in the structure every
/// frame-IMU source hands the stabilizer. `stream_w × stream_h` is the size
/// the frames are worked at. Without a usable gyro file the result holds
/// the calibration only (stabilization stays off).
pub fn build_imu(index: &TakeIndex, stream_w: u32, stream_h: u32) -> DjiOsvImu {
    let take = &index.take;
    let meta = lens_meta(take);
    let mut out = crate::insv_imu::insv_calib_only(&meta, stream_w, stream_h, X6ViewChoice::Auto);
    let timing = capture_timing(take);
    out.readout_ms = timing.readout_ms;
    let Some(hz) = take.clock_start.map(|c| c.counter_hz).filter(|h| *h > 0) else {
        tracing::warn!("raw take {}: no clock record, the gyro cannot be placed", take.dir.display());
        return out;
    };
    let gyro = match x6b::read_gyro(&take.dir, hz) {
        Ok(g) if g.len() >= 16 => g,
        Ok(g) => {
            tracing::warn!("raw take {}: {} gyro samples, no stabilization", take.dir.display(), g.len());
            return out;
        }
        Err(e) => {
            tracing::warn!("raw take {}: gyro: {e}, no stabilization", take.dir.display());
            return out;
        }
    };
    let deg = 180.0 / std::f32::consts::PI;
    const G: f32 = 9.80665;
    // The accelerometer, where the take has one: the camera's own, on the
    // gyro's axes (it holds its direction in the gyro-integrated frame to
    // 1–2° through a hand-held minute, and reads as the camera's own video
    // files' does at rest).
    let frames = (index.pairs[0].exposure_ns[0], index.pairs[index.pairs.len() - 1].exposure_ns[0]);
    let (accel, covered) = accel_at_gyro(&gyro, &x6b::read_accel(&take.dir, hz), frames);
    let samples: Vec<InsvImuSample> = gyro.iter().enumerate()
        .map(|(i, g)| InsvImuSample {
            t_us: g.t_ns / 1000,
            acc_g: accel.get(i).map_or([0.0; 3], |a| [a[0] / G, a[1] / G, a[2] / G]),
            gyr_dps: [g.xyz[0] * deg, g.xyz[1] * deg, g.xyz[2] * deg],
        })
        .collect();
    // With an accelerometer the poses are built as the camera's own video
    // files' are: the gyro's bias from a 6-axis pass, its integration, and
    // the slow trim that keeps a long take level. Without one: the gyro
    // alone, its bias from a still stretch if the take has one.
    let (bias, trim_tau_s) = if accel.is_empty() {
        (rest_bias(&samples), 0.0)
    } else {
        // The bias: from the stretch the accelerometer speaks for, as samples of one rate.
        let (rates, accel, dt) = evenly_sampled(&gyro[covered.clone()], &accel[covered.clone()]);
        if rates.len() != covered.len() {
            tracing::info!("raw take {}: the motion stream runs at more than one rate ({} samples): put on an even grid of {:.4} ms ({} samples) for the gyro's bias",
                take.dir.display(), covered.len(), dt * 1e3, rates.len());
        }
        (vr180_core::gyro::vqf::run(&rates, &accel, None, dt).bias_rad_s, crate::insv_imu::INSV_TAU_ACC_S)
    };
    let quats = crate::insv_imu::integrate_orientation_samples(&samples, bias, trim_tau_s);
    let line = crate::insv_imu::QuatTimeline { t_s: gyro.iter().map(|g| g.t_ns as f64 * 1e-9).collect(), quats: &quats };
    let frame_dur = index.dt_s();
    // To the centre row's start, and on by the gyro's offset as for a camera file.
    let gyro_offset_ms = GYRO_OFFSET_MS - crate::insv_imu::INSV_STUDIO_SAMPLE_LAG_MS;
    let lead = (timing.stamp_to_content_ms + gyro_offset_ms) * 1e-3;
    // Each frame's integration time, where the take lists it: half of it
    // takes the sample from the centre row's start to its mid-exposure. A
    // frame the list has no value for keeps the row's start.
    let half_shutter_s = |i: usize, channel: usize| -> Option<f64> { index.shutter_s(i, channel).map(|s| s * 0.5) };
    let fill = |channel: usize, to: &mut DjiOsvImu| {
        to.frame_quats.clear();
        to.high_rate_quats.clear();
        to.gravity.clear();
        for (i, p) in index.pairs.iter().enumerate() {
            let tc = p.exposure_ns[channel] as f64 * 1e-9 + lead + half_shutter_s(i, channel).unwrap_or(0.0);
            to.frame_quats.push(line.at(tc));
            let start = tc - frame_dur * 0.5;
            let n = crate::insv_imu::HR_PER_FRAME;
            to.high_rate_quats.push((0..n).map(|k| line.at(start + frame_dur * k as f64 / n as f64)).collect::<Vec<Quat>>());
            to.gravity.push([0.0, -1.0, 0.0]);
        }
    };
    out.sample_anchor = vr180_fisheye::SampleAnchor::BlockMid;
    fill(0, &mut out);
    let (first, last) = (gyro[0].t_ns as f64 * 1e-9, gyro[gyro.len() - 1].t_ns as f64 * 1e-9);
    let (f0, f1) = (index.pairs[0].exposure_ns[0] as f64 * 1e-9, index.pairs[index.pairs.len() - 1].exposure_ns[0] as f64 * 1e-9);
    tracing::info!(
        "raw take {}: {} gyro samples @ {:.1} Hz, bias {:?} rad/s, {}, covers {:+.3} s before the first frame to {:+.3} s after the last; {} frame poses, {}",
        take.dir.display(), gyro.len(), (gyro.len() - 1) as f64 / (last - first).max(1e-9), bias,
        if accel.is_empty() {
            "no accelerometer".to_string()
        } else if covered.len() < gyro.len() {
            format!("accelerometer for {} of them (trim {trim_tau_s} s)", covered.len())
        } else {
            format!("accelerometer (trim {trim_tau_s} s)")
        },
        f0 - first, last - f1, out.frame_quats.len(),
        match timing.readout_ms {
            Some(r) => format!("readout {r:.3} ms, centre row {:.3} ms after the stamp, the gyro read {gyro_offset_ms:.2} ms on", timing.stamp_to_content_ms),
            None => format!("no readout record, the gyro read {gyro_offset_ms:.2} ms on"),
        });
    // The shutter times in use, or that the take has none.
    let known: Vec<f64> = (0..index.pairs.len()).filter_map(|i| half_shutter_s(i, 0)).map(|h| h * 2e3).collect();
    if known.is_empty() {
        tracing::info!("raw take {}: no shutter record, the pose is sampled where the centre row starts exposing", take.dir.display());
    } else {
        let (lo, hi) = known.iter().fold((f64::MAX, 0f64), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        tracing::info!("raw take {}: shutter {lo:.3}–{hi:.3} ms from the take for {} of {} frames, the pose is sampled at the centre row's mid-exposure",
            take.dir.display(), known.len(), index.pairs.len());
    }
    if first > f0 || last < f1 {
        tracing::warn!("raw take {}: the gyro does not cover every frame; the uncovered ones hold the nearest pose", take.dir.display());
    }
    out
}

// ── Frames ─────────────────────────────────────────────────────────────

/// Demosaiced eyes handed back by frames that are done with them: a
/// frame's eye textures, one per channel.
#[derive(Default)]
struct EyePool {
    free: Mutex<Vec<Vec<wgpu::Texture>>>,
}

/// Returns a frame's eye textures to the pool with its last holder.
struct EyeLoan {
    pool: Arc<EyePool>,
    eyes: Option<Vec<wgpu::Texture>>,
}

impl Drop for EyeLoan {
    fn drop(&mut self) {
        if let (Some(eyes), Ok(mut free)) = (self.eyes.take(), self.pool.free.lock()) {
            // A frame of 3840² eyes is 236 MB, one 7680² eye twice that:
            // fewer of the large ones are kept.
            let keep = if eyes.first().is_some_and(|e| e.width() as usize > EYE_SIDE) { 4 } else { 12 };
            if free.len() < keep {
                free.push(eyes);
            }
        }
    }
}

/// Where an export takes a take's enhanced frames from: the folder its
/// camera-original DNGs were written into ([`crate::sensor_dng`]) and a raw
/// converter then enhanced in place ([`crate::enhanced_dng`]), and the
/// stretch of the take the export covers (seconds on its timeline; `None`
/// = from the start, to the end).
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedSource {
    pub dir: PathBuf,
    pub t_in: Option<f64>,
    pub t_out: Option<f64>,
}

/// A take's enhanced frames in use: the folder, the side of the pictures
/// the converter left in its files, and which frames it holds a file of.
struct EnhancedFrames {
    dir: PathBuf,
    side: usize,
    /// The requests each channel's folder holds a file of.
    held: [HashSet<u32>; 2],
    /// Eye pictures taken from the folder, and from the capture instead.
    used: usize,
    missing: usize,
}

/// How many of a stretch's largest files are opened for the enhanced
/// pictures' size.
const ENHANCED_SEARCH: usize = 8;

/// The frame files of an enhanced-frames folder: each channel's requests
/// with their file sizes, in order. Only files named as
/// [`crate::sensor_dng::frame_file`] names them count.
fn enhanced_listing(dir: &Path) -> [Vec<(u32, u64)>; 2] {
    std::array::from_fn(|channel| {
        let mut files = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir.join(crate::sensor_dng::channel_dir(channel))) else { return files };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(request) = name.strip_suffix(".dng").and_then(|stem| stem.parse::<u32>().ok()) else { continue };
            if crate::sensor_dng::frame_file(channel, request).ends_with(&format!("/{name}")) {
                files.push((request, entry.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
        files.sort_unstable();
        files
    })
}

/// Which files of a listing to open for the enhanced pictures' size, as
/// `(channel, request)`, and how many files the stretch `wanted` has. A
/// file keeps its mosaic and gains the enhanced picture, so the enhanced
/// ones are the largest: those come first, then the stretch's first, middle
/// and last file.
fn enhanced_candidates(listing: &[Vec<(u32, u64)>; 2], wanted: impl Fn(u32) -> bool) -> (usize, Vec<(usize, u32)>) {
    let mut files: Vec<(usize, u32, u64)> = listing.iter().enumerate()
        .flat_map(|(channel, files)| files.iter().filter(|f| wanted(f.0)).map(move |f| (channel, f.0, f.1)))
        .collect();
    let count = files.len();
    let by_name: Vec<(usize, u32)> = [0, count / 2, count.saturating_sub(1)].iter()
        .filter_map(|&i| files.get(i).map(|f| (f.0, f.1))).collect();
    files.sort_by(|a, b| b.2.cmp(&a.2));
    let mut out: Vec<(usize, u32)> = files.iter().take(ENHANCED_SEARCH).map(|f| (f.0, f.1)).collect();
    for file in by_name {
        if !out.contains(&file) {
            out.push(file);
        }
    }
    (count, out)
}

/// What an enhanced-frames folder holds for a stretch of a take.
struct EnhancedFind {
    listing: [Vec<(u32, u64)>; 2],
    /// Frames of the stretch, the files the folder has for them, and how
    /// many of those were opened.
    frames: usize,
    files: usize,
    looked: usize,
    /// The first enhanced picture met.
    picture: Option<crate::enhanced_dng::EnhancedPicture>,
}

impl EnhancedFind {
    /// The side of the enhanced pictures, where they can stand in for the
    /// take's frames: square, a whole multiple of the mosaic's side.
    fn side(&self) -> Option<u32> {
        self.picture.as_ref().filter(|p| p.width == p.height && p.width as usize % EYE_SIDE == 0 && p.width > 0).map(|p| p.width)
    }
}

/// Look through `source`'s folder for the enhanced pictures of its stretch
/// of the take: from the frame shown at the stretch's start to the one
/// shown at its end.
fn find_enhanced(index: &TakeIndex, source: &EnhancedSource) -> EnhancedFind {
    let listing = enhanced_listing(&source.dir);
    let frames = index.pairs.len();
    let first = source.t_in.map(|t| index.frame_at(t.max(0.0))).unwrap_or(0);
    let last = source.t_out.map(|t| index.frame_at(t.max(0.0))).unwrap_or(frames).min(frames.saturating_sub(1));
    let wanted: HashSet<u32> = index.pairs.iter().skip(first).take((last + 1).saturating_sub(first)).map(|p| p.request).collect();
    let (files, candidates) = enhanced_candidates(&listing, |request| wanted.contains(&request));
    let picture = candidates.iter().find_map(|&(channel, request)|
        crate::enhanced_dng::find(&source.dir.join(crate::sensor_dng::frame_file(channel, request))).ok().flatten());
    EnhancedFind { listing, frames: wanted.len(), files, looked: candidates.len(), picture }
}

/// The side of the enhanced pictures an export of `source`'s stretch of the
/// take at `path` reads its frames from, or `None` when it reads the
/// capture: what [`X6RawIter::with_enhanced_frames`] goes by, for a caller
/// that sizes the export before opening it.
pub fn enhanced_side(path: &Path, source: &EnhancedSource) -> Option<u32> {
    // Pictures this build cannot decode are not there for it.
    if !crate::enhanced_dng::decodable() {
        return None;
    }
    // No folder: nothing to open the take for.
    if !(0..2).any(|channel| source.dir.join(crate::sensor_dng::channel_dir(channel)).is_dir()) {
        return None;
    }
    let index = open(path).ok()?;
    // Only a paired take's frames have enhanced pictures to stand in for them.
    if index.take.layout != Layout::Paired {
        return None;
    }
    find_enhanced(&index, source).side()
}

/// A take's Camera Raw settings, found in the folder of its camera-original
/// DNGs ([`camera_raw_settings`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraRawSettings {
    /// The file they were read from.
    pub file: PathBuf,
    pub develop: crate::raw_export::ReadDevelop,
    /// The same frame's other eye has settings of its own that are not these.
    pub eyes_differ: bool,
}

/// How many of the nearest frames' files are opened for settings.
const SETTINGS_SEARCH: usize = 12;

/// The Camera Raw settings a raw converter left for the take at `path`, in
/// the folder `dir` of its camera-original DNGs: those of the frame shown at
/// `t_s` (channel 0's file, then channel 1's), or, when that frame has no
/// file with settings, of the nearest frame that has. `Ok(None)`: the
/// folder holds no file with such settings near that frame.
pub fn camera_raw_settings(path: &Path, dir: &Path, t_s: f64) -> Result<Option<CameraRawSettings>> {
    let listing = enhanced_listing(dir);
    if listing.iter().all(|files| files.is_empty()) {
        return Ok(None);
    }
    let index = open(path)?;
    let wanted = index.pairs.get(index.frame_at(t_s.max(0.0))).map_or(0, |p| p.request);
    // Every listed frame, nearest first; channel 0 before channel 1.
    let mut requests: Vec<u32> = listing.iter().flat_map(|files| files.iter().map(|f| f.0)).collect();
    requests.sort_unstable();
    requests.dedup();
    requests.sort_by_key(|r| r.abs_diff(wanted));
    let read = |channel: usize, request: u32| -> Option<(PathBuf, crate::raw_export::ReadDevelop)> {
        if !listing[channel].iter().any(|f| f.0 == request) {
            return None;
        }
        let file = dir.join(crate::sensor_dng::frame_file(channel, request));
        let xmp = crate::enhanced_dng::read_xmp(&file).ok().flatten()?;
        Some((file, crate::raw_export::read_develop_xmp(&xmp)?))
    };
    for &request in requests.iter().take(SETTINGS_SEARCH) {
        for channel in 0..2 {
            if let Some((file, develop)) = read(channel, request) {
                let eyes_differ = channel == 0 && read(1, request).is_some_and(|(_, other)| other.settings != develop.settings);
                return Ok(Some(CameraRawSettings { file, develop, eyes_differ }));
            }
        }
    }
    Ok(None)
}

/// The frames of a raw take as fisheye pairs.
///
/// `swap_eyes` has the dual-stream iterators' meaning: `true` puts channel 1
/// on the left (the default for a VR180-modded body, see
/// [`crate::SourceKind::dual_stream_iter_swap`]). `half` asks for half-size
/// eyes (one pixel per 2×2 site group, a preview's source); `step` > 1 skips
/// frames without decoding them (every frame is independent).
pub struct X6RawIter {
    index: Arc<TakeIndex>,
    file: File,
    next: usize,
    step: usize,
    swap_eyes: bool,
    half: bool,
    gpu: Option<Arc<Device>>,
    record: Vec<u8>,
    /// The frame decoded last: the take's channels one after the other.
    samples: Vec<u16>,
    /// The take's mosaic ([`bayer_of`]).
    bayer: BayerLayout,
    /// One mosaic texture per channel.
    mosaics: Vec<wgpu::Texture>,
    pool: Arc<EyePool>,
    /// Stop after this many frames from the start of the take (0 = all).
    frame_limit: usize,
    /// Without a device: hand out the mosaics themselves, not pictures.
    mosaics_out: bool,
    /// Frames a raw converter enhanced, used in place of the capture's.
    enhanced: Option<EnhancedFrames>,
    /// Read and unpack the frames ahead of the consumer ([`ReadAhead`]).
    read_ahead: bool,
    ahead: Option<ReadAhead>,
}

/// A take's frames read from the capture and unpacked on a thread of their
/// own, ahead of the consumer: reading a pair (22 MB at 60 fps, more at 30)
/// and unpacking its two mosaics take 13 ms from a fast disk and 35 from a
/// network share, which an export otherwise waits for between two frames'
/// GPU work (a take of one lens: its one mosaic, twice a pair's samples).
/// The thread has its own handle on the capture and touches no GPU; the
/// frames come out in order, each exactly as [`X6RawIter::decode_frame`]
/// decodes it.
struct ReadAhead {
    /// The unpacked pairs, in order: the frame's index and its two mosaics.
    frames: std::sync::mpsc::Receiver<Result<(usize, Vec<u16>)>>,
    /// Buffers handed back for the next frames (59 MB each for a pair, 118
    /// for one lens's frame: not allocated per frame).
    spare: std::sync::mpsc::Sender<Vec<u16>>,
    /// The frame the next message holds.
    next: usize,
}

/// Pairs unpacked and waiting at most. With the one being unpacked and the
/// one in use, four buffers of 59 MB.
const READ_AHEAD_FRAMES: usize = 2;

impl ReadAhead {
    /// Start reading at frame `first`, every `step`-th frame, up to `end`.
    fn start(index: &Arc<TakeIndex>, first: usize, step: usize, end: usize) -> Result<Self> {
        let mut file = File::open(index.take.capture_path())?;
        let (tx, frames) = std::sync::mpsc::sync_channel::<Result<(usize, Vec<u16>)>>(READ_AHEAD_FRAMES);
        let (spare, spares) = std::sync::mpsc::channel::<Vec<u16>>();
        let index = index.clone();
        std::thread::Builder::new().name("raw-take-reader".into()).spawn(move || {
            let mut record = Vec::new();
            let mut i = first;
            while i < end {
                let Some(entry) = index.pairs.get(i).copied() else { break };
                let count = index.take.layout.samples();
                let mut samples = spares.try_recv().ok().filter(|b| b.len() == count)
                    .unwrap_or_else(|| vec![0u16; count]);
                let read = decode_pair(&mut file, &entry, index.take.format(), Channels::Both, &mut record, &mut samples, 0);
                let failed = read.is_err();
                // The consumer went away (the export's end, a seek): stop.
                if tx.send(read.map(|()| (i, samples))).is_err() || failed {
                    break;
                }
                i += step;
            }
        }).map_err(take_err)?;
        Ok(Self { frames, spare, next: first })
    }
}

impl Drop for X6RawIter {
    fn drop(&mut self) {
        if let Some(e) = &self.enhanced {
            if e.used + e.missing > 0 {
                tracing::info!("raw take: {} eye pictures from the enhanced frames, {} from the capture", e.used, e.missing);
            }
        }
    }
}

impl std::fmt::Debug for X6RawIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("X6RawIter")
            .field("take", &self.index.take.dir)
            .field("next", &self.next)
            .field("step", &self.step)
            .field("half", &self.half)
            .field("gpu", &self.gpu.is_some())
            .finish_non_exhaustive()
    }
}

impl X6RawIter {
    pub fn new(path: &Path, swap_eyes: bool, half: bool, step: u32, gpu: Option<Arc<Device>>) -> Result<Self> {
        let index = open(path)?;
        let file = File::open(index.take.capture_path())?;
        let (count, bayer) = (index.take.layout.samples(), bayer_of(&index.take));
        Ok(Self {
            index, file, next: 0, step: step.max(1) as usize, swap_eyes, half, gpu,
            record: Vec::new(), samples: vec![0u16; count], bayer, mosaics: Vec::new(),
            pool: Arc::default(), frame_limit: 0, mosaics_out: false, enhanced: None,
            read_ahead: false, ahead: None,
        })
    }

    /// Read and unpack the frames on a thread of their own, ahead of the
    /// consumer ([`ReadAhead`]): for a consumer that takes every frame in
    /// order (an export). The frames are the same; a seek starts over from
    /// the new place.
    pub fn with_read_ahead(mut self) -> Self {
        self.read_ahead = true;
        self
    }

    /// Frame `i`'s two mosaics into the iterator's buffer: from the reader
    /// thread where one runs ([`Self::with_read_ahead`]), else decoded here.
    fn fetch_frame(&mut self, i: usize) -> Result<()> {
        if !self.read_ahead {
            return self.decode_frame(i).map(|_| ());
        }
        if !self.ahead.as_ref().is_some_and(|a| a.next == i) {
            let end = if self.frame_limit > 0 { self.frame_limit.min(self.index.frames()) } else { self.index.frames() };
            self.ahead = Some(ReadAhead::start(&self.index, i, self.step, end)?);
        }
        let ahead = self.ahead.as_mut().expect("just started");
        let (got, samples) = ahead.frames.recv().map_err(|_| take_err("the take's reader stopped early"))??;
        if got != i {
            return Err(take_err(format!("the take's reader handed frame {got} where {i} was due")));
        }
        let used = std::mem::replace(&mut self.samples, samples);
        let _ = ahead.spare.send(used);
        ahead.next = i + self.step;
        Ok(())
    }

    /// Take the frames from `source`'s folder where a raw converter has
    /// enhanced them. The eyes become the enhanced pictures' size (twice the
    /// mosaic's for Super Resolution); a frame the folder has no enhanced
    /// picture of comes from the capture, enlarged to match. Nothing changes
    /// when no frame of the export's stretch has an enhanced picture: the
    /// capture's frames are then used as they are.
    pub fn with_enhanced_frames(mut self, source: Option<&EnhancedSource>) -> Self {
        let Some(source) = source else { return self };
        let dir = &source.dir;
        if self.index.take.layout != Layout::Paired {
            tracing::info!("raw take: enhanced frames are read for a paired take of 3840 x 3840 only: the capture's frames are used");
            return self;
        }
        if !crate::enhanced_dng::decodable() {
            tracing::info!("raw take: enhanced frames are not read on this system (no JPEG XL decoder in this build): the capture's frames are used");
            return self;
        }
        let found = find_enhanced(&self.index, source);
        match (found.side(), &found.picture) {
            (Some(side), _) => {
                tracing::info!("raw take {}: enhanced frames of {side}x{side} in {} ({} files for the {} frames of this stretch)",
                    self.index.take.dir.display(), dir.display(), found.files, found.frames);
                self.enhanced = Some(EnhancedFrames {
                    dir: dir.clone(), side: side as usize,
                    held: found.listing.map(|files| files.into_iter().map(|f| f.0).collect()),
                    used: 0, missing: 0,
                });
                self.half = false;
            }
            (None, Some(p)) => tracing::warn!("raw take: the enhanced frames in {} are {}x{}, not a whole multiple of the take's: not used", dir.display(), p.width, p.height),
            (None, None) if found.files > 0 => tracing::warn!("raw take: none of the {} files of this stretch in {} holds an enhanced picture ({} looked at): the capture's frames are used", found.files, dir.display(), found.looked),
            (None, None) => tracing::info!("raw take: {} holds no frame of this stretch: the capture's frames are used", dir.display()),
        }
        self
    }

    /// The side of the enhanced pictures in use, if any are.
    pub fn enhanced_side(&self) -> Option<u32> {
        self.enhanced.as_ref().map(|e| e.side as u32)
    }

    /// Hand out each frame as its two mosaics ([`MOSAIC_BIT_DEPTH`] pairs)
    /// instead of demosaicing: for a consumer on a thread that must not
    /// touch the GPU, whose renderer then calls [`demosaic_pair`].
    pub fn mosaic_pairs(mut self) -> Self {
        self.mosaics_out = true;
        self.gpu = None;
        self.half = false;
        self
    }

    /// Stop after `frames` frames of the take (0 = no limit).
    pub fn with_frame_limit(mut self, frames: usize) -> Self {
        self.frame_limit = frames;
        self
    }

    pub fn index(&self) -> &Arc<TakeIndex> {
        &self.index
    }

    /// Width and height of the eyes handed out.
    fn eye_size(&self) -> (u32, u32) {
        let layout = self.index.take.layout;
        let (w, h) = (layout.width() as u32, layout.height() as u32);
        match &self.enhanced {
            Some(e) => (e.side as u32, e.side as u32),
            None if self.half => (w / 2, h / 2),
            None => (w, h),
        }
    }

    /// The take holds one lens's frames: a pair is its left eye alone.
    fn one_eye(&self) -> bool {
        self.index.take.layout.channels() == 1
    }

    /// Frame `i`'s two channels from the enhanced frames' folder, as 16-bit
    /// camera values (RGBA): channel 0, then channel 1. A channel the folder
    /// has no enhanced picture of is the capture's frame, demosaiced and
    /// enlarged to the same size.
    fn enhanced_channels(&mut self, i: usize) -> Result<[Vec<u16>; 2]> {
        let request = self.index.pairs.get(i).ok_or_else(|| take_err(format!("frame {i} is past the take's end")))?.request;
        let (dir, side, held) = match &self.enhanced {
            Some(e) => (e.dir.clone(), e.side, [e.held[0].contains(&request), e.held[1].contains(&request)]),
            None => return Err(take_err("no enhanced frames")),
        };
        // The two channels' pictures decode side by side.
        let channel = |channel: usize| -> Result<Vec<u16>> {
            // A frame the folder does not hold is not worth a line of its own.
            if !held[channel] {
                return Ok(Vec::new());
            }
            let path = dir.join(crate::sensor_dng::frame_file(channel, request));
            match crate::enhanced_dng::find(&path) {
                Ok(Some(p)) if p.width as usize == side && p.height as usize == side => {
                    let mut px = vec![0u16; side * side * 4];
                    p.decode_rgba16(&path, &mut px)?;
                    return Ok(px);
                }
                Ok(Some(p)) => tracing::warn!("{}: its enhanced picture is {}x{}, the others {side}x{side}: the capture's frame is used", path.display(), p.width, p.height),
                Ok(None) => {}
                Err(e) => tracing::warn!("{}: {e}: the capture's frame is used", path.display()),
            }
            Ok(Vec::new())
        };
        let (first, second) = if held[0] && held[1] {
            std::thread::scope(|scope| {
                let other = scope.spawn(|| channel(1));
                (channel(0), other.join().unwrap_or_else(|_| Err(take_err("the enhanced picture's decoder stopped"))))
            })
        } else {
            (channel(0), channel(1))
        };
        let mut eyes: [Vec<u16>; 2] = [first?, second?];
        let absent = eyes.iter().filter(|e| e.is_empty()).count();
        if absent > 0 {
            self.decode_frame(i)?;
            for (channel, eye) in eyes.iter_mut().enumerate().filter(|(_, e)| e.is_empty()) {
                let rgb = crate::bayer::demosaic(&self.samples[channel * EYE_SAMPLES..(channel + 1) * EYE_SAMPLES], EYE_SIDE, EYE_SIDE, &BAYER, false);
                *eye = enlarged(&rgb, EYE_SIDE, side / EYE_SIDE);
            }
        }
        if let Some(e) = self.enhanced.as_mut() {
            if absent > 0 && e.missing == 0 {
                tracing::warn!("raw take: request {request} has no enhanced picture for {absent} of its two eyes in {}: the capture's frame stands in, enlarged (further ones are counted, not listed)", e.dir.display());
            }
            e.used += 2 - absent;
            e.missing += absent;
        }
        Ok(eyes)
    }

    /// The mosaic of the frame decoded last: channel 0 then channel 1 (a
    /// take of one lens: its one channel).
    pub fn last_mosaic(&self) -> &[u16] {
        &self.samples
    }

    /// Decode frame `i`'s mosaics into the iterator's buffer.
    pub fn decode_frame(&mut self, i: usize) -> Result<&[u16]> {
        let entry = *self.index.pairs.get(i).ok_or_else(|| take_err(format!("frame {i} is past the take's end")))?;
        decode_pair(&mut self.file, &entry, self.index.take.format(), Channels::Both, &mut self.record, &mut self.samples, 0)?;
        Ok(&self.samples)
    }

    fn gpu_pair(&mut self, dev: &Arc<Device>) -> Result<GpuPair> {
        let layout = self.index.take.layout;
        let (w, h, channels, per) = (layout.width() as u32, layout.height() as u32, layout.channels(), layout.eye_samples());
        let (out_w, out_h) = self.eye_size();
        if self.mosaics.len() != channels {
            self.mosaics = (0..channels).map(|_| dev.new_bayer_texture(w, h)).collect();
        }
        for (mosaic, samples) in self.mosaics.iter().zip(self.samples.chunks(per)) {
            dev.write_bayer_texture(mosaic, samples);
        }
        let pooled = self.pool.free.lock().ok().and_then(|mut f| f.pop())
            .filter(|eyes| eyes.len() == channels && eyes.iter().all(|e| (e.width(), e.height()) == (out_w, out_h)));
        let eyes = pooled.unwrap_or_else(|| (0..channels).map(|_| dev.new_linear_eye_texture(out_w, out_h)).collect());
        for (mosaic, eye) in self.mosaics.iter().zip(&eyes) {
            dev.demosaic_bayer(mosaic, &self.bayer, eye)?;
        }
        let (left, right) = match eyes.as_slice() {
            [e0, e1] if self.swap_eyes => (e1.clone(), e0.clone()),
            [e0, e1] => (e0.clone(), e1.clone()),
            // One eye: it stands in both places.
            eyes => (eyes[0].clone(), eyes[0].clone()),
        };
        Ok(GpuPair { left, right, one_eye: channels == 1, keep: Box::new(EyeLoan { pool: self.pool.clone(), eyes: Some(eyes) }) })
    }

    fn cpu_pair(&self) -> (Vec<u8>, Vec<u8>) {
        let layout = self.index.take.layout;
        let (w, h, per) = (layout.width(), layout.height(), layout.eye_samples());
        // The GPU demosaic's twin, one channel at a time.
        let eye = |channel: usize| -> Vec<u8> {
            let rgb = crate::bayer::demosaic(&self.samples[channel * per..(channel + 1) * per], w, h, &self.bayer, self.half);
            let mut bytes = Vec::with_capacity(rgb.len() * 8);
            for px in &rgb {
                for v in [px[0], px[1], px[2], 1.0] {
                    bytes.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes());
                }
            }
            bytes
        };
        if self.one_eye() {
            return (eye(0), Vec::new());
        }
        let (c0, c1) = (eye(0), eye(1));
        if self.swap_eyes { (c1, c0) } else { (c0, c1) }
    }
}

/// A demosaiced picture (`side × side`, camera values) enlarged `factor`
/// times, bilinearly, as 16-bit RGBA: what stands in for a frame a raw
/// converter did not enhance.
fn enlarged(rgb: &[[f32; 3]], side: usize, factor: usize) -> Vec<u16> {
    let out_side = side * factor;
    let mut out = vec![u16::MAX; out_side * out_side * 4];
    let f = factor as f32;
    crate::bayer::fill_rows(&mut out, out_side * 4, |y, row| {
        let sy = ((y as f32 + 0.5) / f - 0.5).clamp(0.0, side as f32 - 1.0);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(side - 1);
        for (x, px) in row.chunks_exact_mut(4).enumerate() {
            let sx = ((x as f32 + 0.5) / f - 0.5).clamp(0.0, side as f32 - 1.0);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(side - 1);
            let (a, b, c, d) = (rgb[y0 * side + x0], rgb[y0 * side + x1], rgb[y1 * side + x0], rgb[y1 * side + x1]);
            for k in 0..3 {
                let v = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
                px[k] = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
    });
    out
}

impl FisheyePairIter for X6RawIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        let i = self.next;
        if i >= self.index.frames() || (self.frame_limit > 0 && i >= self.frame_limit) {
            return Ok(None);
        }
        let (eye_w, eye_h) = self.eye_size();
        let pts_s = self.index.pts_s(i);
        if self.enhanced.is_some() && !self.mosaics_out {
            // Enhanced pictures are square.
            let side = eye_w;
            let [c0, c1] = self.enhanced_channels(i)?;
            self.next = i + self.step;
            let (left, right) = if self.swap_eyes { (c1, c0) } else { (c0, c1) };
            if let Some(dev) = self.gpu.clone() {
                let upload = |px: &[u16]| {
                    let tex = dev.new_camera_eye_texture(side, side);
                    dev.write_camera_eye_texture(&tex, px);
                    tex
                };
                let pair = GpuPair { left: upload(&left), right: upload(&right), one_eye: false, keep: Box::new(()) };
                return Ok(Some(FisheyePair {
                    left: Vec::new(), right: Vec::new(), eye_w: side, eye_h: side, bit_depth: 16, pts_s,
                    gpu: Some(Arc::new(pair)),
                }));
            }
            let bytes = |px: &[u16]| bytemuck::cast_slice::<u16, u8>(px).to_vec();
            return Ok(Some(FisheyePair { left: bytes(&left), right: bytes(&right), eye_w: side, eye_h: side, bit_depth: 16, pts_s, gpu: None }));
        }
        self.fetch_frame(i)?;
        self.next = i + self.step;
        if let Some(dev) = self.gpu.clone() {
            let pair = self.gpu_pair(&dev)?;
            return Ok(Some(FisheyePair {
                left: Vec::new(), right: Vec::new(), eye_w, eye_h, bit_depth: 16, pts_s,
                gpu: Some(Arc::new(pair)),
            }));
        }
        if self.mosaics_out {
            let per = self.index.take.layout.eye_samples();
            let bytes = |ch: usize| bytemuck::cast_slice::<u16, u8>(&self.samples[ch * per..(ch + 1) * per]).to_vec();
            let (left, right) = match (self.one_eye(), self.swap_eyes) {
                (true, _) => (bytes(0), Vec::new()),
                (false, true) => (bytes(1), bytes(0)),
                (false, false) => (bytes(0), bytes(1)),
            };
            return Ok(Some(FisheyePair { left, right, eye_w, eye_h, bit_depth: MOSAIC_BIT_DEPTH, pts_s, gpu: None }));
        }
        let (left, right) = self.cpu_pair();
        Ok(Some(FisheyePair { left, right, eye_w, eye_h, bit_depth: 16, pts_s, gpu: None }))
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        self.next = self.index.frame_at(target_s.max(0.0));
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        self.eye_size()
    }

    fn source_decimation(&self) -> u32 {
        self.step as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A take says which way up the camera was through the first seconds of
    /// its accelerometer record.
    #[test]
    fn a_take_says_which_way_up_the_camera_was() {
        let dir = std::env::temp_dir().join(format!("vr180-mount-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(x6b::CLIP_FILE), r#"{"layout":"paired-bayer-v1","fps_setting":60,"size":[3840,3840]}"#).unwrap();
        // No record: nothing to go by.
        assert_eq!(mounted_upside_down(&dir), None);
        let record = |x: f32| {
            let mut text = String::from("counter_ticks,x,y,z\n");
            for k in 0..400u64 {
                text.push_str(&format!("{},{},{},{}\n", 330359180137 + k * 19300, x + 0.01 * (k % 3) as f32, -0.2, 0.02));
            }
            text
        };
        std::fs::write(dir.join(x6b::ACCEL_FILE), record(-9.84)).unwrap();
        assert_eq!(mounted_upside_down(&dir), Some(false));
        std::fs::write(dir.join(x6b::ACCEL_FILE), record(9.86)).unwrap();
        assert_eq!(mounted_upside_down(&dir), Some(true));
        // One of the take's own files names the take as well.
        assert_eq!(mounted_upside_down(&dir.join(x6b::CAPTURE_FILE)), Some(true));
        // On its side: the record says nothing.
        std::fs::write(dir.join(x6b::ACCEL_FILE), record(0.4)).unwrap();
        assert_eq!(mounted_upside_down(&dir), None);
        // A record that holds a header and no more.
        std::fs::write(dir.join(x6b::ACCEL_FILE), "counter_ticks,x,y,z\n").unwrap();
        assert_eq!(mounted_upside_down(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The part of a frame a lens puts a picture on: a circle around its
    /// principal point, cut by the frame where the frame is the smaller.
    #[test]
    fn the_picture_area_is_the_lens_circle_within_the_frame() {
        // A circle well inside a frame: what lies outside it is the frame less the disc.
        let area = PictureArea::within(400, 300, vec![[200.0, 150.0, 100.0, 100.0]]);
        let disc = std::f64::consts::PI * 100.0 * 100.0;
        assert!((area.outside() as f64 - (400.0 * 300.0 - disc)).abs() < 300.0, "{}", area.outside());
        assert!(area.holds(0, 200, 150) && area.holds(0, 101, 150) && area.holds(0, 299, 150) && area.holds(0, 200, 51));
        assert!(!area.holds(0, 99, 150) && !area.holds(0, 301, 150) && !area.holds(0, 200, 49) && !area.holds(0, 0, 0) && !area.holds(0, 280, 230));
        assert!(!area.holds(1, 200, 150) && !area.holds(0, 200, 300));
        // A frame that holds a lens's whole view, scaled: the circle is cut on all four sides and only the corners
        // lie outside (the two channels' centres apart, as two lenses' are).
        let whole = PictureArea::within(6000, 6000, vec![[3007.5, 3006.3, 3145.1, 3145.1], [2992.5, 3008.1, 3145.1, 3145.1]]);
        assert!((whole.share_outside() - 0.157).abs() < 0.002, "{}", whole.share_outside());
        assert!(whole.holds(0, 0, 3000) && whole.holds(0, 5999, 3000) && whole.holds(1, 3000, 0) && whole.holds(1, 3000, 5999));
        assert!(!whole.holds(0, 0, 0) && !whole.holds(1, 5999, 5999) && !whole.holds(0, 700, 700) && whole.holds(0, 900, 900));
        // The middle of the sensor at its own pixels: nothing outside.
        let middle = PictureArea::within(3968, 3968, vec![[1990.0, 1980.0, IMAGE_CIRCLE, IMAGE_CIRCLE]; 2]);
        assert_eq!((middle.outside(), middle.share_outside()), (0, 0.0));
        // Pixels that are wider than tall: the circle is an ellipse in the frame.
        let wide = PictureArea::within(400, 300, vec![[200.0, 150.0, 150.0, 75.0]]);
        assert!(wide.holds(0, 60, 150) && !wide.holds(0, 40, 150) && wide.holds(0, 200, 80) && !wide.holds(0, 200, 70));
        assert_eq!(wide.circles(), &[[200.0, 150.0, 150.0, 75.0]]);
    }

    /// What lies outside is set to one value, the channels one after the
    /// other, and nothing inside is touched.
    #[test]
    fn the_corners_are_set_to_black() {
        let area = PictureArea::within(8, 6, vec![[3.5, 2.5, 3.0, 3.0], [100.0, 100.0, 1.0, 1.0]]);
        let mut frame: Vec<u16> = (0..96).map(|v| 1000 + v as u16).collect();
        let before = frame.clone();
        area.blacken(&mut frame, 64);
        for channel in 0..2 {
            for y in 0..6 {
                for x in 0..8 {
                    let at = channel * 48 + y * 8 + x;
                    assert_eq!(frame[at], if area.holds(channel, x, y) { before[at] } else { 64 }, "channel {channel} ({x}, {y})");
                }
            }
        }
        // The first channel keeps its middle, the second lies wholly outside its circle.
        assert!(frame[..48].iter().any(|v| *v != 64) && frame[..48].iter().any(|v| *v == 64) && frame[48..].iter().all(|v| *v == 64));
        assert_eq!(area.outside(), frame.iter().filter(|v| **v == 64).count());
    }

    const SAMPLES: &str = "/Volumes/Database/x6/1115 raw video";

    #[test]
    fn a_kept_pair_list_reads_back_and_a_stale_one_does_not() {
        let dir = std::env::temp_dir().join(format!("vr180-x6raw-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let take = Take {
            stated_levels: None, white_balance_file: None,
            layout: Layout::Paired, cfa: x6b::CFA, black_level: x6b::BLACK_LEVEL, white_level: x6b::WHITE_LEVEL,
            dir: dir.clone(), fps_setting: 60, width: 3840, height: 3840, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: None, exposure_file: None,
            seal: x6b::Seal { fps_setting: 60, request_window: (10, 13), received_pairs: 3, pairs: 3, bytes: 3 * 8192, eye_frames: None, eye_gaps: None },
            capture_bytes: 3 * 8192,
        };
        let pair = |request: u32, offset: u64| PairEntry {
            request, offset, len: 8192, profile: x6b::Profile::Raised60, exposure_ns: [16_683_333 * request as u64, 16_683_333 * request as u64 - 29_000], other: None, shutter_ns: [0; 2],
        };
        let pairs = vec![pair(10, 8192), pair(11, 0), pair(13, 16384)];
        let file = dir.join("t.pairs");
        store_index(&file, &take, &pairs);
        assert_eq!(load_index(&file, &take).as_deref(), Some(&pairs[..]));
        // Another capture length, or another pair count: not this take's list.
        let mut longer = take.clone();
        longer.capture_bytes += 4096;
        assert!(load_index(&file, &longer).is_none());
        let mut more = take.clone();
        more.seal.pairs = 4;
        assert!(load_index(&file, &more).is_none());
        // A row outside the capture is refused.
        store_index(&file, &take, &[pair(10, 0), pair(11, 8192), pair(12, 3 * 8192)]);
        assert!(load_index(&file, &take).is_none());

        let index = TakeIndex::new(take, pairs);
        assert_eq!(index.missing, vec![12]);
        assert_eq!(index.frames(), 3);
        assert!((index.pts_s(2) - 3.0 * 0.016_683_333).abs() < 1e-9, "{}", index.pts_s(2));
        assert_eq!(index.timeline.len(), 3);
        assert!((index.duration_s() - (3.0 * 0.016_683_333 + 1001.0 / 60000.0)).abs() < 1e-9);
        // A time in the gap shows the frame after it; past the end, the last.
        assert_eq!(index.frame_at(0.0), 0);
        assert_eq!(index.frame_at(0.017), 1);
        assert_eq!(index.frame_at(0.035), 2);
        assert_eq!(index.frame_at(9.0), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rest_bias_needs_a_still_stretch() {
        let still = |n: usize, off: f32| (0..n).map(move |i| InsvImuSample {
            t_us: i as u64 * 1000, acc_g: [0.0; 3], gyr_dps: [0.2 + off, -0.1, 0.05 + 0.01 * ((i % 7) as f32 - 3.0)],
        });
        let moving = |n: usize| (0..n).map(|i| InsvImuSample {
            t_us: i as u64 * 1000, acc_g: [0.0; 3], gyr_dps: [30.0 * (i as f32 * 0.05).sin(), 12.0, -8.0 * (i as f32 * 0.02).cos()],
        });
        let mixed: Vec<_> = moving(1500).chain(still(1024, 0.0)).chain(moving(700)).collect();
        let b = rest_bias(&mixed);
        let d = std::f32::consts::PI / 180.0;
        assert!((b[0] - 0.2 * d).abs() < 1e-4 && (b[1] + 0.1 * d).abs() < 1e-4, "{b:?}");
        assert_eq!(rest_bias(&moving(4000).collect::<Vec<_>>()), [0.0; 3]);
    }

    #[test]
    fn the_accelerometer_is_read_at_the_gyros_times() {
        let at = |t_ms: u64, v: f32| x6b::GyroSample { t_ns: t_ms * 1_000_000, xyz: [v, -v, 2.0 * v] };
        let gyro: Vec<_> = (0..40).map(|i| at(100 + i, 0.0)).collect();
        // The same stamps: the readings themselves.
        let same: Vec<_> = (0..45).map(|i| at(100 + i, i as f32)).collect();
        let ms = |a: u64, b: u64| (a * 1_000_000, b * 1_000_000);
        let (a, covered) = accel_at_gyro(&gyro, &same, ms(100, 139));
        assert_eq!((a.len(), a[0], a[39], covered), (40, [0.0, 0.0, 0.0], [39.0, -39.0, 78.0], 0..40));
        // Stamps half a sample off: the reading in between.
        let off: Vec<_> = (0..45).map(|i| x6b::GyroSample { t_ns: (100 + i) * 1_000_000 - 500_000, xyz: [i as f32, 0.0, 0.0] }).collect();
        let (a, _) = accel_at_gyro(&gyro, &off, ms(100, 139));
        assert!((a[10][0] - 10.5).abs() < 1e-4 && (a[39][0] - 39.5).abs() < 1e-4, "{:?} {:?}", a[10], a[39]);
        // Too few samples, or a stream that ends before the frames begin: none.
        assert!(accel_at_gyro(&gyro, &same[..10], ms(100, 139)).0.is_empty());
        let short: Vec<_> = (0..16).map(|i| at(100 + i, 1.0)).collect();
        let long_gyro: Vec<_> = (0..2000).map(|i| at(100 + i, 0.0)).collect();
        assert_eq!(accel_at_gyro(&long_gyro, &short, ms(100 + 500, 2099)), (Vec::new(), 0..0));
        // One that starts a little after the gyro and ends a little before
        // it serves: the samples outside take its nearest reading, and it
        // speaks for every one of them.
        let inner: Vec<_> = (26..1980).map(|i| at(100 + i, i as f32)).collect();
        let (a, covered) = accel_at_gyro(&long_gyro, &inner, ms(100, 2099));
        assert_eq!((a.len(), a[0][0], a[26][0], a[1000][0], a[1999][0], covered), (2000, 26.0, 26.0, 1000.0, 1979.0, 0..2000));
        assert!(accel_at_gyro(&[], &same, ms(100, 139)).0.is_empty());
        // One that starts 1.65 s after the gyro, half a second into the
        // frames: it serves. It speaks for the gyro samples inside its own
        // span; those up to half a second ahead take its first reading, and
        // the ones before have none.
        let late: Vec<_> = (1650..2000).map(|i| at(100 + i, i as f32)).collect();
        let (a, covered) = accel_at_gyro(&long_gyro, &late, ms(100 + 1100, 2099));
        assert_eq!((a.len(), a[0], a[1149], a[1150][0], a[1650][0], a[1999][0], covered), (2000, [0.0; 3], [0.0; 3], 1650.0, 1650.0, 1999.0, 1650..2000));
        // A short stream with frames inside it serves them, and no further.
        let (a, covered) = accel_at_gyro(&long_gyro, &short, ms(100, 110));
        assert_eq!((a.len(), a[515][0], a[516], covered), (2000, 1.0, [0.0; 3], 0..16));
    }

    /// A take's frame timing is its sensor record's: the readout, and the
    /// centre row half of it after the stamp. Without the record, nothing.
    #[test]
    fn the_frame_timing_is_the_sensor_records() {
        let dir = std::env::temp_dir().join(format!("vr180-take-timing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut take = Take {
            stated_levels: None, white_balance_file: None,
            layout: Layout::Paired, cfa: x6b::CFA, black_level: x6b::BLACK_LEVEL, white_level: x6b::WHITE_LEVEL,
            dir: dir.clone(), fps_setting: 30, width: 3840, height: 3840, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: None, exposure_file: None,
            seal: x6b::Seal { fps_setting: 30, request_window: (1, 1), received_pairs: 1, pairs: 1, bytes: 8192, eye_frames: None, eye_gaps: None },
            capture_bytes: 8192,
        };
        assert_eq!(capture_timing(&take), CaptureTiming { readout_ms: None, stamp_to_content_ms: 0.0 });
        let eye = r#"{"bayer_output_wh":[3840,3840],"line_period_ns":3791.6666628749995,"rolling_shutter_first_to_last_row_ns":14556208}"#;
        std::fs::write(dir.join("sensor-metadata.json"), format!(r#"{{"eyes":{{"CAM0":{eye},"CAM1":{eye}}}}}"#)).unwrap();
        take.sensor_metadata = Some("sensor-metadata.json".into());
        let t = capture_timing(&take);
        assert!((t.readout_ms.unwrap() - 14.556208).abs() < 1e-5 && (t.stamp_to_content_ms - 7.278104).abs() < 1e-5, "{t:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A take that lists its frames' integration times is sampled half a
    /// shutter later than one that does not: the centre row's mid-exposure.
    /// A frame the list has no value for keeps the row's start.
    #[test]
    fn a_listed_shutter_time_moves_the_sample_to_mid_exposure() {
        let dir = std::env::temp_dir().join(format!("vr180-take-shutter-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hz = 19_200_000u64;
        // A camera turning at one radian a second about one axis, two seconds of it.
        let gyro: String = std::iter::once("counter_ticks,x,y,z\n".to_string())
            .chain((0..2000u64).map(|i| format!("{},1.0,0,0\n", i * hz / 1000))).collect();
        std::fs::write(dir.join(x6b::GYRO_FILE), gyro).unwrap();
        let eye = r#"{"bayer_output_wh":[3840,3840],"line_period_ns":4000.0,"rolling_shutter_first_to_last_row_ns":15356000}"#;
        std::fs::write(dir.join("sensor-metadata.json"), format!(r#"{{"eyes":{{"CAM0":{eye},"CAM1":{eye}}}}}"#)).unwrap();
        let take = Take {
            stated_levels: None, white_balance_file: None,
            layout: Layout::Paired, cfa: x6b::CFA, black_level: x6b::BLACK_LEVEL, white_level: x6b::WHITE_LEVEL,
            dir: dir.clone(), fps_setting: 50, width: 3840, height: 3840,
            clock_start: Some(x6b::ClockAnchor { counter_ticks: 0, counter_hz: hz, ..Default::default() }),
            audio_file: None, audio_timing: None, sensor_metadata: Some("sensor-metadata.json".into()), exposure_file: None,
            seal: x6b::Seal { fps_setting: 50, request_window: (10, 13), received_pairs: 4, pairs: 4, bytes: 4 * 8192, eye_frames: None, eye_gaps: None },
            capture_bytes: 4 * 8192,
        };
        let pair = |request: u32| PairEntry {
            request, offset: 0, len: 8192, profile: x6b::Profile::ALL[0],
            exposure_ns: [500_000_000 + 20_000_000 * (request as u64 - 10), 500_000_000 + 20_000_000 * (request as u64 - 10)],
            other: None, shutter_ns: [0; 2],
        };
        let index = TakeIndex::new(take.clone(), (10..14).map(pair).collect());
        assert!(index.shutter_ns.iter().all(|s| *s == [None, None]) && index.shutter_s(0, 0).is_none());
        // How far the camera has turned at each frame's sample, radians.
        let turned = |imu: &DjiOsvImu| -> Vec<f64> { imu.frame_quats.iter().map(|q| 2.0 * (q.w as f64).abs().min(1.0).acos()).collect() };
        let plain = turned(&build_imu(&index, 3840, 3840));
        // Half the readout after each stamp and the gyro's offset on: 0.5 s + 7.678 ms + 0.47 ms, then 20 ms a frame.
        assert!((GYRO_OFFSET_MS - crate::insv_imu::INSV_STUDIO_SAMPLE_LAG_MS - 0.47).abs() < 1e-9);
        assert!((plain[0] - 0.508_148).abs() < 1e-4 && (plain[3] - plain[0] - 0.060).abs() < 2e-4, "{plain:?}");
        // Ten milliseconds for request 10, twelve for 11 (one eye only), none for 12, nothing listed for 13.
        std::fs::write(dir.join(x6b::EXPOSURE_FILE), "request,cam0_exposure_ns,cam1_exposure_ns\n10,10000000,10000000\n11,,12000000\n12,0,\n").unwrap();
        let index = TakeIndex::new(take.clone(), (10..14).map(pair).collect());
        assert_eq!((index.shutter_s(0, 0), index.shutter_s(1, 0), index.shutter_s(1, 1), index.shutter_s(2, 0), index.shutter_s(3, 1)), (Some(0.010), Some(0.012), Some(0.012), None, None));
        let listed = turned(&build_imu(&index, 3840, 3840));
        let later: Vec<f64> = listed.iter().zip(&plain).map(|(a, b)| a - b).collect();
        assert!((later[0] - 0.005).abs() < 2e-4 && (later[1] - 0.006).abs() < 2e-4 && later[2].abs() < 2e-4 && later[3].abs() < 2e-4, "{later:?}");
        // Not every pair has a time: the frames' times stay the stamps' (20 ms apart here).
        assert!((index.pts_s(1) - 0.020).abs() < 1e-9 && index.t0_ns() == 500_000_000);

        // The sensor reads out at an even pace while a change of shutter moves
        // the START of the exposure: a 50 fps take whose shutter drops from 18
        // to 4 ms has the next stamp 14 ms late. On the stamps alone that reads
        // as a frame missing (34 ms is nearer two periods than one); with the
        // list, the frames are 20 ms apart, as they were taken.
        let stamps = [500_000_000u64, 520_000_000, 554_000_000, 574_000_000];
        let shutters = [18_000_000u64, 18_000_000, 4_000_000, 4_000_000];
        let pairs: Vec<PairEntry> = stamps.iter().enumerate()
            .map(|(i, t)| PairEntry { request: 10 + i as u32, offset: 0, len: 8192, profile: x6b::Profile::ALL[0], exposure_ns: [*t, *t], other: None, shutter_ns: [0; 2] }).collect();
        std::fs::remove_file(dir.join(x6b::EXPOSURE_FILE)).unwrap();
        let bare = TakeIndex::new(take.clone(), pairs.clone());
        assert_eq!((bare.timeline.dropped_frames(), bare.t0_ns()), (1, 500_000_000));
        let list: String = std::iter::once("request,cam0_exposure_ns,cam1_exposure_ns\n".to_string())
            .chain(shutters.iter().enumerate().map(|(i, s)| format!("{},{s},{s}\n", 10 + i))).collect();
        std::fs::write(dir.join(x6b::EXPOSURE_FILE), list).unwrap();
        let even = TakeIndex::new(take, pairs);
        assert_eq!(even.timeline.dropped_frames(), 0);
        assert!((0..4).all(|i| (even.pts_s(i) - 0.020 * i as f64).abs() < 1e-9), "{:?}", (0..4).map(|i| even.pts_s(i)).collect::<Vec<_>>());
        assert_eq!(even.t0_ns(), 518_000_000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A take of the camera build that lists its frames' integration times
    /// (skipped when the volume is not mounted): every frame has one, the
    /// frames are evenly spaced although the shutter changed 32 times, and
    /// the poses are built with them.
    #[test]
    fn a_take_with_an_exposure_list_keeps_an_even_timeline() {
        let dir = Path::new("/Volumes/Database/x6/1116 Raw Video/RAWVIDEO-20261116-205643");
        if !dir.exists() { return; }
        let index = open(dir).unwrap();
        assert_eq!((index.frames(), index.take.exposure_file.as_deref()), (3050, Some("exposure.csv")));
        assert!(index.shutter_ns.iter().all(|s| s[0].is_some() && s[0] == s[1]));
        let (lo, hi) = index.shutter_ns.iter().fold((u64::MAX, 0), |(lo, hi), s| (lo.min(s[0].unwrap()), hi.max(s[0].unwrap())));
        assert_eq!((lo, hi), (6_617_155, 16_429_989));
        assert_eq!(index.timeline.dropped_frames(), 0);
        // Even to a few microseconds over the whole take.
        let dt = index.dt_s();
        let worst = (0..index.frames()).map(|i| (index.pts_s(i) - i as f64 * dt).abs()).fold(0f64, f64::max);
        assert!(worst < 30e-6, "{worst}");
        let imu = build_imu(&index, 3840, 3840);
        assert_eq!(imu.frame_quats.len(), 3050);
    }

    /// A take of the camera build that records the sensor's row timing and
    /// the accelerometer (skipped when the volume is not mounted): its
    /// readout is the record's, its poses are built with the accelerometer,
    /// and the earth frame they are in has the accelerometer's vector up.
    #[test]
    fn a_take_with_a_sensor_record_builds_its_poses_from_it() {
        let dir = Path::new("/Volumes/Database/x6/1116 Raw Video/RAWVIDEO-20261116-162028");
        if !dir.exists() { return; }
        let index = open(dir).unwrap();
        assert_eq!(index.take.sensor_metadata.as_deref(), Some("sensor-metadata.json"));
        let timing = capture_timing(&index.take);
        assert!((timing.readout_ms.unwrap() - 14.556208).abs() < 1e-4 && (timing.stamp_to_content_ms - 7.278104).abs() < 1e-4);
        let imu = build_imu(&index, 3840, 3840);
        assert_eq!(imu.frame_quats.len(), index.frames());
        assert!((imu.readout_ms.unwrap() - 14.556208).abs() < 1e-4);
        assert!((crate::dji_imu::readout_ms_for(crate::SourceKind::X6Raw, 29.97, Some(&imu)) - 14.556208).abs() < 1e-4);
        // The accelerometer at rest, turned by a frame's pose, points up (+z of the poses' earth frame).
        let hz = index.take.clock_start.unwrap().counter_hz;
        let accel = x6b::read_accel(dir, hz);
        assert!(accel.len() > 30_000);
        let mid = index.frames() / 2;
        let t = index.pairs[mid].exposure_ns[0];
        let near = accel.iter().min_by_key(|a| a.t_ns.abs_diff(t)).unwrap();
        let q = imu.frame_quats[mid];
        let (a, n) = (near.xyz, (near.xyz[0].powi(2) + near.xyz[1].powi(2) + near.xyz[2].powi(2)).sqrt());
        let v = [a[0] / n, a[1] / n, a[2] / n];
        // q v q*: the z component of the rotated vector.
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        let up = 2.0 * (x * z - w * y) * v[0] + 2.0 * (y * z + w * x) * v[1] + (1.0 - 2.0 * (x * x + y * y)) * v[2];
        assert!(up > 0.995, "accelerometer in the earth frame: z = {up}");
    }

    #[test]
    fn a_frame_without_an_enhanced_picture_is_enlarged() {
        // 2x2 → 4x4: the corners keep their values, the middle is their mix.
        let rgb = [[0.0f32, 0.0, 0.0], [1.0, 0.5, 0.25], [0.5, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let out = enlarged(&rgb, 2, 2);
        assert_eq!(out.len(), 4 * 4 * 4);
        let px = |x: usize, y: usize| [out[(y * 4 + x) * 4], out[(y * 4 + x) * 4 + 1], out[(y * 4 + x) * 4 + 2], out[(y * 4 + x) * 4 + 3]];
        assert_eq!(px(0, 0), [0, 0, 0, 65535]);
        assert_eq!(px(3, 0), [65535, 32768, 16384, 65535]);
        assert_eq!(px(3, 3), [65535, 65535, 65535, 65535]);
        // (1, 0) sits a quarter of the way from the first source pixel to the second.
        assert_eq!(px(1, 0), [16384, 8192, 4096, 65535]);
        // Once (factor 1) it is the picture itself.
        let same = enlarged(&rgb, 2, 1);
        assert_eq!(&same[4..8], &[65535, 32768, 16384, 65535]);
    }

    #[test]
    fn the_largest_files_of_a_stretch_are_looked_at_first() {
        let dir = std::env::temp_dir().join(format!("vr180-enhanced-candidates-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for channel in 0..2 {
            std::fs::create_dir_all(dir.join(crate::sensor_dng::channel_dir(channel))).unwrap();
        }
        // Forty plain frames a channel, and one large one late in channel 1.
        for channel in 0..2 {
            for request in 100..140u32 {
                let len = if (channel, request) == (1, 131) { 900 } else { 100 + (request as usize % 7) };
                std::fs::write(dir.join(crate::sensor_dng::frame_file(channel, request)), vec![0u8; len]).unwrap();
            }
        }
        // What is not a frame file of the folder's naming is not listed.
        for stray in ["CAM0/._000000100.dng", "CAM0/notes.txt", "CAM0/364.dng", "CAM1/000000131-Enhanced.dng"] {
            std::fs::write(dir.join(stray), vec![0u8; 5000]).unwrap();
        }
        let listing = enhanced_listing(&dir);
        assert_eq!((listing[0].len(), listing[1].len()), (40, 40));
        assert_eq!((listing[0][0], listing[1][31]), ((100, 102), (131, 900)));
        // The whole take: the large file first, and the ends of the listing among the rest.
        let (count, files) = enhanced_candidates(&listing, |_| true);
        assert_eq!(count, 80);
        assert_eq!(files[0], (1, 131));
        assert!(files.len() >= ENHANCED_SEARCH && files.len() <= ENHANCED_SEARCH + 3, "{}", files.len());
        assert!(files.contains(&(0, 100)) && files.contains(&(1, 139)));
        // A stretch the large file is not in: its own files only.
        let (count, files) = enhanced_candidates(&listing, |request| (100..110).contains(&request));
        assert_eq!(count, 20);
        assert!(files.iter().all(|f| (100..110).contains(&f.1)) && !files.contains(&(1, 131)));
        // A stretch the folder holds nothing of, and a folder that is not there.
        assert_eq!(enhanced_candidates(&listing, |request| request > 500), (0, Vec::new()));
        let _ = std::fs::remove_dir_all(&dir);
        let none = enhanced_listing(&dir);
        assert!(none[0].is_empty() && none[1].is_empty());
    }

    /// The sample frame Camera Raw enhanced (skipped when the volume is not
    /// mounted or the folder no longer holds it): the take's frames come at
    /// the enhanced size, the enhanced eye from the folder and the other
    /// from the capture.
    #[test]
    fn enhanced_frames_replace_the_captures() {
        let take = Path::new(SAMPLES).join("RAWVIDEO-20261116-011723");
        let dir = Path::new(SAMPLES).join("RAWVIDEO-20261116-011723_OriginalDNG");
        let file = dir.join("CAM0/000000364.dng");
        if !take.exists() || !cfg!(target_os = "macos") || !matches!(crate::enhanced_dng::find(&file), Ok(Some(_))) { return; }
        // Without the folder, and with a folder that holds nothing: the capture's own size.
        let plain = X6RawIter::new(&take, false, false, 1, None).unwrap().with_enhanced_frames(None);
        assert_eq!((plain.eye_dims(), plain.enhanced_side()), ((3840, 3840), None));
        let whole = |dir: &Path| EnhancedSource { dir: dir.to_path_buf(), t_in: None, t_out: None };
        let empty = std::env::temp_dir().join(format!("vr180-no-enhanced-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(X6RawIter::new(&take, false, false, 1, None).unwrap().with_enhanced_frames(Some(&whole(&empty))).eye_dims(), (3840, 3840));
        let _ = std::fs::remove_dir_all(&empty);
        let mut iter = X6RawIter::new(&take, false, true, 1, None).unwrap().with_enhanced_frames(Some(&whole(&dir)));
        assert_eq!((iter.eye_dims(), iter.enhanced_side()), ((7680, 7680), Some(7680)));
        let i = iter.index().pairs.iter().position(|p| p.request == 364).expect("request 364");
        let t = iter.index().pts_s(i);
        // A stretch around the frame uses it; a stretch it is not in leaves the capture as it is.
        let stretch = |t_in: f64, t_out: f64| X6RawIter::new(&take, false, false, 1, None).unwrap()
            .with_enhanced_frames(Some(&EnhancedSource { dir: dir.clone(), t_in: Some(t_in), t_out: Some(t_out) })).eye_dims();
        let dt = 1.0 / 50.0;
        assert_eq!(stretch(t - 0.1, t + 0.1), (7680, 7680));
        let side = |t_in: f64, t_out: f64| enhanced_side(&take, &EnhancedSource { dir: dir.clone(), t_in: Some(t_in), t_out: Some(t_out) });
        assert_eq!((side(t - 0.1, t + 0.1), side(t + dt, t + 2.0)), (Some(7680), None));
        assert_eq!(enhanced_side(&take, &whole(&dir)), Some(7680));
        assert_eq!(stretch(t, t), (7680, 7680));
        assert_eq!(stretch(t + dt, t + 2.0), (3840, 3840));
        assert_eq!(stretch(0.0, t - dt), (3840, 3840));
        iter.seek(t).unwrap();
        let pair = iter.next_pair().unwrap().expect("a frame");
        assert_eq!((pair.eye_w, pair.eye_h, pair.bit_depth), (7680, 7680, 16));
        assert_eq!((pair.left.len(), pair.right.len()), (7680 * 7680 * 8, 7680 * 7680 * 8));
        // Channel 0 (left here: no swap) is the enhanced picture: the file's own values.
        let px = |eye: &[u8], x: usize, y: usize| -> [u16; 3] {
            let at = (y * 7680 + x) * 8;
            [0, 1, 2].map(|c| u16::from_le_bytes([eye[at + 2 * c], eye[at + 2 * c + 1]]))
        };
        let got = px(&pair.left, 4100, 2500);
        assert!(got.iter().zip(&[4516u16, 6556, 2314]).all(|(g, w)| (*g as i32 - *w as i32).abs() <= 6), "{got:?}");
        // Channel 1 was not enhanced: the capture's frame, enlarged — the same scene, about as bright.
        let other = px(&pair.right, 4100, 2500);
        assert!(other[1] > 1500 && other[1] < 20000, "{other:?}");
    }

    /// The frame Camera Raw enhanced also carries the settings it was left
    /// with there (skipped when the sample is not there): they are read for
    /// that frame and, the folder holding no other file with settings, for
    /// any other time of the take.
    #[test]
    fn a_takes_camera_raw_settings_are_read_from_its_frames() {
        let take = Path::new(SAMPLES).join("RAWVIDEO-20261116-011723");
        let dir = Path::new(SAMPLES).join("RAWVIDEO-20261116-011723_OriginalDNG");
        let file = dir.join("CAM0/000000364.dng");
        let Ok(Some(xmp)) = crate::enhanced_dng::read_xmp(&file) else { return };
        if !take.exists() || !xmp.contains("crs:Exposure2012") { return; }
        // The other eye's file was never opened in a raw converter: no packet.
        assert_eq!(crate::enhanced_dng::read_xmp(&dir.join("CAM1/000000364.dng")).unwrap(), None);
        let index = open(&take).unwrap();
        let t = index.pts_s(index.pairs.iter().position(|p| p.request == 364).unwrap());
        for at in [t, 0.0, t + 6.0] {
            let found = camera_raw_settings(&take, &dir, at).unwrap().expect("settings in the folder");
            assert_eq!(found.file, file);
            assert!(!found.eyes_differ);
            let s = found.develop.settings;
            // As it was left in Camera Raw: as shot, Shadows +100, the detail panel's defaults.
            assert_eq!((s.temperature, s.tint, s.exposure, s.shadows, s.highlights), (None, None, 0.0, 100.0, 0.0));
            assert_eq!((s.sharpen.amount, s.sharpen.radius, s.sharpen.detail, s.sharpen.masking, s.color_noise), (40.0, 1.0, 25.0, 0.0, 25.0));
            assert!(found.develop.not_carried.is_empty(), "{:?}", found.develop.not_carried);
        }
        // A folder with no frames: nothing.
        let empty = std::env::temp_dir().join(format!("vr180-no-settings-{}", std::process::id()));
        std::fs::create_dir_all(empty.join("CAM0")).unwrap();
        assert_eq!(camera_raw_settings(&take, &empty, t).unwrap(), None);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// The sample takes (skipped when the volume is not mounted): they open,
    /// their timelines are the stated rates, and their lens and gyro build.
    /// A take of one lens: that lens's entry of the factory string alone
    /// (channel 0 the second, as in a paired take), the frame at the
    /// sensor's own resolution, its own levels, its one sensor's timing.
    #[test]
    fn a_take_of_one_lens_is_described_by_its_lens() {
        let dir = std::env::temp_dir().join(format!("vr180-take-one-lens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lens = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193_2.455430_7227.070_7227.070_11569.100_3877.970_-0.132_0.081_89.544_-0.000064_-0.000049_-0.032576_1.31835759_-1.57224715_5.29361629_3.47188210_0.00000000_0.00016541_-0.00072159_-0.00095306_0.01344961_-0.00061933_-0.00056505_0.01970853_-0.02243911_15488_7744_193_394240";
        std::fs::write(dir.join(x6b::CALIBRATION_FILE), format!(r#"{{"eis_calib":{{"offset_v6":"{lens}"}}}}"#)).unwrap();
        let take = |channel: usize| Take {
            stated_levels: None, white_balance_file: None,
            layout: Layout::Single { channel }, cfa: x6b::CFA, black_level: 64, white_level: 1023,
            dir: dir.clone(), fps_setting: 30, width: 7680, height: 7680, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: Some("sensor-metadata.json".into()), exposure_file: None,
            seal: x6b::Seal { fps_setting: 30, request_window: (900, 900), received_pairs: 1, pairs: 1, bytes: 8192, eye_frames: None, eye_gaps: None },
            capture_bytes: 8192,
        };
        for (channel, fx, second) in [(0usize, 7227.07f32, true), (1, 7216.62, false)] {
            let take = take(channel);
            let meta = lens_meta(&take);
            assert!(meta.single_lens && !meta.back_to_back && meta.lenses.len() == 1, "channel {channel}");
            assert_eq!((meta.lenses[0].is_second_lens(), meta.dimension), (second, Some((7680, 7680))), "channel {channel}");
            // Both eyes of the calibration are that lens, at the sensor's scale.
            let imu = crate::insv_imu::insv_calib_only(&meta, 7680, 7680, X6ViewChoice::Auto);
            let (a, b) = (imu.lens_a.omni.expect("factory lens"), imu.lens_b.omni.expect("factory lens"));
            assert!((a.fx - fx).abs() < 1e-2 && a == b, "channel {channel}: {a:?}");
            assert_eq!(bayer_of(&take), BayerLayout { cfa: x6b::CFA, black: 64.0, white: 1023.0, neutral: BAYER.neutral });
        }
        // A paired take keeps both entries and its own levels.
        let paired = Take { layout: Layout::Paired, width: 3840, height: 3840, black_level: x6b::BLACK_LEVEL, white_level: x6b::WHITE_LEVEL, ..take(0) };
        assert!(lens_meta(&paired).lenses.len() == 2 && !lens_meta(&paired).single_lens);
        assert_eq!(bayer_of(&paired), BAYER);
        // The readout: the one sensor's record.
        std::fs::write(dir.join("sensor-metadata.json"),
            r#"{"eyes":{"CAM1":{"bayer_output_wh":[7680,7680],"line_period_ns":4000.000009472,"rolling_shutter_first_to_last_row_ns":30716000}}}"#).unwrap();
        assert_eq!(capture_timing(&take(0)).readout_ms, None, "another sensor's record says nothing of this one");
        let t = capture_timing(&take(1));
        assert!((t.readout_ms.unwrap() - 30.716).abs() < 1e-4 && (t.stamp_to_content_ms - 15.358).abs() < 1e-4, "{t:?}");
        // A kept list of one lens's records is not held to the seal's window
        // (sensor frame numbers), and is another layout's for a paired take.
        let entry = PairEntry { request: 41, offset: 0, len: 8192, profile: x6b::Profile::Single10, exposure_ns: [7_000_000_000; 2], other: None, shutter_ns: [0; 2] };
        let file = dir.join("t.pairs");
        store_index(&file, &take(1), &[entry]);
        assert_eq!(load_index(&file, &take(1)).as_deref(), Some(&[entry][..]));
        let as_paired = Take { layout: Layout::Paired, seal: x6b::Seal { request_window: (41, 41), ..take(1).seal }, ..take(1) };
        assert!(load_index(&file, &as_paired).is_none());
        let index = TakeIndex::new(take(1), vec![entry, PairEntry { request: 43, exposure_ns: [7_066_733_333; 2], ..entry }]);
        assert_eq!((index.frames(), index.missing.clone()), (2, vec![42]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A take is placed in the calibration's frame by its sensor record:
    /// the frames this app placed like the camera's own video keep their
    /// place, a frame of part of the view is a crop, and a reduced copy of
    /// a frame keeps its origin.
    #[test]
    fn a_take_is_placed_by_its_sensor_record() {
        use vr180_fisheye::x6b::{PairedFrame, PAIRED_FRAMES};
        let dir = std::env::temp_dir().join(format!("vr180-take-placed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let take = |layout: Layout, side: u32, record: Option<&str>| Take {
            stated_levels: None, white_balance_file: None,
            layout, cfa: x6b::CFA, black_level: 64, white_level: 1023,
            dir: dir.clone(), fps_setting: 30, width: side, height: side, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: record.map(str::to_owned), exposure_file: None,
            seal: x6b::Seal { fps_setting: 30, request_window: (1, 1), received_pairs: 1, pairs: 1, bytes: 8192, eye_frames: None, eye_gaps: None },
            capture_bytes: 8192,
        };
        let eye = |roi_y: [u32; 2], bin: u32, kept: [u32; 4], out: [u32; 2], window: Option<[u32; 4]>| format!(
            r#"{{"analog_roi_register_xyxy_inclusive":[0,{},8191,{}],"binning_xy":[{bin},{bin}],"digital_crop_after_binning_xywh":{kept:?},"bayer_output_wh":{out:?},"line_period_ns":4000.0,"rolling_shutter_first_to_last_row_ns":null{}}}"#,
            roi_y[0], roi_y[1], window.map_or(String::new(), |w| format!(r#","recorded_bayer_window_xywh":{w:?}"#)));
        let write = |eye: String| std::fs::write(dir.join("r.json"), format!(r#"{{"eyes":{{"CAM0":{eye},"CAM1":{eye}}}}}"#)).unwrap();
        let frame = |side: u16| -> PairedFrame { PAIRED_FRAMES.into_iter().find(|f| f.side == side).unwrap() };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // No record: nothing stated, the frame placed like the camera's own video.
        let bare = take(Layout::Paired, 3840, None);
        assert!(stated_frame(&bare).is_none() && lens_meta(&bare).stated_frame.is_none());
        let as_video = lens_meta(&bare).stream_geometry(3840, 3840, X6ViewChoice::Auto);
        assert_eq!((as_video.scale_x, as_video.origin_x, as_video.origin_y, as_video.cropped), (0.5, 32.0, 32.0, false));
        // 3840×3840, binned: the same place by its record.
        write(eye([256, 7935], 2, [128, 0, 3840, 3840], [3840, 3840], None));
        let placed = take(Layout::Paired, 3840, Some("r.json"));
        let g = lens_meta(&placed).stream_geometry(3840, 3840, X6ViewChoice::Auto);
        assert_eq!((g.scale_x, g.scale_y, g.origin_x, g.origin_y, g.cropped, g.readout_fraction), (as_video.scale_x, as_video.scale_y, 32.0, 32.0, false, 1.0));
        // A preview's half-size copy: half the scale, the same origin.
        let half = lens_meta(&placed).stream_geometry(1920, 1920, X6ViewChoice::Auto);
        assert_eq!((half.scale_x, half.origin_x, half.cropped), (0.25, 32.0, false));
        // One lens at 7680×7680.
        write(eye([256, 7935], 1, [256, 0, 7680, 7680], [7680, 7680], None));
        let one = take(Layout::Single { channel: 0 }, 7680, Some("r.json"));
        let g = lens_meta(&one).stream_geometry(7680, 7680, X6ViewChoice::Auto);
        assert_eq!((g.scale_x, g.origin_x, g.origin_y, g.cropped, g.mode), (1.0, 32.0, 32.0, false, vr180_fisheye::X6Mode::SingleLens));
        // 5504×5504: more than the video's window, scaled by 16/23. Not a crop.
        write(eye([128, 8063], 1, [136, 12, 7920, 7912], [5504, 5504], None));
        let g = lens_meta(&take(Layout::PairedOf(frame(5504)), 5504, Some("r.json"))).stream_geometry(5504, 5504, X6ViewChoice::Auto);
        assert!(close(g.scale_x, 16.0 / 23.0) && close(g.scale_y, 16.0 / 23.0) && close(g.origin_x, -88.0) && close(g.origin_y, -84.0) && !g.cropped, "{g:?}");
        // 4912×4912 and 3968×3968: the sensor's own pixels around its middle, part of the view.
        write(eye([1632, 6559], 1, [1640, 8, 4912, 4912], [4912, 4912], Some([0, 0, 4912, 4912])));
        let g = lens_meta(&take(Layout::PairedOf(frame(4912)), 4912, Some("r.json"))).stream_geometry(4912, 4912, X6ViewChoice::Auto);
        assert_eq!((g.scale_x, g.origin_x, g.origin_y, g.cropped), (1.0, 1416.0, 1416.0, true));
        write(eye([2112, 6079], 1, [1640, 0, 4912, 3968], [4912, 3968], Some([472, 0, 3968, 3968])));
        let small = take(Layout::PairedOf(frame(3968)), 3968, Some("r.json"));
        let g = lens_meta(&small).stream_geometry(3968, 3968, X6ViewChoice::Auto);
        assert_eq!((g.scale_x, g.origin_x, g.origin_y, g.cropped), (1.0, 1888.0, 1888.0, true));
        // Its rows: the record's line period over the rows read.
        assert!((capture_timing(&small).readout_ms.unwrap() - 4000.0 * 3967.0 * 1e-6).abs() < 1e-4);
        assert_eq!(bayer_of(&small), BayerLayout { cfa: x6b::CFA, black: 64.0, white: 1023.0, neutral: BAYER.neutral });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_motion_stream_of_two_rates_is_put_on_one() {
        // Rates that rise evenly with time: a value between two samples is known exactly.
        let sample = |t_ns: u64| x6b::GyroSample { t_ns, xyz: [t_ns as f32 * 1e-9, -2.0 * t_ns as f32 * 1e-9, 0.5] };
        let accel_of = |stream: &[x6b::GyroSample]| -> Vec<[f32; 3]> { stream.iter().map(|g| [3.0 * g.xyz[0], 1.0, g.xyz[1]]).collect() };
        // One rate, its stamps a few microseconds off the beat: handed over as it is, with its mean step.
        let even: Vec<x6b::GyroSample> = (0..200u64).map(|k| sample(k * 1_000_000 + (k % 3) * 2_000)).collect();
        let accel = accel_of(&even);
        let (rates, at_rates, dt) = evenly_sampled(&even, &accel);
        assert_eq!(rates, even.iter().map(|g| g.xyz).collect::<Vec<_>>());
        assert_eq!(at_rates, accel);
        assert_eq!(dt, ((even[199].t_ns - even[0].t_ns) as f64 * 1e-9 / 199.0) as f32);
        // A stream that starts at a fifth of its rate (40 samples 5 ms apart, then 400 a millisecond apart) and
        // falls back to it (10 more): 5 + 1 samples of its usual step for each slow one.
        let (mut t, mut two) = (0u64, Vec::new());
        for k in 0..450 {
            two.push(sample(t));
            t += if (40..439).contains(&k) { 1_000_000 } else { 5_000_000 };
        }
        let accel = accel_of(&two);
        let span_ms = (two[449].t_ns / 1_000_000) as usize;
        assert_eq!(span_ms, 40 * 5 + 399 + 10 * 5);
        let (rates, at_rates, dt) = evenly_sampled(&two, &accel);
        assert_eq!((rates.len(), at_rates.len()), (span_ms + 1, span_ms + 1));
        assert!((dt - 1e-3).abs() < 1e-9, "{dt}");
        for (k, (rate, acc)) in rates.iter().zip(&at_rates).enumerate() {
            let want = sample(k as u64 * 1_000_000).xyz;
            assert!((rate[0] - want[0]).abs() < 1e-6 && (rate[1] - want[1]).abs() < 1e-6 && rate[2] == 0.5, "sample {k}: {rate:?}");
            assert!((acc[0] - 3.0 * want[0]).abs() < 1e-5 && acc[1] == 1.0 && (acc[2] - want[1]).abs() < 1e-6, "sample {k}: {acc:?}");
        }
        // A single sample missing is a second rate as well: the hole is filled.
        let holed: Vec<x6b::GyroSample> = (0..100u64).filter(|k| *k != 50).map(|k| sample(k * 1_000_000)).collect();
        let (rates, _, dt) = evenly_sampled(&holed, &accel_of(&holed));
        assert_eq!(rates.len(), 100);
        assert!((dt - 1e-3).abs() < 1e-9 && (rates[50][0] - 0.05).abs() < 1e-6);
    }

    /// A take of one lens on disk (`VR180_ONE_LENS_TAKE=<folder>`; skipped
    /// without it): a single-eye source whose frames are one mosaic.
    #[test]
    fn a_take_of_one_lens_opens_as_one_eye() {
        let Some(dir) = std::env::var_os("VR180_ONE_LENS_TAKE").map(PathBuf::from) else { return };
        assert!(is_take(&dir) && single_eye(&dir) && single_eye(&dir.join("capture.x6b")));
        assert!(crate::source_kind::is_single_eye(crate::SourceKind::X6Raw, &dir));
        let quick = probe(&dir).unwrap();
        assert_eq!((quick.width, quick.height), (7680, 7680));
        assert!((quick.fps - 29.97).abs() < 1e-2);
        let index = open(&dir).unwrap();
        assert!(index.take.layout.single_channel().is_some() && index.frames() > 0);
        // Every frame has its integration time, and the timeline is even.
        assert!((0..index.frames()).all(|i| index.shutter_s(i, 0).is_some() && index.shutter_s(i, 1).is_some()));
        let imu = build_imu(&index, 7680, 7680);
        assert_eq!(imu.frame_quats.len(), index.frames());
        assert!(imu.lens_a.omni.is_some() && imu.lens_a.omni.map(|l| l.fx) == imu.lens_b.omni.map(|l| l.fx));
        // A frame is one mosaic: the left, with no right.
        let mut frames = X6RawIter::new(&dir, true, false, 1, None).unwrap().mosaic_pairs();
        assert_eq!(frames.eye_dims(), (7680, 7680));
        let pair = frames.next_pair().unwrap().expect("a frame");
        assert!(pair.single_eye() && pair.right.is_empty() && pair.bit_depth == MOSAIC_BIT_DEPTH);
        assert_eq!(pair.left.len(), 7680 * 7680 * 2);
        let top = bytemuck::cast_slice::<u8, u16>(&pair.left).iter().copied().max().unwrap_or(0);
        assert!(top <= index.take.white_level.max(1023), "a sample of {top}");
        // The enhanced frames of a paired take are not looked for.
        assert_eq!(enhanced_side(&dir, &EnhancedSource { dir: dir.clone(), t_in: None, t_out: None }), None);
    }

    #[test]
    fn sample_takes_open() {
        for (name, fps, frames) in [("RAWVIDEO-20261116-011723", 50.0, 1167usize), ("RAWVIDEO-20261116-021546", 60000.0 / 1001.0, 3021)] {
            let dir = Path::new(SAMPLES).join(name);
            if !dir.exists() { continue; }
            assert!(is_take(&dir) && is_take(&dir.join("capture.x6b")) && !is_take(&dir.join("audio.aac")));
            let quick = probe(&dir).unwrap();
            assert_eq!((quick.width, quick.height), (3840, 3840));
            assert!((quick.fps as f64 - fps).abs() < 1e-3);
            let index = open(&dir).unwrap();
            assert_eq!(index.frames(), frames);
            assert!(index.missing.is_empty());
            assert!((index.duration_s() - frames as f64 / fps).abs() < 0.01, "{}", index.duration_s());
            assert!((probe(&dir).unwrap().duration_sec - index.duration_s()).abs() < 1e-9);
            let imu = build_imu(&index, 3840, 3840);
            assert_eq!(imu.frame_quats.len(), frames);
            assert!(imu.imu_to_cam.is_some());
            let a = imu.lens_a.omni.expect("factory lens");
            assert!((a.xi - 2.45543).abs() < 1e-5 && (a.fx - 7200.68 * 0.5).abs() < 1e-2, "{a:?}");
            assert!((imu.lens_b.omni.unwrap().fx - 7222.06 * 0.5).abs() < 1e-2);
            let (file, start) = audio_source(&dir).expect("sound");
            assert!(file.ends_with("audio.aac"));
            assert!(start > 0.3 && start < 0.7, "{start}");
        }
    }

    /// A take of `frame` with a record for each eye, as far as its
    /// description goes: `pairs` pairs in the window 500..=503.
    fn eye_take(dir: &Path, frame: x6b::EyeFrame, pairs: u64, capture_bytes: u64) -> Take {
        Take {
            stated_levels: None, white_balance_file: None,
            layout: Layout::EyeRecords(frame), cfa: x6b::CFA, black_level: 64, white_level: 1023,
            dir: dir.to_path_buf(), fps_setting: 60, width: frame.width as u32, height: frame.height as u32, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: None, exposure_file: None,
            seal: x6b::Seal { fps_setting: 60, request_window: (500, 503), received_pairs: pairs, pairs, bytes: capture_bytes, eye_frames: Some([pairs, pairs + 2]), eye_gaps: Some([1, 0]) },
            capture_bytes,
        }
    }

    /// A pair of two records: both places are kept, with what each record
    /// says of its integration time, and a list of one kind is not read as
    /// the other.
    #[test]
    fn a_kept_list_of_pairs_of_two_records_reads_back() {
        let dir = std::env::temp_dir().join(format!("vr180-x6raw-eyeindex-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let take = eye_take(&dir, x6b::EYE_FRAMES[2], 3, 8 * 8192);
        let step = 16_683_333u64;
        let pair = |request: u32, first: u64, second: u64| PairEntry {
            request, offset: first * 8192, len: 8192, profile: x6b::Profile::EyeWideNear,
            exposure_ns: [step * request as u64, step * request as u64 + 29_000],
            other: Some((second * 8192, 8192)), shutter_ns: [4_000_000, 4_000_100],
        };
        let pairs = vec![pair(500, 1, 2), pair(501, 4, 3), pair(503, 7, 6)];
        let file = dir.join("t.pairs");
        store_index(&file, &take, &pairs);
        assert_eq!(load_index(&file, &take).as_deref(), Some(&pairs[..]));
        // Not a list of pairs of one record, nor such a list one of these.
        let paired = Take { layout: Layout::PairedOf(x6b::PAIRED_FRAMES[2]), width: 3968, height: 3968, ..take.clone() };
        assert!(load_index(&file, &paired).is_none());
        store_index(&file, &paired, &pairs.iter().map(|p| PairEntry { profile: x6b::Profile::Frame10, other: None, shutter_ns: [0; 2], ..*p }).collect::<Vec<_>>());
        assert!(load_index(&file, &take).is_none() && load_index(&file, &paired).is_some());
        // A second record outside the capture, or a profile of another frame: refused.
        store_index(&file, &take, &[pair(500, 1, 2), pair(501, 4, 3), pair(503, 7, 8)]);
        assert!(load_index(&file, &take).is_none());
        store_index(&file, &take, &pairs.iter().map(|p| PairEntry { profile: x6b::Profile::EyeNative, ..*p }).collect::<Vec<_>>());
        assert!(load_index(&file, &take).is_none());

        // The index: a pair's integration times are its records' own, its
        // time the middle of the two stamps and of the two integrations
        // after them, and the frame number no pair has is missing.
        let index = TakeIndex::new(take, pairs);
        assert_eq!((index.frames(), &index.missing[..]), (3, &[502u32][..]));
        assert_eq!(index.shutter_ns[1], [Some(4_000_000), Some(4_000_100)]);
        assert!((index.shutter_s(0, 1).unwrap() - 0.004_000_1).abs() < 1e-12);
        assert_eq!(index.t0_ns(), step * 500 + 14_500 + 4_000_050);
        assert!((index.pts_s(2) - 3.0 * step as f64 * 1e-9).abs() < 1e-9, "{}", index.pts_s(2));
        assert_eq!(index.timeline.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The white a take is developed for: the one it records, the two
    /// sensors' met halfway; each sensor's own in that sensor's files; the
    /// reference where the take records none.
    #[test]
    fn a_take_is_developed_for_the_white_balance_it_records() {
        let dir = std::env::temp_dir().join(format!("vr180-x6raw-white-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let take = eye_take(&dir, x6b::EYE_FRAMES[0], 3, 8 * 8192);
        // No record: the reference, for the development and for both sensors' files.
        assert_eq!((camera_of(&take), eye_cameras(&take), white_balance(&take)), (RawCamera::X6, [RawCamera::X6; 2], [None; 2]));
        let eye = |status: &str, r: f64, b: f64| format!(r#"{{"status":"{status}","rgb_gains":[{r},1.0,{b}],"as_shot_neutral":[{},1.0,{}]}}"#, 1.0 / r, 1.0 / b);
        let write = |cam0: String, cam1: String| std::fs::write(dir.join(x6b::WHITE_BALANCE_FILE),
            format!(r#"{{"schema":"sample-white-balance-v1","eyes":{{"CAM0":{cam0},"CAM1":{cam1}}}}}"#)).unwrap();
        let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12);
        // Both sensors: each its own in its files, the development halfway between them.
        write(eye("ok", 1.6, 2.5), eye("ok", 2.5, 1.6));
        let files = eye_cameras(&take);
        assert!(close(files[0].as_shot_neutral, [0.625, 1.0, 0.4]) && close(files[1].as_shot_neutral, [0.4, 1.0, 0.625]), "{files:?}");
        assert!(close(camera_of(&take).as_shot_neutral, [0.5, 1.0, 0.5]), "{:?}", camera_of(&take));
        assert_eq!((camera_of(&take).color_matrix, camera_of(&take).baseline_exposure), (RawCamera::X6.color_matrix, RawCamera::X6.baseline_exposure));
        // One sensor only: its files carry it, the other's the reference; the pair is developed for the one there is.
        write(eye("ok", 1.6, 2.5), eye("unavailable", 2.5, 1.6));
        assert_eq!(eye_cameras(&take)[1], RawCamera::X6);
        assert!(close(camera_of(&take).as_shot_neutral, [0.625, 1.0, 0.4]) && close(eye_cameras(&take)[0].as_shot_neutral, [0.625, 1.0, 0.4]));
        // A take of one lens: that sensor's.
        write(eye("ok", 1.6, 2.5), eye("ok", 2.5, 1.6));
        let one = Take { layout: Layout::Single { channel: 1 }, width: 7680, height: 7680, ..take.clone() };
        assert!(close(camera_of(&one).as_shot_neutral, [0.4, 1.0, 0.625]) && white_balance(&one)[0] == white_balance(&one)[1]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
