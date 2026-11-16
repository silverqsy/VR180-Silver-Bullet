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
//! - [`build_imu`] — the lens calibration (the factory string, placed like
//!   the camera's own 8K video frame) and the gyro as per-frame poses.
//! - [`X6RawIter`] — the frames. With a GPU device a pair is decoded,
//!   uploaded as two mosaics and demosaiced there into camera-linear 16-bit
//!   eyes; without one it is demosaiced on the CPU.
//! - [`audio_source`] — the sound file and where the picture starts in it.
//!
//! The two capture channels follow the camera's video streams: channel 0 is
//! the lens the stock video writes as stream 0 (calibration entry 2, the
//! RIGHT eye on a VR180-modded body), channel 1 the other.
//!
//! The frames are the sensor's linear values (black level subtracted, white
//! level at 1.0, no white balance): they need the raw development stage of
//! the colour stack before anything display-referred.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::insta360::{Insta360Meta, InsvImuSample, X6ViewChoice, X6_DEFAULT_WINDOW};
use vr180_fisheye::x6b::{self, Channels, PairEntry, Take, EYE_SAMPLES, EYE_SIDE};
use vr180_fisheye::DjiOsvImu;

use crate::decode::VideoProbe;
use crate::fisheye_decode::{FisheyePair, FisheyePairIter, GpuPair};
use crate::frame_timeline::FrameTimeline;
use crate::gpu::Device;
use crate::raw_gpu::BayerLayout;
use crate::{Error, Result};

fn take_err(e: impl std::fmt::Display) -> Error {
    Error::Ffmpeg(e.to_string())
}

/// `FisheyePair::bit_depth` of a pair whose two buffers are MOSAICS (one
/// little-endian `u16` per site, `eye_w × eye_h` sites) and not pictures:
/// what [`X6RawIter::mosaic_pairs`] hands a consumer that demosaics on its
/// own device ([`demosaic_pair`]).
pub const MOSAIC_BIT_DEPTH: u8 = 14;

/// Demosaic a mosaic pair (see [`MOSAIC_BIT_DEPTH`]) on `dev` at full size:
/// the two camera-linear eyes, left and right.
pub fn demosaic_pair(dev: &Device, pair: &FisheyePair) -> Result<(wgpu::Texture, wgpu::Texture)> {
    let (w, h) = (pair.eye_w, pair.eye_h);
    let want = w as usize * h as usize * 2;
    if pair.bit_depth != MOSAIC_BIT_DEPTH || pair.left.len() != want || pair.right.len() != want {
        return Err(take_err("not a mosaic pair"));
    }
    let eye = |bytes: &[u8]| -> Result<wgpu::Texture> {
        let mosaic = dev.new_bayer_texture(w, h);
        dev.write_bayer_bytes(&mosaic, bytes);
        let out = dev.new_linear_eye_texture(w, h);
        dev.demosaic_bayer(&mosaic, &BAYER, &out)?;
        Ok(out)
    };
    Ok((eye(&pair.left)?, eye(&pair.right)?))
}

/// The mosaic of a take's frames.
pub const BAYER: BayerLayout = BayerLayout {
    cfa: x6b::CFA,
    black: x6b::BLACK_LEVEL as f32,
    white: x6b::WHITE_LEVEL as f32,
    // `RawCamera::X6`'s as-shot neutral (a test in `bayer` holds them equal).
    neutral: [0.565_429_7, 1.0, 0.401_367_2],
};

/// A take, opened: its description and its pairs in request order.
#[derive(Debug)]
pub struct TakeIndex {
    pub take: Take,
    pub pairs: Vec<PairEntry>,
    /// Requests of the take's window that were never saved.
    pub missing: Vec<u32>,
    /// Time of every pair in seconds from the first one.
    pub timeline: FrameTimeline,
    pts_s: Vec<f64>,
}

impl TakeIndex {
    fn new(take: Take, pairs: Vec<PairEntry>) -> Self {
        let t0 = pairs.first().map_or(0, |p| p.time_ns());
        let pts_s: Vec<f64> = pairs.iter().map(|p| (p.time_ns() - t0) as f64 * 1e-9).collect();
        let dt = 1.0 / take.nominal_fps();
        let missing = x6b::missing_requests(&take, &pairs);
        Self { timeline: FrameTimeline::from_pts(pts_s.clone(), dt), take, pairs, missing, pts_s }
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
        self.pairs.first().map_or(0, |p| p.time_ns())
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

const INDEX_MAGIC: &[u8; 8] = b"X6BIDX01";

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
    let rest = bytes.strip_prefix(INDEX_MAGIC)?;
    if rest.len() < 16 {
        return None;
    }
    let (head, rows) = rest.split_at(16);
    let capture = u64::from_le_bytes(head[..8].try_into().ok()?);
    let count = u64::from_le_bytes(head[8..].try_into().ok()?);
    const ROW: usize = 37;
    if capture != take.capture_bytes || count != take.seal.pairs || rows.len() != count as usize * ROW {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    for r in rows.chunks_exact(ROW) {
        let u64_at = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        out.push(PairEntry {
            request: u32::from_le_bytes(r[..4].try_into().unwrap()),
            profile: *x6b::Profile::ALL.get(r[4] as usize)?,
            offset: u64_at(5),
            len: u64_at(13),
            exposure_ns: [u64_at(21), u64_at(29)],
        });
    }
    // The same checks the walk makes on what the rows claim.
    let (lo, hi) = take.seal.request_window;
    let ok = out.windows(2).all(|w| w[0].request < w[1].request)
        && out.iter().all(|p| p.request >= lo && p.request <= hi && p.profile.fps() == take.fps_setting && p.offset + p.len <= take.capture_bytes);
    ok.then_some(out)
}

fn store_index(file: &Path, take: &Take, pairs: &[PairEntry]) {
    let mut bytes = INDEX_MAGIC.to_vec();
    bytes.extend_from_slice(&take.capture_bytes.to_le_bytes());
    bytes.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    for p in pairs {
        bytes.extend_from_slice(&p.request.to_le_bytes());
        bytes.push(x6b::Profile::ALL.iter().position(|q| *q == p.profile).unwrap_or(0) as u8);
        bytes.extend_from_slice(&p.offset.to_le_bytes());
        bytes.extend_from_slice(&p.len.to_le_bytes());
        bytes.extend_from_slice(&p.exposure_ns[0].to_le_bytes());
        bytes.extend_from_slice(&p.exposure_ns[1].to_le_bytes());
    }
    if let Some(parent) = file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(file, bytes) {
        tracing::warn!("raw take: could not keep the pair list at {}: {e}", file.display());
    }
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
            return Ok(Arc::new(TakeIndex::new(take.clone(), pairs)));
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
                let index = TakeIndex::new(take.clone(), pairs);
                tracing::info!(
                    "raw take {}: {} pairs at {} fps listed in {:.1?} ({} missing, {:.0} MB/s both eyes)",
                    take.dir.display(), index.frames(), take.fps_setting, t0.elapsed(), index.missing.len(),
                    index.bytes_per_second() / 1e6);
                if let Some(f) = on_disk {
                    store_index(&f, &take, &index.pairs);
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
    let rate = (take.capture_bytes as f64 * 8.0 / duration_sec.max(1e-6) / 2.0) as u64;
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

/// What the camera's metadata would say about the take's frames, as far as
/// the take states it: the factory lens string, the frame placed like the
/// camera's own 8K video frame (the 7680² window of each sensor half,
/// binned 2×2), channel 0 = calibration entry 2. No motion data.
pub fn lens_meta(take: &Take) -> Insta360Meta {
    let calib_string = x6b::read_calibration_string(&take.dir).unwrap_or_default();
    let lenses = vr180_fisheye::insta360::parse_calib_string(&calib_string).unwrap_or_default();
    if lenses.len() < 2 {
        tracing::warn!("raw take {}: no factory lens string, the generic X6 profile is used", take.dir.display());
    }
    Insta360Meta {
        model: "Insta360 X6".into(),
        calib_string,
        lenses,
        window_crop: Some(X6_DEFAULT_WINDOW),
        dimension: Some((take.width, take.height)),
        frame_rate: Some(take.fps_setting),
        track_order_reversed: Some(true),
        ..Default::default()
    }
}

/// Timing of the capture that the take itself does not state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureTiming {
    /// Sensor readout (first row to last), ms. `None` = not known: no
    /// per-row rolling-shutter correction.
    pub readout_ms: Option<f32>,
    /// From a frame's exposure stamp to the middle of its centre row's
    /// exposure, ms.
    pub stamp_to_content_ms: f64,
}

/// The timing of a take's frames. The take states neither its readout time
/// nor its shutter time, so nothing here is known yet: the pose of a frame
/// is sampled at its exposure stamp and rows are not corrected one by one.
pub fn capture_timing(_take: &Take) -> CaptureTiming {
    CaptureTiming { readout_ms: None, stamp_to_content_ms: 0.0 }
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
    let samples: Vec<InsvImuSample> = gyro.iter()
        .map(|g| InsvImuSample { t_us: g.t_ns / 1000, acc_g: [0.0; 3], gyr_dps: [g.xyz[0] * deg, g.xyz[1] * deg, g.xyz[2] * deg] })
        .collect();
    let bias = rest_bias(&samples);
    // No accelerometer in the take: the poses are the integrated gyro alone
    // (no levelling, no slow trim).
    let quats = crate::insv_imu::integrate_orientation_samples(&samples, bias, 0.0);
    let line = crate::insv_imu::QuatTimeline { t_s: gyro.iter().map(|g| g.t_ns as f64 * 1e-9).collect(), quats: &quats };
    let frame_dur = index.dt_s();
    let lead = timing.stamp_to_content_ms * 1e-3;
    let fill = |channel: usize, to: &mut DjiOsvImu| {
        to.frame_quats.clear();
        to.high_rate_quats.clear();
        to.gravity.clear();
        for p in &index.pairs {
            let tc = p.exposure_ns[channel] as f64 * 1e-9 + lead;
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
        "raw take {}: {} gyro samples @ {:.1} Hz, bias {:?} rad/s, covers {:+.3} s before the first frame to {:+.3} s after the last; {} frame poses",
        take.dir.display(), gyro.len(), (gyro.len() - 1) as f64 / (last - first).max(1e-9), bias, f0 - first, last - f1, out.frame_quats.len());
    if first > f0 || last < f1 {
        tracing::warn!("raw take {}: the gyro does not cover every frame; the uncovered ones hold the nearest pose", take.dir.display());
    }
    out
}

// ── Frames ─────────────────────────────────────────────────────────────

/// Demosaiced eyes handed back by frames that are done with them.
#[derive(Default)]
struct EyePool {
    free: Mutex<Vec<(wgpu::Texture, wgpu::Texture)>>,
}

/// Returns a frame's two eye textures to the pool with its last holder.
struct EyeLoan {
    pool: Arc<EyePool>,
    eyes: Option<(wgpu::Texture, wgpu::Texture)>,
}

impl Drop for EyeLoan {
    fn drop(&mut self) {
        if let (Some(eyes), Ok(mut free)) = (self.eyes.take(), self.pool.free.lock()) {
            if free.len() < 12 {
                free.push(eyes);
            }
        }
    }
}

/// A take's frames as a raw converter enhanced them: the folder its
/// camera-original DNGs were written into ([`crate::sensor_dng`]) and the
/// side of the pictures the converter left in them
/// ([`crate::enhanced_dng`]).
struct EnhancedFrames {
    dir: PathBuf,
    side: usize,
    /// Eye pictures taken from the folder, and from the capture instead.
    used: usize,
    missing: usize,
}

/// How many files are looked at for the first enhanced picture.
const ENHANCED_SEARCH: usize = 600;

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
    samples: Vec<u16>,
    mosaics: Option<(wgpu::Texture, wgpu::Texture)>,
    pool: Arc<EyePool>,
    /// Stop after this many frames from the start of the take (0 = all).
    frame_limit: usize,
    /// Without a device: hand out the mosaics themselves, not pictures.
    mosaics_out: bool,
    /// Frames a raw converter enhanced, used in place of the capture's.
    enhanced: Option<EnhancedFrames>,
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
        Ok(Self {
            index, file, next: 0, step: step.max(1) as usize, swap_eyes, half, gpu,
            record: Vec::new(), samples: vec![0u16; 2 * EYE_SAMPLES], mosaics: None,
            pool: Arc::default(), frame_limit: 0, mosaics_out: false, enhanced: None,
        })
    }

    /// Take the frames from `dir` where a raw converter has enhanced them:
    /// the folder the take's camera-original DNGs were written into
    /// ([`crate::sensor_dng`]), whose files then hold the enhanced picture
    /// ([`crate::enhanced_dng`]). The eyes become that picture's size
    /// (twice the mosaic's for Super Resolution); a frame the folder has no
    /// enhanced picture of comes from the capture, enlarged to match.
    /// Nothing changes when the folder holds no enhanced picture.
    pub fn with_enhanced_frames(mut self, dir: Option<&Path>) -> Self {
        let Some(dir) = dir else { return self };
        // The first, the middle and the last pair, then every pair in order.
        let n = self.index.pairs.len();
        let order = [0, n / 2, n.saturating_sub(1)].into_iter().chain(0..n);
        let mut looked = 0usize;
        let mut found = None;
        'search: for i in order {
            for channel in 0..2 {
                if looked >= ENHANCED_SEARCH { break 'search; }
                looked += 1;
                let path = dir.join(crate::sensor_dng::frame_file(channel, self.index.pairs[i].request));
                if let Ok(Some(p)) = crate::enhanced_dng::find(&path) {
                    found = Some(p);
                    break 'search;
                }
            }
        }
        match found {
            Some(p) if p.width == p.height && p.width as usize % EYE_SIDE == 0 => {
                tracing::info!("raw take {}: enhanced frames of {}x{} in {}", self.index.take.dir.display(), p.width, p.height, dir.display());
                self.enhanced = Some(EnhancedFrames { dir: dir.to_path_buf(), side: p.width as usize, used: 0, missing: 0 });
                self.half = false;
            }
            Some(p) => tracing::warn!("raw take: the enhanced frames in {} are {}x{}, not a whole multiple of the take's: not used", dir.display(), p.width, p.height),
            None => tracing::warn!("raw take: no enhanced frame in {} ({looked} files looked at): the capture's frames are used", dir.display()),
        }
        self
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

    fn eye_side(&self) -> u32 {
        match &self.enhanced {
            Some(e) => e.side as u32,
            None if self.half => EYE_SIDE as u32 / 2,
            None => EYE_SIDE as u32,
        }
    }

    /// Frame `i`'s two channels from the enhanced frames' folder, as 16-bit
    /// camera values (RGBA): channel 0, then channel 1. A channel the folder
    /// has no enhanced picture of is the capture's frame, demosaiced and
    /// enlarged to the same size.
    fn enhanced_channels(&mut self, i: usize) -> Result<[Vec<u16>; 2]> {
        let (dir, side) = match &self.enhanced {
            Some(e) => (e.dir.clone(), e.side),
            None => return Err(take_err("no enhanced frames")),
        };
        let request = self.index.pairs.get(i).ok_or_else(|| take_err(format!("frame {i} is past the take's end")))?.request;
        // The two channels' pictures decode side by side.
        let channel = |channel: usize| -> Result<Vec<u16>> {
            let path = dir.join(crate::sensor_dng::frame_file(channel, request));
            // A frame the folder does not hold is not worth a line of its own.
            if !path.is_file() {
                return Ok(Vec::new());
            }
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
        let (first, second) = std::thread::scope(|scope| {
            let other = scope.spawn(|| channel(1));
            (channel(0), other.join().unwrap_or_else(|_| Err(take_err("the enhanced picture's decoder stopped"))))
        });
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

    /// The mosaic of the frame decoded last: channel 0 then channel 1.
    pub fn last_mosaic(&self) -> &[u16] {
        &self.samples
    }

    /// Decode frame `i`'s two mosaics into the iterator's buffer.
    pub fn decode_frame(&mut self, i: usize) -> Result<&[u16]> {
        let entry = *self.index.pairs.get(i).ok_or_else(|| take_err(format!("frame {i} is past the take's end")))?;
        x6b::decode_pair(&mut self.file, &entry, self.index.take.fps_setting, Channels::Both, &mut self.record, &mut self.samples, 0)
            .map_err(take_err)?;
        Ok(&self.samples)
    }

    fn gpu_pair(&mut self, dev: &Arc<Device>) -> Result<GpuPair> {
        let side = EYE_SIDE as u32;
        let out_side = self.eye_side();
        let (m0, m1) = self.mosaics.get_or_insert_with(|| (dev.new_bayer_texture(side, side), dev.new_bayer_texture(side, side)));
        dev.write_bayer_texture(m0, &self.samples[..EYE_SAMPLES]);
        dev.write_bayer_texture(m1, &self.samples[EYE_SAMPLES..]);
        let pooled = self.pool.free.lock().ok().and_then(|mut f| f.pop()).filter(|(a, _)| a.width() == out_side);
        let (e0, e1) = pooled.unwrap_or_else(|| (dev.new_linear_eye_texture(out_side, out_side), dev.new_linear_eye_texture(out_side, out_side)));
        dev.demosaic_bayer(m0, &BAYER, &e0)?;
        dev.demosaic_bayer(m1, &BAYER, &e1)?;
        let (left, right) = if self.swap_eyes { (e1.clone(), e0.clone()) } else { (e0.clone(), e1.clone()) };
        Ok(GpuPair { left, right, keep: Box::new(EyeLoan { pool: self.pool.clone(), eyes: Some((e0, e1)) }) })
    }

    fn cpu_pair(&self) -> (Vec<u8>, Vec<u8>) {
        // The GPU demosaic's twin, one channel at a time.
        let eye = |channel: usize| -> Vec<u8> {
            let rgb = crate::bayer::demosaic(&self.samples[channel * EYE_SAMPLES..(channel + 1) * EYE_SAMPLES], EYE_SIDE, EYE_SIDE, &BAYER, self.half);
            let mut bytes = Vec::with_capacity(rgb.len() * 8);
            for px in &rgb {
                for v in [px[0], px[1], px[2], 1.0] {
                    bytes.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes());
                }
            }
            bytes
        };
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
    for y in 0..out_side {
        let sy = ((y as f32 + 0.5) / f - 0.5).clamp(0.0, side as f32 - 1.0);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(side - 1);
        for x in 0..out_side {
            let sx = ((x as f32 + 0.5) / f - 0.5).clamp(0.0, side as f32 - 1.0);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(side - 1);
            let (a, b, c, d) = (rgb[y0 * side + x0], rgb[y0 * side + x1], rgb[y1 * side + x0], rgb[y1 * side + x1]);
            for k in 0..3 {
                let v = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
                out[(y * out_side + x) * 4 + k] = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
    }
    out
}

impl FisheyePairIter for X6RawIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        let i = self.next;
        if i >= self.index.frames() || (self.frame_limit > 0 && i >= self.frame_limit) {
            return Ok(None);
        }
        let side = self.eye_side();
        let pts_s = self.index.pts_s(i);
        if self.enhanced.is_some() && !self.mosaics_out {
            let [c0, c1] = self.enhanced_channels(i)?;
            self.next = i + self.step;
            let (left, right) = if self.swap_eyes { (c1, c0) } else { (c0, c1) };
            if let Some(dev) = self.gpu.clone() {
                let upload = |px: &[u16]| {
                    let tex = dev.new_camera_eye_texture(side, side);
                    dev.write_camera_eye_texture(&tex, px);
                    tex
                };
                let pair = GpuPair { left: upload(&left), right: upload(&right), keep: Box::new(()) };
                return Ok(Some(FisheyePair {
                    left: Vec::new(), right: Vec::new(), eye_w: side, eye_h: side, bit_depth: 16, pts_s,
                    gpu: Some(Arc::new(pair)),
                }));
            }
            let bytes = |px: &[u16]| bytemuck::cast_slice::<u16, u8>(px).to_vec();
            return Ok(Some(FisheyePair { left: bytes(&left), right: bytes(&right), eye_w: side, eye_h: side, bit_depth: 16, pts_s, gpu: None }));
        }
        self.decode_frame(i)?;
        self.next = i + self.step;
        if let Some(dev) = self.gpu.clone() {
            let pair = self.gpu_pair(&dev)?;
            return Ok(Some(FisheyePair {
                left: Vec::new(), right: Vec::new(), eye_w: side, eye_h: side, bit_depth: 16, pts_s,
                gpu: Some(Arc::new(pair)),
            }));
        }
        if self.mosaics_out {
            let bytes = |ch: usize| bytemuck::cast_slice::<u16, u8>(&self.samples[ch * EYE_SAMPLES..(ch + 1) * EYE_SAMPLES]).to_vec();
            let (left, right) = if self.swap_eyes { (bytes(1), bytes(0)) } else { (bytes(0), bytes(1)) };
            return Ok(Some(FisheyePair { left, right, eye_w: side, eye_h: side, bit_depth: MOSAIC_BIT_DEPTH, pts_s, gpu: None }));
        }
        let (left, right) = self.cpu_pair();
        Ok(Some(FisheyePair { left, right, eye_w: side, eye_h: side, bit_depth: 16, pts_s, gpu: None }))
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        self.next = self.index.frame_at(target_s.max(0.0));
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        (self.eye_side(), self.eye_side())
    }

    fn source_decimation(&self) -> u32 {
        self.step as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: &str = "/Volumes/Database/x6/1115 raw video";

    #[test]
    fn a_kept_pair_list_reads_back_and_a_stale_one_does_not() {
        let dir = std::env::temp_dir().join(format!("vr180-x6raw-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let take = Take {
            dir: dir.clone(), fps_setting: 60, width: 3840, height: 3840, clock_start: None,
            audio_file: None, audio_timing: None,
            seal: x6b::Seal { fps_setting: 60, request_window: (10, 13), received_pairs: 3, pairs: 3, bytes: 3 * 8192 },
            capture_bytes: 3 * 8192,
        };
        let pair = |request: u32, offset: u64| PairEntry {
            request, offset, len: 8192, profile: x6b::Profile::Raised60, exposure_ns: [16_683_333 * request as u64, 16_683_333 * request as u64 - 29_000],
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
        assert_eq!(plain.eye_dims(), (3840, 3840));
        let empty = std::env::temp_dir().join(format!("vr180-no-enhanced-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(X6RawIter::new(&take, false, false, 1, None).unwrap().with_enhanced_frames(Some(&empty)).eye_dims(), (3840, 3840));
        let _ = std::fs::remove_dir_all(&empty);
        let mut iter = X6RawIter::new(&take, false, true, 1, None).unwrap().with_enhanced_frames(Some(&dir));
        assert_eq!(iter.eye_dims(), (7680, 7680));
        let i = iter.index().pairs.iter().position(|p| p.request == 364).expect("request 364");
        let t = iter.index().pts_s(i);
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

    /// The sample takes (skipped when the volume is not mounted): they open,
    /// their timelines are the stated rates, and their lens and gyro build.
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
}
