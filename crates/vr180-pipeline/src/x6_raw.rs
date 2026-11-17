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

use std::collections::{HashMap, HashSet};
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
        let shutter_ns: Vec<[Option<u64>; 2]> = pairs.iter().map(|p| listed.get(&p.request).copied().unwrap_or([None; 2])).collect();
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
/// file states.
pub fn capture_timing(take: &Take) -> CaptureTiming {
    let readout_ms = x6b::read_sensor_readout(take).map(|eyes| (eyes[0].first_to_last_row_ns * 1e-6) as f32);
    CaptureTiming { readout_ms, stamp_to_content_ms: readout_ms.map_or(0.0, |r| r as f64 * 0.5) }
}

/// The accelerometer (m/s²) at each gyro sample's time: the two streams
/// share a clock, and nearly every stamp. Empty when the take has too
/// little of it to be of use.
fn accel_at_gyro(gyro: &[x6b::GyroSample], accel: &[x6b::GyroSample]) -> Vec<[f32; 3]> {
    if accel.len() < 16 || gyro.is_empty() {
        return Vec::new();
    }
    // It has to cover the gyro's span, give or take a few samples.
    let slack = 20_000_000;
    if accel[0].t_ns > gyro[0].t_ns + slack || accel[accel.len() - 1].t_ns + slack < gyro[gyro.len() - 1].t_ns {
        return Vec::new();
    }
    let mut j = 0usize;
    gyro.iter().map(|g| {
        while j + 2 < accel.len() && accel[j + 1].t_ns <= g.t_ns {
            j += 1;
        }
        let (a, b) = (accel[j], accel[j + 1]);
        let span = b.t_ns.saturating_sub(a.t_ns).max(1) as f64;
        let f = ((g.t_ns as f64 - a.t_ns as f64) / span).clamp(0.0, 1.0) as f32;
        [0, 1, 2].map(|k| a.xyz[k] + (b.xyz[k] - a.xyz[k]) * f)
    }).collect()
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
    let accel = accel_at_gyro(&gyro, &x6b::read_accel(&take.dir, hz));
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
        let rates: Vec<[f32; 3]> = gyro.iter().map(|g| g.xyz).collect();
        let dt = ((gyro[gyro.len() - 1].t_ns - gyro[0].t_ns) as f64 * 1e-9 / (gyro.len() - 1) as f64) as f32;
        (vr180_core::gyro::vqf::run(&rates, &accel, None, dt).bias_rad_s, crate::insv_imu::INSV_TAU_ACC_S)
    };
    let quats = crate::insv_imu::integrate_orientation_samples(&samples, bias, trim_tau_s);
    let line = crate::insv_imu::QuatTimeline { t_s: gyro.iter().map(|g| g.t_ns as f64 * 1e-9).collect(), quats: &quats };
    let frame_dur = index.dt_s();
    let lead = timing.stamp_to_content_ms * 1e-3;
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
        if accel.is_empty() { "no accelerometer".to_string() } else { format!("accelerometer (trim {trim_tau_s} s)") },
        f0 - first, last - f1, out.frame_quats.len(),
        match timing.readout_ms {
            Some(r) => format!("readout {r:.3} ms, centre row {:.3} ms after the stamp", timing.stamp_to_content_ms),
            None => "no readout record".to_string(),
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

    /// Take the frames from `source`'s folder where a raw converter has
    /// enhanced them. The eyes become the enhanced pictures' size (twice the
    /// mosaic's for Super Resolution); a frame the folder has no enhanced
    /// picture of comes from the capture, enlarged to match. Nothing changes
    /// when no frame of the export's stretch has an enhanced picture: the
    /// capture's frames are then used as they are.
    pub fn with_enhanced_frames(mut self, source: Option<&EnhancedSource>) -> Self {
        let Some(source) = source else { return self };
        let dir = &source.dir;
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
            audio_file: None, audio_timing: None, sensor_metadata: None, exposure_file: None,
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
    fn the_accelerometer_is_read_at_the_gyros_times() {
        let at = |t_ms: u64, v: f32| x6b::GyroSample { t_ns: t_ms * 1_000_000, xyz: [v, -v, 2.0 * v] };
        let gyro: Vec<_> = (0..40).map(|i| at(100 + i, 0.0)).collect();
        // The same stamps: the readings themselves.
        let same: Vec<_> = (0..45).map(|i| at(100 + i, i as f32)).collect();
        let a = accel_at_gyro(&gyro, &same);
        assert_eq!((a.len(), a[0], a[39]), (40, [0.0, 0.0, 0.0], [39.0, -39.0, 78.0]));
        // Stamps half a sample off: the reading in between.
        let off: Vec<_> = (0..45).map(|i| x6b::GyroSample { t_ns: (100 + i) * 1_000_000 - 500_000, xyz: [i as f32, 0.0, 0.0] }).collect();
        let a = accel_at_gyro(&gyro, &off);
        assert!((a[10][0] - 10.5).abs() < 1e-4 && (a[39][0] - 39.5).abs() < 1e-4, "{:?} {:?}", a[10], a[39]);
        // Too few samples, or a stream that stops well short of the gyro's: none.
        assert!(accel_at_gyro(&gyro, &same[..10]).is_empty());
        let short: Vec<_> = (0..16).map(|i| at(100 + i, 1.0)).collect();
        let long_gyro: Vec<_> = (0..200).map(|i| at(100 + i, 0.0)).collect();
        assert!(accel_at_gyro(&long_gyro, &short).is_empty());
        assert!(accel_at_gyro(&[], &same).is_empty());
    }

    /// A take's frame timing is its sensor record's: the readout, and the
    /// centre row half of it after the stamp. Without the record, nothing.
    #[test]
    fn the_frame_timing_is_the_sensor_records() {
        let dir = std::env::temp_dir().join(format!("vr180-take-timing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut take = Take {
            dir: dir.clone(), fps_setting: 30, width: 3840, height: 3840, clock_start: None,
            audio_file: None, audio_timing: None, sensor_metadata: None, exposure_file: None,
            seal: x6b::Seal { fps_setting: 30, request_window: (1, 1), received_pairs: 1, pairs: 1, bytes: 8192 },
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
            dir: dir.clone(), fps_setting: 50, width: 3840, height: 3840,
            clock_start: Some(x6b::ClockAnchor { counter_ticks: 0, counter_hz: hz, ..Default::default() }),
            audio_file: None, audio_timing: None, sensor_metadata: Some("sensor-metadata.json".into()), exposure_file: None,
            seal: x6b::Seal { fps_setting: 50, request_window: (10, 13), received_pairs: 4, pairs: 4, bytes: 4 * 8192 },
            capture_bytes: 4 * 8192,
        };
        let pair = |request: u32| PairEntry {
            request, offset: 0, len: 8192, profile: x6b::Profile::ALL[0],
            exposure_ns: [500_000_000 + 20_000_000 * (request as u64 - 10), 500_000_000 + 20_000_000 * (request as u64 - 10)],
        };
        let index = TakeIndex::new(take.clone(), (10..14).map(pair).collect());
        assert!(index.shutter_ns.iter().all(|s| *s == [None, None]) && index.shutter_s(0, 0).is_none());
        // How far the camera has turned at each frame's sample, radians.
        let turned = |imu: &DjiOsvImu| -> Vec<f64> { imu.frame_quats.iter().map(|q| 2.0 * (q.w as f64).abs().min(1.0).acos()).collect() };
        let plain = turned(&build_imu(&index, 3840, 3840));
        // Half the readout after each stamp: 0.5 s + 7.678 ms, then 20 ms a frame.
        assert!((plain[0] - 0.507_678).abs() < 2e-4 && (plain[3] - plain[0] - 0.060).abs() < 2e-4, "{plain:?}");
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
            .map(|(i, t)| PairEntry { request: 10 + i as u32, offset: 0, len: 8192, profile: x6b::Profile::ALL[0], exposure_ns: [*t, *t] }).collect();
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
