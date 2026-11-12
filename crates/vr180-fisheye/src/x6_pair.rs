use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::insta360::InsvFrameStamp;
use crate::{Error, Result};

pub const PAIR_SIDECAR_SUFFIX: &str = ".x6clock.json";
const SCHEMA_MARK: &str = "-take-clock/";
const CLOCK_ERROR_MAX_NS: i64 = 10_000;
const CLOCK_DRIFT_SLACK_NS: i64 = 1_000;
const MIN_COMMON: usize = 3;
const FINGERPRINT_SPAN: u64 = 65536;

fn err(msg: impl Into<String>) -> Error {
    Error::Insv(msg.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairRole {
    Reference,
    Follow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairWitness {
    pub epoch: String,
    pub seq: u64,
    pub mono_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StampClock {
    Unstated,
    Ahead { offset_ns: i64, from_ns: u64, to_ns: u64 },
    Unusable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairSidecar {
    pub layout: u8,
    pub take_id: String,
    pub role: PairRole,
    pub boot: String,
    pub peer_boot: String,
    pub profile: String,
    pub original_name: String,
    pub file_size: u64,
    pub sample_sha256: String,
    pub stamp_clock: StampClock,
    pub witnesses: Vec<PairWitness>,
}

#[derive(serde::Deserialize)]
struct RawSidecar {
    #[serde(default)]
    schema: String,
    #[serde(default)]
    take_id: String,
    #[serde(default)]
    role: String,
    #[serde(default)]
    boot: String,
    #[serde(default)]
    peer_boot: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    original_name: String,
    #[serde(default)]
    file_id: RawFileId,
    #[serde(default)]
    exposure_clock: Option<RawExposureClock>,
    #[serde(default)]
    witnesses: Vec<RawWitness>,
}

#[derive(serde::Deserialize)]
struct RawExposureClock {
    #[serde(default)]
    boot_id: String,
    start: Option<RawClockSample>,
    end: Option<RawClockSample>,
}

#[derive(serde::Deserialize)]
struct RawClockSample {
    monotonic_ns: Option<u64>,
    boottime_minus_monotonic_ns: Option<i64>,
    error_bound_ns: Option<i64>,
}

#[derive(serde::Deserialize, Default)]
struct RawFileId {
    #[serde(default)]
    size: u64,
    #[serde(default)]
    sample_sha256: String,
}

#[derive(serde::Deserialize)]
struct RawWitness {
    #[serde(default)]
    epoch: String,
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    mono_ns: u64,
}

pub fn sidecar_path(media: &Path) -> PathBuf {
    let mut name = media.as_os_str().to_os_string();
    name.push(PAIR_SIDECAR_SUFFIX);
    PathBuf::from(name)
}

fn media_of_sidecar(sidecar: &Path) -> Option<PathBuf> {
    let name = sidecar.file_name()?.to_str()?;
    let stem = name.strip_suffix(PAIR_SIDECAR_SUFFIX)?;
    (!stem.is_empty()).then(|| sidecar.with_file_name(stem))
}

pub fn parse_sidecar(text: &str) -> Result<PairSidecar> {
    let raw: RawSidecar = serde_json::from_str(text).map_err(|e| err(format!("sidecar: {e}")))?;
    let layout: u8 = match raw.schema.rsplit_once(SCHEMA_MARK) {
        Some((_, "1")) => 1,
        Some((_, "2")) => 2,
        _ => return Err(err(format!("sidecar: unsupported layout {:?}", raw.schema))),
    };
    let role = match raw.role.as_str() {
        "R" => PairRole::Reference,
        "F" => PairRole::Follow,
        other => return Err(err(format!("sidecar: unknown role {other:?}"))),
    };
    if raw.take_id.is_empty() {
        return Err(err("sidecar: no id"));
    }
    let stamp_clock = if layout == 1 { StampClock::Unstated } else { stamp_clock_of(raw.exposure_clock, &raw.boot) };
    Ok(PairSidecar {
        layout,
        take_id: raw.take_id,
        role,
        boot: raw.boot,
        peer_boot: raw.peer_boot,
        profile: raw.profile,
        original_name: raw.original_name,
        file_size: raw.file_id.size,
        sample_sha256: raw.file_id.sample_sha256.to_ascii_lowercase(),
        stamp_clock,
        witnesses: raw
            .witnesses
            .into_iter()
            .map(|w| PairWitness { epoch: w.epoch, seq: w.seq, mono_ns: w.mono_ns })
            .collect(),
    })
}

fn stamp_clock_of(raw: Option<RawExposureClock>, boot: &str) -> StampClock {
    let bad = |why: &str| StampClock::Unusable(why.to_string());
    let Some(clock) = raw else {
        return bad("no clock data");
    };
    if clock.boot_id != boot {
        return bad("clock data from another session");
    }
    let sample = |s: Option<RawClockSample>| -> Option<(u64, i64, i64)> {
        let s = s?;
        Some((s.monotonic_ns?, s.boottime_minus_monotonic_ns?, s.error_bound_ns?))
    };
    let (Some((t0, off0, err0)), Some((t1, off1, err1))) = (sample(clock.start), sample(clock.end)) else {
        return bad("clock data incomplete");
    };
    if t1 <= t0 {
        return bad("clock data out of order");
    }
    if !(0..=CLOCK_ERROR_MAX_NS).contains(&err0) || !(0..=CLOCK_ERROR_MAX_NS).contains(&err1) {
        return StampClock::Unusable(format!("clock data too loose ({err0} / {err1} ns)"));
    }
    let moved = (off0 - off1).abs();
    if moved > err0 + err1 + CLOCK_DRIFT_SLACK_NS {
        return StampClock::Unusable(format!("clock offset changed by {moved} ns"));
    }
    StampClock::Ahead { offset_ns: (off0 + off1) / 2, from_ns: t0, to_ns: t1 }
}

pub fn read_sidecar(path: &Path) -> Result<PairSidecar> {
    parse_sidecar(&std::fs::read_to_string(path)?)
}

fn peek_take_id(path: &Path) -> Option<String> {
    let mut head = [0u8; 2048];
    let mut f = std::fs::File::open(path).ok()?;
    let n = f.read(&mut head).ok()?;
    let text = String::from_utf8_lossy(&head[..n]);
    let at = text.find("\"take_id\"")?;
    let rest = text[at + 9..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

pub fn file_fingerprint(media: &Path) -> std::io::Result<(u64, String)> {
    let mut f = std::fs::File::open(media)?;
    let size = f.metadata()?.len();
    let mut hash = Sha256::new();
    hash.update(size.to_string().as_bytes());
    hash.update(&[0]);
    let mut buf = vec![0u8; FINGERPRINT_SPAN as usize];
    for offset in [0, size / 2, size.saturating_sub(FINGERPRINT_SPAN)] {
        f.seek(SeekFrom::Start(offset))?;
        let mut got = 0usize;
        while got < buf.len() {
            let n = f.read(&mut buf[got..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        hash.update(&buf[..got]);
    }
    Ok((size, hex(&hash.finish())))
}

impl PairSidecar {
    pub fn belongs_to(&self, media: &Path) -> std::io::Result<bool> {
        if std::fs::metadata(media)?.len() != self.file_size {
            return Ok(false);
        }
        let (_, digest) = file_fingerprint(media)?;
        Ok(digest == self.sample_sha256)
    }

    pub fn moment_us(&self, w: &PairWitness) -> Option<f64> {
        match &self.stamp_clock {
            StampClock::Unstated => Some(w.mono_ns as f64 / 1000.0),
            StampClock::Ahead { offset_ns, from_ns, to_ns } => {
                (w.mono_ns >= *from_ns && w.mono_ns <= *to_ns).then(|| (w.mono_ns as f64 + *offset_ns as f64) / 1000.0)
            }
            StampClock::Unusable(_) => None,
        }
    }

    pub fn pairs_with(&self, other: &PairSidecar) -> bool {
        self.take_id == other.take_id
            && self.layout == other.layout
            && self.role != other.role
            && self.profile == other.profile
            && self.boot == other.peer_boot
            && self.peer_boot == other.boot
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairEye {
    Left,
    Right,
}

pub fn eye_of_folder(dir: &Path) -> Option<PairEye> {
    let name = dir.file_name()?.to_str()?.trim().to_lowercase();
    let stem = name
        .trim_end_matches('号')
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end_matches([' ', '-', '_', '#']);
    match stem {
        "l" | "left" | "左" | "左眼" => Some(PairEye::Left),
        "r" | "right" | "右" | "右眼" => Some(PairEye::Right),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct PairClips {
    pub left: PathBuf,
    pub right: PathBuf,
    pub left_sidecar: PairSidecar,
    pub right_sidecar: PairSidecar,
    pub eyes_from_folders: bool,
}

#[derive(Debug, Clone)]
pub enum PairSearch {
    NotPaired,
    Unusable(String),
    Found(Box<PairClips>),
}

pub fn find_pair(media: &Path) -> PairSearch {
    let own_path = sidecar_path(media);
    if !own_path.is_file() {
        return PairSearch::NotPaired;
    }
    let own = match read_sidecar(&own_path) {
        Ok(s) => s,
        Err(e) => return PairSearch::Unusable(e.to_string()),
    };
    match own.belongs_to(media) {
        Ok(true) => {}
        Ok(false) => return PairSearch::Unusable("sidecar does not match this file".into()),
        Err(e) => return PairSearch::Unusable(format!("read: {e}")),
    }
    let Some(own_dir) = media.parent() else {
        return PairSearch::Unusable("no parent folder".into());
    };

    let mut folders: Vec<PathBuf> = Vec::new();
    if let Some(parent) = own_dir.parent() {
        if let Ok(read) = std::fs::read_dir(parent) {
            let mut siblings: Vec<PathBuf> = read
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir() && p.as_path() != own_dir)
                .collect();
            let own_eye = eye_of_folder(own_dir);
            siblings.sort_by_key(|p| {
                let other_eye = match (own_eye, eye_of_folder(p)) {
                    (Some(a), Some(b)) if a != b => 0,
                    (_, Some(_)) => 1,
                    _ => 2,
                };
                (other_eye, p.clone())
            });
            folders.extend(siblings);
        }
    }
    folders.push(own_dir.to_path_buf());

    let mut seen_take = false;
    for folder in &folders {
        let Ok(read) = std::fs::read_dir(folder) else { continue };
        let mut candidates: Vec<PathBuf> = read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p != &own_path && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(PAIR_SIDECAR_SUFFIX)))
            .collect();
        candidates.sort();
        for path in candidates {
            if peek_take_id(&path).is_some_and(|id| id != own.take_id) {
                continue;
            }
            let Ok(other) = read_sidecar(&path) else { continue };
            if other.take_id != own.take_id {
                continue;
            }
            seen_take = true;
            if !own.pairs_with(&other) {
                continue;
            }
            let Some(other_media) = media_of_sidecar(&path).filter(|m| m.is_file()) else { continue };
            if !other.belongs_to(&other_media).unwrap_or(false) {
                continue;
            }
            for (side, clip) in [(&own, media), (&other, other_media.as_path())] {
                if let StampClock::Unusable(why) = &side.stamp_clock {
                    let name = clip.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                    return PairSearch::Unusable(format!(
                        "frames cannot be matched: {why} ({name})"));
                }
            }
            let (own_eye, other_eye) = (eye_of_folder(own_dir), other_media.parent().and_then(eye_of_folder));
            let (own_is_left, eyes_from_folders) = match (own_eye, other_eye) {
                (Some(a), Some(b)) if a != b => (a == PairEye::Left, true),
                (Some(a), None) => (a == PairEye::Left, true),
                (None, Some(b)) => (b == PairEye::Right, true),
                _ => (true, false),
            };
            let clips = if own_is_left {
                PairClips { left: media.to_path_buf(), right: other_media, left_sidecar: own, right_sidecar: other, eyes_from_folders }
            } else {
                PairClips { left: other_media, right: media.to_path_buf(), left_sidecar: other, right_sidecar: own, eyes_from_folders }
            };
            return PairSearch::Found(Box::new(clips));
        }
    }
    PairSearch::Unusable(if seen_take {
        "partner found but it does not fit (role, format, layout or ids)".into()
    } else {
        "partner not found in the neighbouring folders".into()
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockMap {
    pub origin_us: f64,
    pub offset_us: f64,
    pub skew: f64,
}

impl ClockMap {
    pub fn apply(&self, from_us: f64) -> f64 {
        from_us + self.offset_us + self.skew * (from_us - self.origin_us)
    }

    pub fn inverse(&self) -> ClockMap {
        ClockMap {
            origin_us: self.origin_us + self.offset_us,
            offset_us: -self.offset_us,
            skew: -self.skew / (1.0 + self.skew),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockFit {
    pub map: ClockMap,
    pub common: usize,
    pub used: usize,
    pub residual_rms_us: f64,
    pub first_us: f64,
    pub last_us: f64,
}

pub fn fit_clocks(a: &PairSidecar, b: &PairSidecar) -> Option<ClockFit> {
    let in_b: HashMap<(&str, u64), f64> =
        b.witnesses.iter().filter_map(|w| Some(((w.epoch.as_str(), w.seq), b.moment_us(w)?))).collect();
    let mut pairs: Vec<(f64, f64)> = a
        .witnesses
        .iter()
        .filter_map(|w| Some((a.moment_us(w)?, *in_b.get(&(w.epoch.as_str(), w.seq))?)))
        .collect();
    pairs.sort_by(|p, q| p.0.total_cmp(&q.0));
    pairs.dedup_by(|p, q| p.0 == q.0);
    let common = pairs.len();
    if common < MIN_COMMON {
        return None;
    }
    let origin = pairs[0].0;
    let line = |pts: &[(f64, f64)]| -> (f64, f64) {
        let n = pts.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(ta, tb) in pts {
            let (x, y) = (ta - origin, tb - ta);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let den = n * sxx - sx * sx;
        if den.abs() < 1e-6 {
            (0.0, sy / n)
        } else {
            let slope = (n * sxy - sx * sy) / den;
            (slope, (sy - slope * sx) / n)
        }
    };
    let residual = |p: &(f64, f64), (slope, at0): (f64, f64)| (p.1 - p.0) - (at0 + slope * (p.0 - origin));
    let mut fit = line(&pairs);
    let mut abs: Vec<f64> = pairs.iter().map(|p| residual(p, fit).abs()).collect();
    abs.sort_by(f64::total_cmp);
    let limit = (abs[abs.len() / 2] * 5.0).max(50.0);
    let kept: Vec<(f64, f64)> = pairs.iter().copied().filter(|p| residual(p, fit).abs() <= limit).collect();
    let used = if kept.len() >= MIN_COMMON && kept.len() < pairs.len() {
        fit = line(&kept);
        kept
    } else {
        pairs.clone()
    };
    let rms = (used.iter().map(|p| residual(p, fit).powi(2)).sum::<f64>() / used.len() as f64).sqrt();
    Some(ClockFit {
        map: ClockMap { origin_us: origin, offset_us: fit.1, skew: fit.0 },
        common,
        used: used.len(),
        residual_rms_us: rms,
        first_us: pairs[0].0,
        last_us: pairs[common - 1].0,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairSync {
    pub a_range: Range<usize>,
    pub b_frames: Vec<u32>,
    pub frame_offset: i64,
    pub held: usize,
    pub phase_median_us: f64,
    pub phase_spread_us: f64,
    pub frame_time_us: f64,
    pub out_of_step: usize,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

pub fn solve_pair_sync(a: &[InsvFrameStamp], b: &[InsvFrameStamp], fit: &ClockFit) -> Result<PairSync> {
    if a.len() < 2 || b.len() < 2 {
        return Err(err("pair: no per-frame stamps"));
    }
    let ta: Vec<f64> = a.iter().map(|s| s.t_us as f64).collect();
    let tb: Vec<f64> = b.iter().map(|s| s.t_us as f64).collect();
    let frame_time = {
        let mut d: Vec<f64> = ta.windows(2).map(|w| w[1] - w[0]).collect();
        median(&mut d)
    };
    if !(frame_time > 0.0) {
        return Err(err("pair: the frame stamps do not advance"));
    }
    let half = frame_time * 0.5;

    let first = ta.partition_point(|&t| t < fit.first_us - half);
    let last = ta.partition_point(|&t| t <= fit.last_us + half);
    let covered = |i: usize| {
        let t = fit.map.apply(ta[i]);
        t >= tb[0] - half && t <= tb[tb.len() - 1] + half
    };
    let first = (first..last).find(|&i| covered(i)).unwrap_or(last);
    let last = (first..last).rev().find(|&i| covered(i)).map_or(first, |i| i + 1);
    if last <= first {
        return Err(err("pair: no common span"));
    }

    let nearest = |i: usize| -> (usize, f64) {
        let t = fit.map.apply(ta[i]);
        let hi = tb.partition_point(|&x| x < t).min(tb.len() - 1);
        let lo = hi.saturating_sub(1);
        let j = if (tb[lo] - t).abs() <= (tb[hi] - t).abs() { lo } else { hi };
        (j, tb[j] - t)
    };
    let matched: Vec<(usize, f64)> = (first..last).map(nearest).collect();
    let mut steady: Vec<f64> = matched.iter().map(|m| m.1).filter(|p| p.abs() <= half).collect();
    if steady.is_empty() {
        return Err(err("pair: no frame within half a frame time"));
    }
    let phase_median = median(&mut steady);

    let tolerance = (frame_time * 0.03).max(300.0);
    let reach = ((250_000.0 / frame_time).round() as usize).max(2);
    let n = matched.len();
    let in_step: Vec<bool> = (0..n)
        .map(|k| {
            let mut window: Vec<f64> = matched[k.saturating_sub(reach)..(k + reach + 1).min(n)]
                .iter()
                .map(|m| m.1)
                .filter(|p| p.abs() <= half)
                .collect();
            window.is_empty() || (median(&mut window) - phase_median).abs() <= tolerance
        })
        .collect();
    let (mut best, mut run_start) = (0..0, 0usize);
    for k in 0..=n {
        if k == n || !in_step[k] {
            if k - run_start > best.len() {
                best = run_start..k;
            }
            run_start = k + 1;
        }
    }
    if best.is_empty() {
        return Err(err("pair: never in step"));
    }
    let kept = &matched[best.clone()];

    let mut offsets: HashMap<i64, usize> = HashMap::new();
    let mut phases: Vec<f64> = Vec::with_capacity(kept.len());
    let mut held = 0usize;
    for (k, &(j, phase)) in kept.iter().enumerate() {
        if phase.abs() > half {
            held += 1;
            continue;
        }
        *offsets.entry(j as i64 - (first + best.start + k) as i64).or_default() += 1;
        phases.push(phase);
    }
    let frame_offset = offsets.into_iter().max_by_key(|&(d, count)| (count, -d.abs())).map_or(0, |(d, _)| d);
    let phase_median_kept = median(&mut phases.clone());
    let mut deviations: Vec<f64> = phases.iter().map(|p| (p - phase_median_kept).abs()).collect();
    let phase_spread = median(&mut deviations) * 1.4826;

    Ok(PairSync {
        a_range: first + best.start..first + best.end,
        b_frames: kept.iter().map(|m| m.0 as u32).collect(),
        frame_offset,
        held,
        phase_median_us: phase_median_kept,
        phase_spread_us: phase_spread,
        frame_time_us: frame_time,
        out_of_step: n - best.len(),
    })
}

struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    length: u64,
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            block: [0; 64],
            filled: 0,
            length: 0,
        }
    }

    fn compress(&mut self) {
        let mut w = [0u32; 64];
        for (i, word) in self.block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled == 64 {
                self.compress();
                self.filled = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bits = self.length * 8;
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.block[56..64].copy_from_slice(&bits.to_be_bytes());
        self.compress();
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha_hex(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex(&h.finish())
    }

    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(sha_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let mut h = Sha256::new();
        let chunk = [b'a'; 997];
        let mut left = 1_000_000usize;
        while left > 0 {
            let n = left.min(chunk.len());
            h.update(&chunk[..n]);
            left -= n;
        }
        assert_eq!(hex(&h.finish()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    fn sidecar_json(take: &str, role: &str, boot: &str, peer: &str, profile: &str, size: u64, sha: &str, witnesses: &[(u64, u64)]) -> String {
        let w: Vec<String> = witnesses
            .iter()
            .map(|(seq, ns)| format!("{{\"epoch\":\"e1\",\"seq\":{seq},\"mono_ns\":{ns}}}"))
            .collect();
        format!(
            "{{\"boot\":\"{boot}\",\"file_id\":{{\"size\":{size},\"sample_sha256\":\"{sha}\"}},\"original_name\":\"x.mp4\",\
             \"peer_boot\":\"{peer}\",\"profile\":\"{profile}\",\"role\":\"{role}\",\"schema\":\"cam-take-clock/1\",\
             \"take_id\":\"{take}\",\"witnesses\":[{}]}}",
            w.join(",")
        )
    }

    #[test]
    fn parse_layouts() {
        let s = parse_sidecar(&sidecar_json("t1", "F", "b1", "b2", "p", 10, "AB", &[(5, 1000), (9, 2000)])).unwrap();
        assert_eq!((s.take_id.as_str(), s.role, s.boot.as_str(), s.peer_boot.as_str()), ("t1", PairRole::Follow, "b1", "b2"));
        assert_eq!((s.file_size, s.sample_sha256.as_str(), s.witnesses.len()), (10, "ab", 2));
        assert_eq!(s.witnesses[1], PairWitness { epoch: "e1".into(), seq: 9, mono_ns: 2000 });
        assert_eq!((s.layout, &s.stamp_clock), (1, &StampClock::Unstated));
        assert_eq!(s.moment_us(&s.witnesses[1]), Some(2.0));
        for unknown in ["take-clock/3", "take-clock/", "take-clock/12", "other/1"] {
            let other = sidecar_json("t1", "F", "b1", "b2", "p", 10, "ab", &[]).replace("take-clock/1", unknown);
            assert!(parse_sidecar(&other).is_err(), "{unknown}");
        }
        assert!(parse_sidecar(&sidecar_json("t1", "X", "b1", "b2", "p", 10, "ab", &[])).is_err());
        assert!(parse_sidecar(&sidecar_json("", "R", "b1", "b2", "p", 10, "ab", &[])).is_err());
        assert!(parse_sidecar("not json").is_err());
    }

    fn sidecar_json_v2(role: &str, boot: &str, peer: &str, clock: Option<&str>, witnesses: &[(u64, u64)]) -> String {
        let base = sidecar_json("t", role, boot, peer, "p", 1, "00", witnesses).replace("take-clock/1", "take-clock/2");
        match clock {
            Some(c) => base.replacen('{', &format!("{{\"exposure_clock\":{c},"), 1),
            None => base,
        }
    }

    fn clock_json(boot: &str, start: (u64, i64, i64), end: (u64, i64, i64)) -> String {
        let one = |(t, off, e): (u64, i64, i64)| format!("{{\"monotonic_ns\":{t},\"boottime_minus_monotonic_ns\":{off},\"error_bound_ns\":{e}}}");
        format!("{{\"boot_id\":\"{boot}\",\"start\":{},\"end\":{}}}", one(start), one(end))
    }

    #[test]
    fn layout2_clock() {
        let parse = |clock: Option<String>| parse_sidecar(&sidecar_json_v2("F", "b1", "b2", clock.as_deref(), &[(1, 5_000), (2, 9_000_000)])).unwrap();
        let s = parse(Some(clock_json("b1", (1_000, 32_290_261_837, 339), (8_000_000, 32_290_261_862, 990))));
        assert_eq!(s.layout, 2);
        assert_eq!(s.stamp_clock, StampClock::Ahead { offset_ns: 32_290_261_849, from_ns: 1_000, to_ns: 8_000_000 });
        assert_eq!(s.moment_us(&s.witnesses[0]), Some((5_000.0 + 32_290_261_849.0) / 1000.0));
        assert_eq!(s.moment_us(&s.witnesses[1]), None);

        let unusable = |clock: Option<String>, what: &str| {
            let s = parse(clock);
            assert!(matches!(s.stamp_clock, StampClock::Unusable(_)), "{what}: {:?}", s.stamp_clock);
            assert_eq!(s.moment_us(&s.witnesses[0]), None, "{what}");
        };
        unusable(None, "none");
        unusable(Some(clock_json("b9", (1_000, 5, 10), (2_000, 5, 10))), "session");
        unusable(Some(clock_json("b1", (2_000, 5, 10), (2_000, 5, 10))), "order");
        unusable(Some(clock_json("b1", (1_000, 5, 10_001), (2_000, 5, 10))), "error");
        unusable(Some(clock_json("b1", (1_000, 5, 10), (2_000, 5, -1))), "negative");
        unusable(Some(clock_json("b1", (1_000, 0, 300), (2_000, 2_001, 700))), "moved");
        assert!(matches!(parse(Some(clock_json("b1", (1_000, 0, 300), (2_000, 2_000, 700)))).stamp_clock, StampClock::Ahead { offset_ns: 1_000, .. }));
        unusable(Some("{\"boot_id\":\"b1\",\"start\":{\"monotonic_ns\":1}}".to_string()), "incomplete");
    }

    #[test]
    fn fit_with_offsets() {
        let (off_a, off_b, apart) = (32_290_261_862i64, 31_873_810_417i64, 26_000_000_000i64);
        let moments: Vec<(u64, u64)> = (0..200).map(|i| (i as u64, 17_000_000_000_000 + i as u64 * 55_000_000)).collect();
        let far = (moments[0].1 - 1_000_000, moments[199].1 + 1_000_000);
        let a = parse_sidecar(&sidecar_json_v2("R", "b1", "b2", Some(&clock_json("b1", (far.0, off_a, 500), (far.1, off_a, 500))), &moments)).unwrap();
        let moments_b: Vec<(u64, u64)> = moments.iter().map(|&(seq, t)| (seq, (t as i64 + apart) as u64)).collect();
        let span_b = ((far.0 as i64 + apart) as u64, (far.1 as i64 + apart) as u64);
        let b = parse_sidecar(&sidecar_json_v2("F", "b2", "b1", Some(&clock_json("b2", (span_b.0, off_b, 500), (span_b.1, off_b, 500))), &moments_b)).unwrap();
        let fit = fit_clocks(&a, &b).unwrap();
        assert_eq!(fit.common, 200);
        let expect_us = (apart + off_b - off_a) as f64 / 1000.0;
        assert!((fit.map.offset_us - expect_us).abs() < 1.0, "{} vs {expect_us}", fit.map.offset_us);
        assert!(fit.map.skew.abs() < 1e-9);
        assert!((fit.first_us - (moments[0].1 as f64 + off_a as f64) / 1000.0).abs() < 1.0);
        let plain = |s: &PairSidecar| PairSidecar { stamp_clock: StampClock::Unstated, ..s.clone() };
        let naive = fit_clocks(&plain(&a), &plain(&b)).unwrap();
        assert!(((naive.map.offset_us - fit.map.offset_us).abs() - 416_451.4).abs() < 1.0);
        let narrow = PairSidecar { stamp_clock: StampClock::Ahead { offset_ns: off_a, from_ns: moments[50].1, to_ns: moments[149].1 }, ..a.clone() };
        assert_eq!(fit_clocks(&narrow, &b).unwrap().common, 100);
        let broken = PairSidecar { stamp_clock: StampClock::Unusable("x".into()), ..a.clone() };
        assert!(fit_clocks(&broken, &b).is_none());
    }

    #[test]
    fn pairs_with_rules() {
        let side = |take, role, boot, peer, profile| parse_sidecar(&sidecar_json(take, role, boot, peer, profile, 1, "00", &[])).unwrap();
        let r = side("t", "R", "b1", "b2", "p");
        assert!(r.pairs_with(&side("t", "F", "b2", "b1", "p")));
        assert!(!r.pairs_with(&side("t", "R", "b2", "b1", "p")));
        assert!(!r.pairs_with(&side("u", "F", "b2", "b1", "p")));
        assert!(!r.pairs_with(&side("t", "F", "b3", "b1", "p")));
        assert!(!r.pairs_with(&side("t", "F", "b2", "b9", "p")));
        assert!(!r.pairs_with(&side("t", "F", "b2", "b1", "q")));
        let newer = parse_sidecar(&sidecar_json_v2("F", "b2", "b1", None, &[])).unwrap();
        assert!(!r.pairs_with(&newer));
        assert!(side("t", "F", "b1", "b2", "p").pairs_with(&side("t", "R", "b2", "b1", "p")));
    }

    #[test]
    fn folder_eye() {
        for (name, eye) in [("L", PairEye::Left), ("left", PairEye::Left), ("R", PairEye::Right), ("Right", PairEye::Right), ("左", PairEye::Left), ("右眼", PairEye::Right),
                            ("L1", PairEye::Left), ("R2", PairEye::Right), ("right 2", PairEye::Right), ("left_01", PairEye::Left), ("左眼1号", PairEye::Left), ("右2", PairEye::Right)] {
            assert_eq!(eye_of_folder(Path::new("/clips").join(name).as_path()), Some(eye), "{name}");
        }
        for name in ["clips", "LR", "rear", "day 1", "3", "LR1", "r2d2", "x 012"] {
            assert_eq!(eye_of_folder(Path::new("/x").join(name).as_path()), None, "{name}");
        }
    }

    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("vr180-pair-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_clip(dir: &Path, name: &str, len: usize, seed: u8, take: &str, role: &str, boot: &str, peer: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let media = dir.join(name);
        let bytes: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect();
        std::fs::write(&media, &bytes).unwrap();
        let (size, sha) = file_fingerprint(&media).unwrap();
        std::fs::write(sidecar_path(&media), sidecar_json(take, role, boot, peer, "p", size, &sha, &[(1, 10), (2, 20), (3, 30)])).unwrap();
        media
    }

    #[test]
    fn fingerprint() {
        let scratch = Scratch::new("fp");
        let len = 5 * FINGERPRINT_SPAN as usize + 123;
        let media = write_clip(&scratch.0, "a.mp4", len, 7, "t", "R", "b1", "b2");
        let bytes = std::fs::read(&media).unwrap();
        let mut h = Sha256::new();
        h.update(len.to_string().as_bytes());
        h.update(&[0]);
        let span = FINGERPRINT_SPAN as usize;
        for at in [0, len / 2, len - span] {
            h.update(&bytes[at..at + span]);
        }
        assert_eq!(file_fingerprint(&media).unwrap(), (len as u64, hex(&h.finish())));
        let own = read_sidecar(&sidecar_path(&media)).unwrap();
        assert!(own.belongs_to(&media).unwrap());

        let mut changed = bytes.clone();
        changed[span + 5] ^= 1;
        std::fs::write(&media, &changed).unwrap();
        assert!(own.belongs_to(&media).unwrap());
        changed[len / 2 + 5] ^= 1;
        std::fs::write(&media, &changed).unwrap();
        assert!(!own.belongs_to(&media).unwrap());

        let small = write_clip(&scratch.0, "small.mp4", 1000, 3, "t2", "R", "b1", "b2");
        let sb = std::fs::read(&small).unwrap();
        let mut h = Sha256::new();
        h.update(b"1000");
        h.update(&[0]);
        h.update(&sb);
        h.update(&sb[500..]);
        h.update(&sb);
        assert_eq!(file_fingerprint(&small).unwrap().1, hex(&h.finish()));
    }

    #[test]
    fn find_basic() {
        let scratch = Scratch::new("find");
        let (l, r) = (scratch.0.join("L"), scratch.0.join("R"));
        let l1 = write_clip(&l, "PRO_001.mp4", 4000, 1, "take1", "F", "bL", "bR");
        let r1 = write_clip(&r, "PRO_771.mp4", 4100, 2, "take1", "R", "bR", "bL");
        let l2 = write_clip(&l, "PRO_002.mp4", 4200, 3, "take2", "F", "bL", "bR");
        write_clip(&r, "PRO_772.mp4", 4300, 4, "take9", "R", "bR", "bL");
        std::fs::write(r.join("notes.txt"), "x").unwrap();

        for asked in [&l1, &r1] {
            match find_pair(asked) {
                PairSearch::Found(p) => {
                    assert_eq!((&p.left, &p.right), (&l1, &r1), "asked for {asked:?}");
                    assert!(p.eyes_from_folders);
                    assert_eq!((p.left_sidecar.role, p.right_sidecar.role), (PairRole::Follow, PairRole::Reference));
                }
                other => panic!("{other:?}"),
            }
        }
        assert!(matches!(find_pair(&l2), PairSearch::Unusable(_)));
        assert!(matches!(find_pair(&r.join("notes.txt")), PairSearch::NotPaired));

        std::fs::write(&r1, vec![9u8; 4100]).unwrap();
        assert!(matches!(find_pair(&r1), PairSearch::Unusable(_)));
        assert!(matches!(find_pair(&l1), PairSearch::Unusable(_)));
    }

    #[test]
    fn find_roles_and_clock() {
        let scratch = Scratch::new("roles");
        let take = scratch.0.join("t3");
        let l = write_clip(&take.join("L1"), "a.mp4", 3000, 1, "t", "R", "bA", "bB");
        let r = write_clip(&take.join("R2"), "b.mp4", 3100, 2, "t", "F", "bB", "bA");
        for asked in [&l, &r] {
            let PairSearch::Found(p) = find_pair(asked) else { panic!() };
            assert_eq!((&p.left, &p.right, p.eyes_from_folders), (&l, &r, true));
            assert_eq!((p.left_sidecar.role, p.right_sidecar.role), (PairRole::Reference, PairRole::Follow));
        }
        let rewrite = |media: &Path, role: &str, boot: &str, peer: &str, clock: Option<String>| {
            let (size, sha) = file_fingerprint(media).unwrap();
            let text = sidecar_json_v2(role, boot, peer, clock.as_deref(), &[(1, 10), (2, 20), (3, 30)])
                .replace("\"size\":1,", &format!("\"size\":{size},"))
                .replace("\"sample_sha256\":\"00\"", &format!("\"sample_sha256\":\"{sha}\""));
            std::fs::write(sidecar_path(media), text).unwrap();
        };
        rewrite(&l, "R", "bA", "bB", Some(clock_json("bA", (1, 500, 10), (100, 500, 10))));
        rewrite(&r, "F", "bB", "bA", None);
        for asked in [&l, &r] {
            match find_pair(asked) {
                PairSearch::Unusable(why) => assert!(why.contains("cannot be matched") && why.contains("b.mp4"), "{why}"),
                other => panic!("{other:?}"),
            }
        }
        rewrite(&r, "F", "bB", "bA", Some(clock_json("bB", (1, 900, 10), (100, 900, 10))));
        assert!(matches!(find_pair(&l), PairSearch::Found(_)));
    }

    #[test]
    fn find_unnamed_folders() {
        let scratch = Scratch::new("anon");
        let a = write_clip(&scratch.0.join("one"), "a.mp4", 3000, 1, "t", "R", "b1", "b2");
        let b = write_clip(&scratch.0.join("two"), "b.mp4", 3100, 2, "t", "F", "b2", "b1");
        let PairSearch::Found(p) = find_pair(&b) else { panic!() };
        assert_eq!((&p.left, &p.right, p.eyes_from_folders), (&b, &a, false));
        let scratch = Scratch::new("half");
        let a = write_clip(&scratch.0.join("R"), "a.mp4", 3000, 1, "t", "R", "b1", "b2");
        let b = write_clip(&scratch.0.join("other"), "b.mp4", 3100, 2, "t", "F", "b2", "b1");
        for asked in [&a, &b] {
            let PairSearch::Found(p) = find_pair(asked) else { panic!() };
            assert_eq!((&p.left, &p.right, p.eyes_from_folders), (&b, &a, true));
        }
    }

    fn stamps(t0: u64, step: f64, n: usize) -> Vec<InsvFrameStamp> {
        (0..n).map(|i| InsvFrameStamp { t_us: t0 + (i as f64 * step).round() as u64, exposure_s: 0.01 }).collect()
    }

    fn witness_pair(times_a_us: &[f64], map: &ClockMap, noise_us: &[f64]) -> (PairSidecar, PairSidecar) {
        let make = |role: &str, ns: Vec<u64>| {
            let w: Vec<(u64, u64)> = ns.iter().enumerate().map(|(i, &t)| (10 * i as u64, t)).collect();
            parse_sidecar(&sidecar_json("t", role, if role == "R" { "b1" } else { "b2" }, if role == "R" { "b2" } else { "b1" }, "p", 1, "00", &w)).unwrap()
        };
        let a: Vec<u64> = times_a_us.iter().map(|t| (t * 1000.0).round() as u64).collect();
        let b: Vec<u64> = times_a_us
            .iter()
            .enumerate()
            .map(|(i, &t)| ((map.apply(t) + noise_us[i % noise_us.len()]) * 1000.0).round() as u64)
            .collect();
        (make("R", a), make("F", b))
    }

    #[test]
    fn fit_line() {
        let truth = ClockMap { origin_us: 5.0e9, offset_us: -31_500_000.0, skew: -6.0e-6 };
        let times: Vec<f64> = (0..600).map(|i| 5.0e9 + 16.0e6 + i as f64 * 55_000.0).collect();
        let mut noise: Vec<f64> = (0..600).map(|i| ((i * 7919) % 401) as f64 - 200.0).collect();
        noise[100] = 9000.0;
        noise[400] = -7000.0;
        let (a, b) = witness_pair(&times, &truth, &noise);
        let fit = fit_clocks(&a, &b).unwrap();
        assert_eq!((fit.common, fit.used), (600, 598));
        assert!((fit.first_us - times[0]).abs() < 1.0 && (fit.last_us - times[599]).abs() < 1.0);
        for t in [times[0], times[300], times[599]] {
            assert!((fit.map.apply(t) - truth.apply(t)).abs() < 30.0, "{}", fit.map.apply(t) - truth.apply(t));
        }
        assert!(fit.residual_rms_us > 50.0 && fit.residual_rms_us < 200.0, "{}", fit.residual_rms_us);
        let back = fit.map.inverse();
        for t in [times[0], times[599] + 3.0e7] {
            assert!((back.apply(fit.map.apply(t)) - t).abs() < 1e-3);
        }
        let rev = fit_clocks(&b, &a).unwrap();
        assert!((rev.map.apply(truth.apply(times[300])) - times[300]).abs() < 30.0);
        let (a2, b2) = witness_pair(&times[..2], &truth, &[0.0]);
        assert!(fit_clocks(&a2, &b2).is_none());
    }

    fn two_clips(n: usize, b_lead: usize, lock_at: f64, phase_before: f64, phase_after: f64) -> (Vec<InsvFrameStamp>, Vec<InsvFrameStamp>, ClockFit) {
        let step: f64 = 1.0e6 * 1001.0 / 30000.0;
        let ta0 = 7.0e9;
        let map = ClockMap { origin_us: ta0, offset_us: 4.2e7, skew: -5.0e-6 };
        let a = stamps(ta0 as u64, step, n);
        let b: Vec<InsvFrameStamp> = (0..n + b_lead)
            .map(|j| {
                let i = j as f64 - b_lead as f64;
                let t_a = ta0 + i * step;
                let phase = if i * step < lock_at * 1.0e6 { phase_before } else { phase_after };
                InsvFrameStamp { t_us: (map.apply(t_a) + phase).round() as u64, exposure_s: 0.01 }
            })
            .collect();
        let first = ta0 + lock_at * 1.0e6 + 2.0e6;
        let fit = ClockFit { map, common: 500, used: 500, residual_rms_us: 100.0, first_us: first, last_us: ta0 + (n as f64 - 3.0) * step };
        (a, b, fit)
    }

    #[test]
    fn sync_basic() {
        let (a, b, fit) = two_clips(2000, 151, 14.0, -15_000.0, -180.0);
        let sync = solve_pair_sync(&a, &b, &fit).unwrap();
        let step: f64 = 1.0e6 * 1001.0 / 30000.0;
        let first = (16.0e6 / step).ceil() as usize;
        assert_eq!(sync.a_range, first..1998);
        assert_eq!(sync.frame_offset, 151);
        assert_eq!(sync.b_frames.len(), sync.a_range.len());
        assert!(sync.b_frames.iter().enumerate().all(|(k, &j)| j as usize == first + k + 151));
        assert_eq!((sync.held, sync.out_of_step), (0, 0));
        assert!((sync.phase_median_us + 180.0).abs() < 2.0, "{}", sync.phase_median_us);
        assert!((sync.frame_time_us - step).abs() < 1.0);
    }

    #[test]
    fn sync_span() {
        let (a, b, mut fit) = two_clips(600, 0, 0.0, 0.0, 0.0);
        let step: f64 = 1.0e6 * 1001.0 / 30000.0;
        let at = |frame: f64| a[0].t_us as f64 + frame * step;
        (fit.first_us, fit.last_us) = (at(100.3), at(499.7));
        assert_eq!(solve_pair_sync(&a, &b, &fit).unwrap().a_range, 100..501);
        (fit.first_us, fit.last_us) = (at(99.7), at(500.3));
        assert_eq!(solve_pair_sync(&a, &b, &fit).unwrap().a_range, 100..501);
        (fit.first_us, fit.last_us) = (at(100.6), at(499.4));
        assert_eq!(solve_pair_sync(&a, &b, &fit).unwrap().a_range, 101..500);
    }

    #[test]
    fn sync_dropped() {
        let (a, mut b, fit) = two_clips(1500, 0, 5.0, 0.0, 0.0);
        b.drain(800..803);
        let sync = solve_pair_sync(&a, &b, &fit).unwrap();
        let at = |i: usize| sync.b_frames[i - sync.a_range.start] as usize;
        assert_eq!((at(799), at(803), at(1200)), (799, 800, 1197));
        assert!(matches!(at(801), 799 | 800));
        assert_eq!((at(800), at(802)), (799, 800));
        assert_eq!(sync.held, 3);
        assert_eq!(sync.frame_offset, -3);
        assert_eq!(sync.out_of_step, 0);
    }

    #[test]
    fn sync_out_of_step() {
        let (a, mut b, mut fit) = two_clips(1500, 0, 0.0, 0.0, 0.0);
        let step: f64 = 1.0e6 * 1001.0 / 30000.0;
        for s in b.iter_mut().skip(1410) {
            s.t_us += 6000;
        }
        fit.first_us = a[30].t_us as f64;
        fit.last_us = a[1490].t_us as f64;
        let sync = solve_pair_sync(&a, &b, &fit).unwrap();
        assert_eq!(sync.a_range.start, 30);
        let reach = (250_000.0 / step).round() as usize;
        assert!(sync.a_range.end >= 1410 - reach && sync.a_range.end <= 1410 + reach, "{:?}", sync.a_range);
        assert!(sync.out_of_step >= 1490 - 1410 - reach);
        assert!(sync.phase_median_us.abs() < 2.0);

        let (a, b, mut fit) = two_clips(300, 0, 0.0, 0.0, 0.0);
        fit.map.offset_us += 1.0e9;
        assert!(solve_pair_sync(&a, &b, &fit).is_err());
    }
}
