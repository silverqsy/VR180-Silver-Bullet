use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use vr180_fisheye::insta360::{read_insta360_identity, read_insta360_meta, InsvFrameStamp};
use vr180_fisheye::x6_pair::{find_pair, fit_clocks, solve_pair_sync, ClockFit, PairSearch, PairSync, StampClock};
use vr180_fisheye::Insta360Meta;

use crate::decode::VideoProbe;
use crate::frame_timeline::FrameTimeline;

#[derive(Debug, Clone)]
pub struct PairPlan {
    pub paths: [PathBuf; 2],
    pub clip_timelines: [FrameTimeline; 2],
    pub range: Range<usize>,
    pub partner: Vec<u32>,
    pub timeline: FrameTimeline,
    pub probe: VideoProbe,
    pub meta: Insta360Meta,
    pub audio_slot: usize,
    pub clip_start_s: [f64; 2],
    pub sync: PairSync,
    pub clocks: ClockFit,
}

impl PairPlan {
    pub fn clip_frame(&self, slot: usize, row: usize) -> usize {
        let row = row.min(self.partner.len().saturating_sub(1));
        if slot == 0 { self.range.start + row } else { self.partner[row] as usize }
    }

    pub fn clip_time(&self, slot: usize, t_s: f64) -> f64 {
        let row = self.timeline.row(t_s.max(0.0));
        let within = t_s.max(0.0) - self.timeline.pts(row).unwrap_or(0.0);
        self.clip_timelines[slot].pts(self.clip_frame(slot, row)).unwrap_or(0.0) + within.max(0.0)
    }

    pub fn audio_source(&self) -> (&Path, f64) {
        (&self.paths[self.audio_slot], self.clip_start_s[self.audio_slot])
    }
}

type PlanCache = Mutex<HashMap<PathBuf, Option<Arc<PairPlan>>>>;

fn cache() -> &'static PlanCache {
    static CACHE: OnceLock<PlanCache> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

thread_local! {
    static BUILDING: Cell<bool> = const { Cell::new(false) };
}

pub fn plan_for(path: &Path) -> Option<Arc<PairPlan>> {
    if BUILDING.with(|b| b.get()) {
        return None;
    }
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
    if !matches!(ext.as_deref(), Some("mp4" | "mov" | "insv")) {
        return None;
    }
    let mut plans = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(known) = plans.get(path) {
        return known.clone();
    }
    BUILDING.with(|b| b.set(true));
    let built = build_plan(path);
    BUILDING.with(|b| b.set(false));
    let plan = match built {
        Ok(plan) => plan.map(Arc::new),
        Err(why) => {
            tracing::warn!("{}: sidecar present, not usable: {why}", path.display());
            None
        }
    };
    if plans.len() > 256 {
        plans.clear();
    }
    match &plan {
        Some(p) => {
            for clip in &p.paths {
                plans.insert(clip.clone(), Some(p.clone()));
            }
            plans.insert(path.to_path_buf(), Some(p.clone()));
        }
        None => {
            plans.insert(path.to_path_buf(), None);
        }
    }
    plan
}

pub fn forget(path: &Path) {
    let mut plans = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Some(plan)) = plans.remove(path) {
        for clip in &plan.paths {
            plans.remove(clip);
        }
    }
}

pub fn audio_source(path: &Path) -> (PathBuf, f64) {
    match plan_for(path) {
        Some(plan) => {
            let (clip, start_s) = plan.audio_source();
            (clip.to_path_buf(), start_s)
        }
        None => (path.to_path_buf(), 0.0),
    }
}

pub fn read_meta(path: &Path) -> vr180_fisheye::Result<Insta360Meta> {
    match plan_for(path) {
        Some(plan) => Ok(plan.meta.clone()),
        None => read_insta360_meta(path),
    }
}

pub fn read_identity(path: &Path) -> vr180_fisheye::Result<Insta360Meta> {
    match plan_for(path) {
        Some(plan) => {
            let mut meta = plan.meta.clone();
            meta.imu.clear();
            meta.frames.clear();
            meta.frames_b.clear();
            Ok(meta)
        }
        None => read_insta360_identity(path),
    }
}

fn build_plan(path: &Path) -> Result<Option<PairPlan>, String> {
    let clips = match find_pair(path) {
        PairSearch::NotPaired => return Ok(None),
        PairSearch::Unusable(why) => return Err(why),
        PairSearch::Found(clips) => clips,
    };
    let paths = [clips.right.clone(), clips.left.clone()];
    let sidecars = [&clips.right_sidecar, &clips.left_sidecar];
    let name = |p: &Path| p.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();

    let mut metas = Vec::with_capacity(2);
    for p in &paths {
        let meta = read_insta360_meta(p).map_err(|e| format!("{}: {e}", name(p)))?;
        if !meta.single_lens || meta.lenses.len() != 1 {
            return Err(format!("{}: not a single-lens clip", name(p)));
        }
        if meta.frames.len() < 2 {
            return Err(format!("{}: no per-frame stamps", name(p)));
        }
        metas.push(meta);
    }
    let (meta1, meta0) = (metas.pop().unwrap(), metas.pop().unwrap());
    if meta0.dimension != meta1.dimension || meta0.window_crop != meta1.window_crop {
        return Err("frame layout differs".into());
    }

    let clocks = fit_clocks(sidecars[0], sidecars[1])
        .ok_or("fewer than three common marks")?;
    let sync = solve_pair_sync(&meta0.frames, &meta1.frames, &clocks).map_err(|e| e.to_string())?;
    if clocks.residual_rms_us > 0.2 * sync.frame_time_us || sync.phase_spread_us > 0.2 * sync.frame_time_us {
        return Err(format!(
            "clocks do not line up (rms {:.0} µs marks, {:.0} µs frames, frame time {:.0} µs)",
            clocks.residual_rms_us, sync.phase_spread_us, sync.frame_time_us));
    }

    let clip_timelines = [
        FrameTimeline::from_mp4_file(&paths[0]).map_err(|e| format!("{}: {e}", name(&paths[0])))?,
        FrameTimeline::from_mp4_file(&paths[1]).map_err(|e| format!("{}: {e}", name(&paths[1])))?,
    ];
    let mut range = sync.a_range.clone();
    let mut partner = sync.b_frames.clone();
    while !partner.is_empty()
        && (range.end > clip_timelines[0].len() || *partner.last().unwrap() as usize >= clip_timelines[1].len())
    {
        partner.pop();
        range.end -= 1;
    }
    if partner.len() < 2 {
        return Err("empty span".into());
    }
    let t0 = clip_timelines[0].pts(range.start).unwrap_or(0.0);
    let dt = clip_timelines[0].dt_s();
    let timeline = FrameTimeline::from_pts(
        range.clone().map(|i| clip_timelines[0].pts(i).unwrap_or(t0) - t0).collect(),
        dt,
    );
    let clip_start_s = [t0, clip_timelines[1].pts(partner[0] as usize).unwrap_or(0.0)];

    let probes = [
        crate::decode::probe_video_file(&paths[0]).map_err(|e| e.to_string())?,
        crate::decode::probe_video_file(&paths[1]).map_err(|e| e.to_string())?,
    ];
    if (probes[0].width, probes[0].height) != (probes[1].width, probes[1].height) {
        return Err(format!(
            "frame size differs ({}x{} / {}x{})",
            probes[0].width, probes[0].height, probes[1].width, probes[1].height));
    }
    if (probes[0].fps - probes[1].fps).abs() > 0.01 {
        return Err(format!("frame rate differs ({} / {})", probes[0].fps, probes[1].fps));
    }
    let probe = VideoProbe {
        duration_sec: timeline.duration_s(),
        video_bitrates_bps: [probes[0].video_bitrates_bps[0], probes[1].video_bitrates_bps[0]],
        ..probes[0]
    };

    let four_channel = |p: &Path| crate::audio::probe_foa_aac(p).ok().flatten().is_some();
    let audio_slot = if !four_channel(&paths[0]) && four_channel(&paths[1]) { 1 } else { 0 };

    let meta = merged_meta(meta0, &meta1, &range, &partner, &clocks);
    let seconds = |frames: usize| frames as f64 * dt;
    let offsets = match (&sidecars[0].stamp_clock, &sidecars[1].stamp_clock) {
        (StampClock::Ahead { offset_ns: right, .. }, StampClock::Ahead { offset_ns: left, .. }) => {
            format!(", stamp offsets {:.3} s / {:.3} s", *right as f64 * 1e-9, *left as f64 * 1e-9)
        }
        _ => String::new(),
    };
    tracing::info!(
        "pair: R {} + L {}{} — {} frames, start {:.2} s / {:.2} s, offset {}, phase {:+.0} µs (spread {:.0}), \
         {} marks (rms {:.0} µs, skew {:+.1} ppm{}), held {}, cut {}; unused {:.2} s head, {:.2} s tail",
        name(&paths[0]), name(&paths[1]),
        if clips.eyes_from_folders { "" } else { " (eyes by open order)" },
        partner.len(), clip_start_s[0], clip_start_s[1],
        sync.frame_offset, sync.phase_median_us, sync.phase_spread_us,
        clocks.common, clocks.residual_rms_us, clocks.map.skew * 1e6, offsets, sync.held, sync.out_of_step,
        seconds(range.start), seconds(clip_timelines[0].len().saturating_sub(range.end)),
    );
    Ok(Some(PairPlan {
        paths, clip_timelines, range, partner, timeline, probe, meta, audio_slot, clip_start_s, sync, clocks,
    }))
}

fn merged_meta(
    right: Insta360Meta,
    left: &Insta360Meta,
    range: &Range<usize>,
    partner: &[u32],
    clocks: &ClockFit,
) -> Insta360Meta {
    let to_right_clock = clocks.map.inverse();
    let rows = range.len().min(partner.len());
    let frames_b: Vec<InsvFrameStamp> = right.frames[range.start..range.start + rows].to_vec();
    let frames: Vec<InsvFrameStamp> = partner[..rows]
        .iter()
        .map(|&j| {
            let s = left.frames[(j as usize).min(left.frames.len() - 1)];
            InsvFrameStamp { t_us: to_right_clock.apply(s.t_us as f64).round().max(0.0) as u64, exposure_s: s.exposure_s }
        })
        .collect();
    let mut lenses = left.lenses.clone();
    lenses.extend(right.lenses.iter().cloned());
    Insta360Meta {
        lenses,
        frames,
        frames_b,
        track_order_reversed: Some(true),
        single_lens: true,
        ..right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vr180_fisheye::insta360::parse_calib_string;
    use vr180_fisheye::x6_pair::ClockMap;

    fn lens(fx: f32) -> vr180_fisheye::InsvLensCalib {
        let s = format!("1_2.455430_{fx}_{fx}_3842.500_3888.000_0_0_89.5_0_0_-0.0326_1.3_-1.4_5.8_1.0_0_0_0_0_0_0_0_0_0_7744_7744_193_1");
        parse_calib_string(&s).unwrap().remove(0)
    }

    fn clip_meta(fx: f32, t0: u64, n: usize) -> Insta360Meta {
        Insta360Meta {
            model: "Insta360 X6".into(),
            lenses: vec![lens(fx)],
            frames: (0..n).map(|i| InsvFrameStamp { t_us: t0 + 33_367 * i as u64, exposure_s: 0.01 }).collect(),
            readout_ms: Some(30.72),
            single_lens: true,
            ..Default::default()
        }
    }

    #[test]
    fn merged_meta_layout() {
        let (right, left) = (clip_meta(7221.0, 5_000_000_000, 2000), clip_meta(7210.0, 5_040_000_000 - 151 * 33_367 + 900 * 33_367, 2000));
        let clocks = ClockFit {
            map: ClockMap { origin_us: 5.0e9, offset_us: 4.0e7, skew: 0.0 },
            common: 10, used: 10, residual_rms_us: 50.0, first_us: 0.0, last_us: 0.0,
        };
        let range = 900..1900;
        let partner: Vec<u32> = (0..1000).map(|k| 151 + k).collect();
        let imu_len = right.imu.len();
        let m = merged_meta(right.clone(), &left, &range, &partner, &clocks);
        assert_eq!((m.lenses.len(), m.lenses[0].fx, m.lenses[1].fx), (2, 7210.0, 7221.0));
        assert_eq!((m.track_order_reversed, m.single_lens), (Some(true), true));
        assert_eq!((m.frames_b.len(), m.frames.len()), (1000, 1000));
        assert_eq!(m.frames_b[0], right.frames[900]);
        assert_eq!(m.frames_b[999], right.frames[1899]);
        for k in [0usize, 500, 999] {
            assert_eq!(m.frames[k].t_us, m.frames_b[k].t_us, "row {k}");
        }
        assert_eq!((m.imu.len(), m.readout_ms, m.model.as_str()), (imu_len, Some(30.72), "Insta360 X6"));
    }

    #[test]
    fn clip_time_map() {
        let dt = 1001.0 / 30000.0;
        let clip0 = FrameTimeline::cfr(3000, dt);
        let clip1 = FrameTimeline::from_pts((0..3000).map(|i| (if i > 1200 { i + 3 } else { i }) as f64 * dt).collect(), dt);
        let range = 900..2900;
        let partner: Vec<u32> = (0..2000u32).map(|k| if k <= 200 { 1000 + k } else { (1000 + k).saturating_sub(3).max(1200) }).collect();
        let timeline = FrameTimeline::cfr(2000, dt);
        let plan = PairPlan {
            paths: [PathBuf::from("r.mp4"), PathBuf::from("l.mp4")],
            clip_start_s: [900.0 * dt, 1000.0 * dt],
            clip_timelines: [clip0, clip1],
            range, partner, timeline,
            probe: VideoProbe { width: 4, height: 4, fps: 29.97, duration_sec: 0.0, hlg: false, video_bitrates_bps: [0; 2] },
            meta: Insta360Meta::default(),
            audio_slot: 1,
            sync: PairSync { a_range: 0..0, b_frames: vec![], frame_offset: 0, held: 0, phase_median_us: 0.0, phase_spread_us: 0.0, frame_time_us: 0.0, out_of_step: 0 },
            clocks: ClockFit { map: ClockMap { origin_us: 0.0, offset_us: 0.0, skew: 0.0 }, common: 0, used: 0, residual_rms_us: 0.0, first_us: 0.0, last_us: 0.0 },
        };
        assert_eq!((plan.clip_frame(0, 0), plan.clip_frame(1, 0), plan.clip_frame(0, 1999), plan.clip_frame(1, 5000)), (900, 1000, 2899, 2996));
        let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
        assert!(near(plan.clip_time(0, 0.0), 900.0 * dt) && near(plan.clip_time(1, 0.0), 1000.0 * dt));
        assert!(near(plan.clip_time(0, 300.0 * dt), 1200.0 * dt));
        assert!(near(plan.clip_time(1, 1000.0 * dt), 2000.0 * dt), "{}", plan.clip_time(1, 1000.0 * dt) / dt);
        assert!(near(plan.clip_time(0, -1.0), 900.0 * dt));
        assert_eq!(plan.audio_source(), (Path::new("l.mp4"), 1000.0 * dt));
    }

    #[test]
    fn plain_files() {
        let dir = std::env::temp_dir().join(format!("vr180-take-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("plain.mp4");
        std::fs::write(&clip, b"not a movie").unwrap();
        assert!(plan_for(&clip).is_none());
        assert!(plan_for(&dir.join("missing.mp4")).is_none());
        assert!(plan_for(&dir.join("list.ffconcat")).is_none());
        std::fs::write(vr180_fisheye::x6_pair::sidecar_path(&clip), "{\"schema\":\"x-take-clock/1\",\"take_id\":\"t\",\"role\":\"R\"}").unwrap();
        forget(&clip);
        assert!(plan_for(&clip).is_none());
        assert!(plan_for(&clip).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
