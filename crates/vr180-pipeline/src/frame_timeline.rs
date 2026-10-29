//! Per-frame presentation times of a clip's video track, read from the
//! container's own sample tables — the map from a decoded frame's `pts` to
//! its row in every per-frame table (stabilization poses, rolling-shutter
//! windows, exposure records, eye-exposure state).
//!
//! The cameras' per-frame metadata (DJI frame blocks, Insta360 exposure
//! stamps, Kandao PTS/EXP tables) has ONE entry per ENCODED frame, in frame
//! order. A constant-rate clip lets `round(pts / dt)` recover the row. A clip
//! with dropped frames does not: the Insta360 X6 (like any MP4 writer that
//! keeps real time) leaves holes in the video track's `stts` where the sensor
//! dropped frames (2×, 3× … 10× the frame period — the exposure stamps in the
//! trailer have the same holes at the same rows), and libavformat's
//! `avg_frame_rate` is then `frames / duration`, LOWER than the nominal rate
//! (measured 48.16, 47.40, 44.54 fps on 50 fps X6 clips). With that `dt`,
//! `round(pts / dt)` lands tens of rows early long before the first hole and
//! the stabilizer applies another frame's pose. Indexing by the sample table
//! is exact for both kinds of file: `row(pts)` is a binary search over the
//! frame times, the nominal rate is the dominant sample delta, and the gaps
//! say how many output slots an exported frame has to hold so the picture
//! stays on the audio's clock.
//!
//! Only the `moov` is read (box headers to find it, then the atom), so a
//! clip on a network share costs one seek — no packet is touched.

use std::path::{Path, PathBuf};

use crate::spherical_inject::find_atom;
use crate::{Error, Result};

/// Presentation times of every video frame, ascending, on the pipeline's
/// clip timeline (segment-rebased for a chapter chain), plus the nominal
/// frame period. `Default` is the empty timeline (= "unknown, assume a
/// constant rate").
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameTimeline {
    pts_s: Vec<f64>,
    dt_s: f64,
}

/// What the video track's sample tables say about its frame rate: the media
/// timescale and the dominant sample delta (in ticks), i.e. the nominal rate
/// `timescale / delta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NominalRate {
    pub timescale: u32,
    pub delta_ticks: u32,
    /// Frames in the track (sample count).
    pub frames: usize,
}

impl NominalRate {
    pub fn fps(&self) -> f32 {
        if self.delta_ticks == 0 { 0.0 } else { self.timescale as f32 / self.delta_ticks as f32 }
    }
}

impl FrameTimeline {
    /// Build from explicit frame times (seconds, ascending) and a frame period.
    pub fn from_pts(mut pts_s: Vec<f64>, dt_s: f64) -> Self {
        pts_s.retain(|p| p.is_finite());
        pts_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { pts_s, dt_s: if dt_s.is_finite() && dt_s > 0.0 { dt_s } else { 0.0 } }
    }

    /// A constant-rate timeline of `frames` frames at `dt_s` (what a file
    /// without holes reads as; used by sources that have no sample table).
    pub fn cfr(frames: usize, dt_s: f64) -> Self {
        Self::from_pts((0..frames).map(|i| i as f64 * dt_s).collect(), dt_s)
    }

    /// Read the largest video track of an MP4-family file (`.mp4`, `.mov`,
    /// `.osv`, `.insv`, `.360`): `stts` (+ `ctts`, + the edit list's start)
    /// → one presentation time per frame, in seconds from the track start.
    pub fn from_mp4(path: &Path) -> Result<Self> {
        let moov = read_moov(path)?;
        let (track, _) = largest_video_track(&moov)?;
        Ok(Self::from_pts(track.pts_s, track.rate.delta_ticks as f64 / track.rate.timescale as f64))
    }

    /// The nominal frame rate of the largest video track — the dominant
    /// sample delta — without building the per-frame list.
    pub fn nominal_rate(path: &Path) -> Result<NominalRate> {
        let moov = read_moov(path)?;
        Ok(largest_video_track(&moov)?.0.rate)
    }

    /// Timeline of a chapter chain: each segment's frames rebased onto the
    /// clip timeline with the same offsets the segmented iterators use (the
    /// VIDEO track's duration of every preceding segment, from the `moov`;
    /// the ffmpeg probe's duration when the `moov` can't be read). `None`
    /// when any segment has no readable sample table — callers then keep the
    /// constant-rate arithmetic.
    pub fn from_segments(segments: &[PathBuf]) -> Option<Self> {
        if segments.is_empty() {
            return None;
        }
        if segments.len() == 1 {
            return Self::from_mp4(&segments[0]).ok();
        }
        let mut pts = Vec::new();
        let mut dt = 0.0_f64;
        let mut offset = 0.0_f64;
        for (i, seg) in segments.iter().enumerate() {
            let tl = Self::from_mp4(seg).ok()?;
            if i == 0 {
                dt = tl.dt_s;
            } else if (tl.dt_s - dt).abs() > 1e-6 {
                tracing::warn!(
                    "frame timeline: segment {} has a different frame period ({:.4} vs {:.4} ms) — using the first",
                    seg.display(), tl.dt_s * 1e3, dt * 1e3
                );
            }
            pts.extend(tl.pts_s.iter().map(|p| p + offset));
            let dur = crate::decode::probe_video_duration_via_moov(seg).ok()
                .or_else(|| crate::decode::probe_video(seg).ok().map(|p| p.duration_sec))?;
            offset += dur;
        }
        Some(Self::from_pts(pts, dt))
    }

    pub fn len(&self) -> usize { self.pts_s.len() }
    pub fn is_empty(&self) -> bool { self.pts_s.is_empty() }
    /// Nominal frame period (seconds).
    pub fn dt_s(&self) -> f64 { self.dt_s }
    /// Nominal frame rate (the dominant sample delta), 0 when unknown.
    pub fn nominal_fps(&self) -> f64 { if self.dt_s > 0.0 { 1.0 / self.dt_s } else { 0.0 } }
    /// Presentation time of frame `row`.
    pub fn pts(&self, row: usize) -> Option<f64> { self.pts_s.get(row).copied() }
    /// Time the last frame ends (its pts + one period); 0 when empty.
    pub fn duration_s(&self) -> f64 {
        self.pts_s.last().map(|p| p + self.dt_s).unwrap_or(0.0)
    }

    /// The frame on screen at `pts_s`: the last frame whose time is at or
    /// before it (a quarter period of slack absorbs the float error between
    /// `ticks × time_base` and `ticks / timescale`). Inside a hole this is
    /// the frame before the hole — the one a player keeps showing. Before
    /// the first frame it is frame 0.
    pub fn row(&self, pts_s: f64) -> usize {
        let i = self.pts_s.partition_point(|&p| p <= pts_s + 0.25 * self.dt_s);
        i.saturating_sub(1)
    }

    /// The first frame at or after `t_s` (quarter-period slack), clamped to
    /// the last frame — a seek target's frame.
    pub fn row_at_or_after(&self, t_s: f64) -> usize {
        let i = self.pts_s.partition_point(|&p| p < t_s - 0.25 * self.dt_s);
        i.min(self.pts_s.len().saturating_sub(1))
    }

    /// Output slots frame `row` has to hold beyond its own before the next
    /// frame starts: `round(Δt / dt) − 1`, 0 for a regular step and for the
    /// last frame. A frame followed by a 3× hole holds 2 extra slots.
    pub fn hold_after(&self, row: usize) -> usize {
        match (self.pts_s.get(row), self.pts_s.get(row + 1)) {
            (Some(a), Some(b)) if self.dt_s > 0.0 => {
                let steps = ((b - a) / self.dt_s).round();
                if steps.is_finite() && steps >= 2.0 { steps as usize - 1 } else { 0 }
            }
            _ => 0,
        }
    }

    /// Rows followed by a hole (places where frames are missing).
    pub fn gaps(&self) -> usize {
        (0..self.pts_s.len()).filter(|&r| self.hold_after(r) > 0).count()
    }

    /// Frames missing from the track: the sum of every hole's extra slots.
    pub fn dropped_frames(&self) -> usize {
        (0..self.pts_s.len()).map(|r| self.hold_after(r)).sum()
    }

    /// Short description for logs.
    pub fn summary(&self) -> String {
        let dropped = self.dropped_frames();
        if dropped == 0 {
            format!("{} frames @ {:.3} fps, no holes", self.len(), self.nominal_fps())
        } else {
            format!(
                "{} frames @ {:.3} fps, {} dropped in {} holes (avg {:.3} fps)",
                self.len(), self.nominal_fps(), dropped, self.gaps(),
                if self.duration_s() > 0.0 { self.len() as f64 / self.duration_s() } else { 0.0 }
            )
        }
    }
}

/// The row of per-frame tables that goes with a decoded frame: the sample
/// table's answer when the clip has one, else the constant-rate guess
/// `round(pts / dt)` (and `fallback` — the loop counter — when the frame
/// carries no usable pts).
pub fn stab_row(timeline: Option<&FrameTimeline>, pts_s: f64, dt_s: f64, fallback: usize) -> usize {
    if !(pts_s.is_finite() && pts_s >= 0.0) {
        return fallback;
    }
    match timeline {
        Some(tl) if !tl.is_empty() => tl.row(pts_s),
        _ => if dt_s > 0.0 { (pts_s / dt_s).round() as usize } else { fallback },
    }
}

/// How many times an exported frame is written so the output stays
/// constant-rate on the audio's clock: `lead` extra copies BEFORE it (only
/// for the first frame written, when the trim-in fell inside a hole and the
/// first available frame comes later than the cut) and `hold` extra copies
/// AFTER it (the hole that follows it in the source). Both are 0 for a
/// constant-rate clip or an unknown timeline, and capped so the output never
/// exceeds `total_slots` frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FillPlan {
    pub lead: u64,
    pub hold: u64,
}

impl FillPlan {
    /// Frames this plan writes in all (the frame itself included).
    pub fn copies(&self) -> u64 { 1 + self.lead + self.hold }
}

pub fn export_fill_plan(
    timeline: Option<&FrameTimeline>,
    row: usize,
    pts_s: f64,
    t_in_s: f64,
    written: u64,
    total_slots: u64,
) -> FillPlan {
    let Some(tl) = timeline.filter(|t| !t.is_empty() && t.dt_s() > 0.0) else {
        return FillPlan::default();
    };
    let budget = total_slots.saturating_sub(written).saturating_sub(1);
    let mut lead = 0u64;
    if written == 0 && pts_s.is_finite() {
        let slots_late = ((pts_s - t_in_s) / tl.dt_s()).round();
        if slots_late.is_finite() && slots_late >= 1.0 {
            lead = (slots_late as u64).min(budget);
        }
    }
    let hold = (tl.hold_after(row) as u64).min(budget.saturating_sub(lead));
    FillPlan { lead, hold }
}

// ── MP4 sample-table reader ─────────────────────────────────────────────

struct VideoTrack {
    pts_s: Vec<f64>,
    rate: NominalRate,
    /// tkhd width × height (16.16 fixed → integer), for picking the real
    /// eyes over a thumbnail track.
    area: u64,
}

/// Locate and load the `moov` atom reading box headers only (seeks past a
/// multi-GB `mdat`).
fn read_moov(path: &Path) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let fsize = f.metadata()?.len();
    let mut pos: u64 = 0;
    while pos + 8 <= fsize {
        let mut hdr = [0u8; 16];
        f.seek(SeekFrom::Start(pos))?;
        f.read_exact(&mut hdr[..8])?;
        let sz32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        let tag = [hdr[4], hdr[5], hdr[6], hdr[7]];
        let sz: u64 = if sz32 == 1 {
            f.read_exact(&mut hdr[8..16])?;
            u64::from_be_bytes(hdr[8..16].try_into().unwrap())
        } else if sz32 == 0 { fsize - pos } else { sz32 as u64 };
        if sz < 8 { break; }
        if &tag == b"moov" {
            if pos + sz > fsize || sz > 256 << 20 {
                return Err(Error::Ffmpeg("mp4: bad moov size".into()));
            }
            let mut moov = vec![0u8; sz as usize];
            f.seek(SeekFrom::Start(pos))?;
            f.read_exact(&mut moov)?;
            return Ok(moov);
        }
        pos = pos.checked_add(sz).ok_or_else(|| Error::Ffmpeg("mp4: box overflow".into()))?;
    }
    Err(Error::Ffmpeg("mp4: no moov".into()))
}

fn be32(b: &[u8], o: usize) -> Option<u32> { b.get(o..o + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap())) }
fn be64(b: &[u8], o: usize) -> Option<u64> { b.get(o..o + 8).map(|s| u64::from_be_bytes(s.try_into().unwrap())) }

/// Walk every `trak`, parse the video ones, return the one with the largest
/// picture (ties → the one with more frames) and the number of video tracks.
fn largest_video_track(moov: &[u8]) -> Result<(VideoTrack, usize)> {
    let (mv0, mvsz) = find_atom(moov, 0, moov.len(), b"moov")
        .ok_or_else(|| Error::Ffmpeg("mp4: moov header".into()))?;
    let mut best: Option<VideoTrack> = None;
    let mut n_video = 0usize;
    let mut tpos = mv0 + 8;
    let tend = mv0 + mvsz;
    while let Some((t0, tsz)) = find_atom(moov, tpos, tend, b"trak") {
        if let Some(track) = parse_video_track(moov, t0 + 8, t0 + tsz) {
            n_video += 1;
            let better = match &best {
                None => true,
                Some(b) => track.area > b.area || (track.area == b.area && track.pts_s.len() > b.pts_s.len()),
            };
            if better { best = Some(track); }
        }
        tpos = t0 + tsz;
    }
    best.map(|t| (t, n_video)).ok_or_else(|| Error::Ffmpeg("mp4: no video track with a sample table".into()))
}

/// One `trak`: `None` unless it is a video track with `stts` samples.
fn parse_video_track(moov: &[u8], start: usize, end: usize) -> Option<VideoTrack> {
    let (md0, mdsz) = find_atom(moov, start, end, b"mdia")?;
    let (mdia_s, mdia_e) = (md0 + 8, md0 + mdsz);
    // hdlr: version/flags(4) pre_defined(4) handler_type(4).
    let (h0, hsz) = find_atom(moov, mdia_s, mdia_e, b"hdlr")?;
    if hsz < 20 || moov.get(h0 + 16..h0 + 20)? != b"vide" {
        return None;
    }
    // mdhd: v0 creation(4) modification(4) timescale(4) duration(4);
    //       v1 creation(8) modification(8) timescale(4) duration(8).
    let (mh0, _) = find_atom(moov, mdia_s, mdia_e, b"mdhd")?;
    let c = mh0 + 8;
    let timescale = if *moov.get(c)? == 1 { be32(moov, c + 20)? } else { be32(moov, c + 12)? };
    if timescale == 0 { return None; }

    // tkhd: width/height (16.16) at v0 +76/+80, v1 +88/+92 from the content.
    let area = find_atom(moov, start, end, b"tkhd").and_then(|(k0, _)| {
        let c = k0 + 8;
        let (wo, ho) = if *moov.get(c)? == 1 { (88, 92) } else { (76, 80) };
        Some((be32(moov, c + wo)? >> 16) as u64 * (be32(moov, c + ho)? >> 16) as u64)
    }).unwrap_or(0);

    let (mi0, misz) = find_atom(moov, mdia_s, mdia_e, b"minf")?;
    let (sb0, sbsz) = find_atom(moov, mi0 + 8, mi0 + misz, b"stbl")?;
    let (stbl_s, stbl_e) = (sb0 + 8, sb0 + sbsz);

    // stts: version/flags(4) entry_count(4) then (sample_count, sample_delta)*.
    let (st0, stsz) = find_atom(moov, stbl_s, stbl_e, b"stts")?;
    let n = be32(moov, st0 + 12)? as usize;
    if n == 0 || st0 + 16 + n * 8 > st0 + stsz { return None; }
    let mut dts: Vec<u64> = Vec::new();
    let mut hist: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut t: u64 = 0;
    for i in 0..n {
        let count = be32(moov, st0 + 16 + i * 8)?;
        let delta = be32(moov, st0 + 20 + i * 8)?;
        *hist.entry(delta).or_insert(0) += count as u64;
        dts.reserve(count as usize);
        for _ in 0..count {
            dts.push(t);
            t += delta as u64;
        }
    }
    if dts.is_empty() { return None; }
    // Dominant delta = the nominal frame period (ties → the shorter).
    let (delta_ticks, _) = hist.iter()
        .map(|(&d, &c)| (d, c))
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
    if delta_ticks == 0 { return None; }

    // ctts (optional): version/flags(4) entry_count(4) then (count, offset)*;
    // the offset is signed in practice (v1 by spec).
    let mut cts: Vec<i64> = dts.iter().map(|&d| d as i64).collect();
    if let Some((ct0, ctsz)) = find_atom(moov, stbl_s, stbl_e, b"ctts") {
        let m = be32(moov, ct0 + 12)? as usize;
        if ct0 + 16 + m * 8 <= ct0 + ctsz {
            let mut k = 0usize;
            'outer: for i in 0..m {
                let count = be32(moov, ct0 + 16 + i * 8)? as usize;
                let off = be32(moov, ct0 + 20 + i * 8)? as i32 as i64;
                for _ in 0..count {
                    if k >= cts.len() { break 'outer; }
                    cts[k] += off;
                    k += 1;
                }
            }
        }
    }

    // Edit list: a leading empty edit (media_time −1) delays the track; the
    // first real edit's media_time is where presentation starts. Both in
    // the pattern libavformat applies — camera files have one edit at 0.
    let mut shift_ticks: i64 = 0;
    if let Some((ed0, edsz)) = find_atom(moov, start, end, b"edts") {
        if let Some((el0, elsz)) = find_atom(moov, ed0 + 8, ed0 + edsz, b"elst") {
            let ver = *moov.get(el0 + 8)?;
            let cnt = be32(moov, el0 + 12)? as usize;
            let mut o = el0 + 16;
            let mvhd_ts = find_atom(moov, 0, moov.len(), b"moov")
                .and_then(|(m0, msz)| find_atom(moov, m0 + 8, m0 + msz, b"mvhd"))
                .and_then(|(v0, _)| {
                    let c = v0 + 8;
                    if *moov.get(c)? == 1 { be32(moov, c + 20) } else { be32(moov, c + 12) }
                })
                .filter(|&ts| ts > 0);
            for _ in 0..cnt {
                if o + if ver == 1 { 20 } else { 12 } > el0 + elsz { break; }
                let (seg_dur, media_time) = if ver == 1 {
                    (be64(moov, o)? as i64, be64(moov, o + 8)? as i64)
                } else {
                    (be32(moov, o)? as i64, be32(moov, o + 4)? as i32 as i64)
                };
                o += if ver == 1 { 20 } else { 12 };
                if media_time < 0 {
                    // Empty edit: movie-timescale delay before the track starts.
                    if let Some(mts) = mvhd_ts {
                        shift_ticks += (seg_dur as f64 * timescale as f64 / mts as f64).round() as i64;
                    }
                    continue;
                }
                shift_ticks -= media_time;
                break;
            }
        }
    }

    let ts = timescale as f64;
    let pts_s: Vec<f64> = cts.iter()
        .map(|&c| (c + shift_ticks) as f64 / ts)
        .filter(|p| *p >= -(delta_ticks as f64) / ts)
        .collect();
    Some(VideoTrack {
        rate: NominalRate { timescale, delta_ticks, frames: pts_s.len() },
        pts_s,
        area,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gappy() -> FrameTimeline {
        // 50 fps: frames 0..5 regular, a 3× hole after frame 5, then regular.
        let mut pts: Vec<f64> = (0..6).map(|i| i as f64 * 0.02).collect();
        let mut t = 0.10 + 0.06;
        for _ in 0..5 { pts.push(t); t += 0.02; }
        FrameTimeline::from_pts(pts, 0.02)
    }

    #[test]
    fn rows_follow_the_sample_table() {
        let tl = gappy();
        assert_eq!(tl.len(), 11);
        assert_eq!(tl.row(0.0), 0);
        assert_eq!(tl.row(0.1), 5);
        // Inside the hole: the frame before it.
        assert_eq!(tl.row(0.12), 5);
        assert_eq!(tl.row(0.14), 5);
        // First frame after the hole is row 6, not round(0.16/0.02)=8.
        assert_eq!(tl.row(0.16), 6);
        assert_eq!(tl.row(0.16 + 1e-9), 6);
        assert_eq!(tl.row(0.24), 10);
        assert_eq!(tl.row(9.0), 10);
        assert_eq!(tl.row(-1.0), 0);
        assert_eq!(tl.row_at_or_after(0.12), 6);
        assert_eq!(tl.row_at_or_after(0.1), 5);
        assert_eq!(tl.hold_after(5), 2);
        assert_eq!(tl.hold_after(4), 0);
        assert_eq!(tl.hold_after(10), 0);
        assert_eq!(tl.dropped_frames(), 2);
        assert_eq!(tl.gaps(), 1);
        assert!((tl.nominal_fps() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn stab_row_falls_back_to_constant_rate() {
        let tl = gappy();
        assert_eq!(stab_row(Some(&tl), 0.16, 0.02, 99), 6);
        assert_eq!(stab_row(None, 0.16, 0.02, 99), 8);
        assert_eq!(stab_row(Some(&tl), f64::NAN, 0.02, 99), 99);
        assert_eq!(stab_row(Some(&FrameTimeline::default()), 0.16, 0.02, 99), 8);
    }

    #[test]
    fn fill_plan_holds_through_holes_and_respects_the_budget() {
        let tl = gappy();
        // Regular frame: nothing extra.
        assert_eq!(export_fill_plan(Some(&tl), 4, 0.08, 0.0, 4, 1000), FillPlan::default());
        // The frame before the hole holds two more slots.
        assert_eq!(export_fill_plan(Some(&tl), 5, 0.10, 0.0, 5, 1000), FillPlan { lead: 0, hold: 2 });
        // Budget caps the hold (only one slot left after this frame).
        assert_eq!(export_fill_plan(Some(&tl), 5, 0.10, 0.0, 5, 7), FillPlan { lead: 0, hold: 1 });
        assert_eq!(export_fill_plan(Some(&tl), 5, 0.10, 0.0, 5, 6), FillPlan { lead: 0, hold: 0 });
        // Trim-in inside the hole: the first frame written is 0.16 for a cut
        // at 0.12 → it leads by two slots.
        assert_eq!(export_fill_plan(Some(&tl), 6, 0.16, 0.12, 0, 1000), FillPlan { lead: 2, hold: 0 });
        // A later frame never leads.
        assert_eq!(export_fill_plan(Some(&tl), 6, 0.16, 0.12, 3, 1000), FillPlan::default());
        // No timeline: never anything extra.
        assert_eq!(export_fill_plan(None, 5, 0.10, 0.0, 5, 1000), FillPlan::default());
        assert_eq!(FillPlan { lead: 2, hold: 1 }.copies(), 4);
    }

    #[test]
    fn cfr_timeline_matches_round_division() {
        let tl = FrameTimeline::cfr(100, 1001.0 / 30000.0);
        for i in 0..100 {
            let p = i as f64 * 1001.0 / 30000.0;
            assert_eq!(tl.row(p), i);
            assert_eq!(tl.row(p), (p / tl.dt_s()).round() as usize);
            assert_eq!(tl.hold_after(i), 0);
        }
        assert_eq!(tl.dropped_frames(), 0);
    }

    /// A synthetic moov: one video trak (3 frames at 20 ms, then a 3× hole,
    /// then 2 frames) plus a smaller "thumbnail" video trak with 1 frame.
    fn synth_moov() -> Vec<u8> {
        fn atom(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut v = Vec::with_capacity(8 + body.len());
            v.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
            v.extend_from_slice(tag);
            v.extend_from_slice(body);
            v
        }
        fn trak(w: u32, h: u32, timescale: u32, stts: &[(u32, u32)], elst_media_time: Option<i32>) -> Vec<u8> {
            let mut tkhd = vec![0u8; 84];
            tkhd[76..80].copy_from_slice(&(w << 16).to_be_bytes());
            tkhd[80..84].copy_from_slice(&(h << 16).to_be_bytes());
            let mut mdhd = vec![0u8; 24];
            mdhd[12..16].copy_from_slice(&timescale.to_be_bytes());
            let mut hdlr = vec![0u8; 24];
            hdlr[8..12].copy_from_slice(b"vide");
            let mut stts_b = Vec::new();
            stts_b.extend_from_slice(&[0, 0, 0, 0]);
            stts_b.extend_from_slice(&(stts.len() as u32).to_be_bytes());
            for (c, d) in stts {
                stts_b.extend_from_slice(&c.to_be_bytes());
                stts_b.extend_from_slice(&d.to_be_bytes());
            }
            let stbl = atom(b"stbl", &atom(b"stts", &stts_b));
            let minf = atom(b"minf", &stbl);
            let mut mdia = Vec::new();
            mdia.extend(atom(b"mdhd", &mdhd));
            mdia.extend(atom(b"hdlr", &hdlr));
            mdia.extend(minf);
            let mut t = Vec::new();
            t.extend(atom(b"tkhd", &tkhd));
            if let Some(mt) = elst_media_time {
                let mut e = Vec::new();
                e.extend_from_slice(&[0, 0, 0, 0]);
                e.extend_from_slice(&1u32.to_be_bytes());
                e.extend_from_slice(&1000u32.to_be_bytes());
                e.extend_from_slice(&mt.to_be_bytes());
                e.extend_from_slice(&0x0001_0000u32.to_be_bytes());
                t.extend(atom(b"edts", &atom(b"elst", &e)));
            }
            t.extend(atom(b"mdia", &mdia));
            atom(b"trak", &t)
        }
        let mut mvhd = vec![0u8; 100];
        mvhd[12..16].copy_from_slice(&1000u32.to_be_bytes());
        let mut moov = Vec::new();
        moov.extend(atom(b"mvhd", &mvhd));
        moov.extend(trak(688, 344, 5000, &[(1, 100)], None));
        moov.extend(trak(3840, 3840, 5000, &[(2, 100), (1, 300), (2, 100)], Some(0)));
        atom(b"moov", &moov)
    }

    #[test]
    fn parses_a_synthetic_moov() {
        let moov = synth_moov();
        let (track, n_video) = largest_video_track(&moov).unwrap();
        assert_eq!(n_video, 2);
        assert_eq!(track.rate, NominalRate { timescale: 5000, delta_ticks: 100, frames: 5 });
        assert!((track.rate.fps() - 50.0).abs() < 1e-6);
        let tl = FrameTimeline::from_pts(track.pts_s, 0.02);
        assert_eq!(tl.len(), 5);
        assert_eq!(tl.pts(2), Some(0.04));
        assert_eq!(tl.pts(3), Some(0.10));
        assert_eq!(tl.hold_after(2), 2);
        assert_eq!(tl.dropped_frames(), 2);
        assert_eq!(tl.row(0.10), 3);
        assert_eq!(tl.row(0.06), 2);
    }

    #[test]
    fn reads_a_file_with_a_moov() {
        let moov = synth_moov();
        let mut file = Vec::new();
        file.extend_from_slice(&[0, 0, 0, 16]);
        file.extend_from_slice(b"ftyp");
        file.extend_from_slice(b"isom\0\0\0\0");
        file.extend_from_slice(&[0, 0, 0, 8]);
        file.extend_from_slice(b"mdat");
        file.extend(moov);
        let dir = std::env::temp_dir().join(format!("vr180_tl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("synth.mp4");
        std::fs::write(&p, &file).unwrap();
        let tl = FrameTimeline::from_mp4(&p).unwrap();
        assert_eq!(tl.len(), 5);
        assert_eq!(tl.dropped_frames(), 2);
        let rate = FrameTimeline::nominal_rate(&p).unwrap();
        assert_eq!(rate.delta_ticks, 100);
        let chain = FrameTimeline::from_segments(&[p.clone()]).unwrap();
        assert_eq!(chain, tl);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
