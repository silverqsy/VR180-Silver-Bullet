//! ffmpeg-next decode wrapper.
//!
//! Phase 0.2 deliverable: `extract_gpmf_stream` — pull all packets
//! from the .360 file's GoPro metadata data-stream into a single
//! byte buffer. This replaces the Python pipeline's
//! `ffmpeg -i <path> -map 0:3 -c copy -f rawvideo -` subprocess
//! call: in-process libav, no fork, no stdout pipe.
//!
//! Phase 0.4 — software video decode of the two HEVC streams (s0 + s4).
//! Phase 0.6 — hardware decode (VideoToolbox / NVDEC) with
//! IOSurface ↔ Metal or CUDA ↔ Vulkan interop into wgpu.

use crate::{Error, Result};
use crate::spherical_inject::find_atom;
use std::path::Path;

/// Initialize libav once per process (idempotent — `ffmpeg::init()`
/// guards itself internally). Call before the first ffmpeg-next op.
pub fn init() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `init()` returns Result but the only failure mode is "already
        // initialized," which we want to ignore.
        let _ = ffmpeg_next::init();
        // Quiet libav's per-frame stderr chatter unless the user opts in
        // via RUST_LOG / FFREPORT.
        ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    });
}

/// FourCC `'gpmd'` as a little-endian `u32` — the codec tag GoPro
/// uses for the GPMF telemetry data stream in `.360` / `.mp4` files.
/// Stored as bytes `g`,`p`,`m`,`d` on disk; the libav `AVCodecParameters::codec_tag`
/// field is a u32 in host (LE) byte order on Apple Silicon / x86.
const GPMD_TAG: u32 = u32::from_le_bytes(*b"gpmd");

/// Pull the entire GPMF (GoPro Metadata Format) data stream from a
/// .360 file into memory. Equivalent to the Python pipeline's
/// `ffmpeg -i <path> -map 0:3 -c copy -f rawvideo -` subprocess.
///
/// Stream selection: we look specifically for the data stream whose
/// codec tag is `gpmd` (GoPro Metadata). A `.360` file typically
/// has two data streams — index 2 is `tmcd` (timecode, ~4 bytes total)
/// and index 3 is `gpmd` (GPMF telemetry, hundreds of KB). Picking
/// "best Data" returns the timecode, which is wrong.
pub fn extract_gpmf_stream(path: &Path) -> Result<Vec<u8>> {
    init();

    // Fast path: read ONLY the `gpmd` track's samples by parsing the MP4
    // sample table, instead of demuxing the whole file. GoPro `.360`
    // chapters are ~11 GB each but carry only tens of MB of telemetry.
    // Demuxing reads the ENTIRE file — the mov demuxer reads every video
    // sample's bytes even with AVDISCARD_ALL set, since the discard check
    // happens only after the packet is read. On load we do this once for
    // the info panel and once per segment for the stab build, so a
    // multi-segment clip was tens of seconds of pure I/O before the first
    // frame. The sample-table read touches ~moov + the gpmd samples (a few
    // tens of MB) — ~0.05 s. Fall back to the demuxer on any parse miss so
    // an unusual container still works (just slower).
    match extract_gpmf_via_sample_table(path) {
        Ok(blob) if !blob.is_empty() => return Ok(blob),
        Ok(_) => tracing::debug!("gpmf: sample-table read empty for {path:?}; demuxing"),
        Err(e) => tracing::debug!("gpmf: sample-table read failed for {path:?}: {e}; demuxing"),
    }
    extract_gpmf_via_demux(path)
}

/// Original full-demux GPMF extraction — reads every packet and keeps the
/// `gpmd` ones. Correct but reads the whole file; used only as a fallback
/// when [`extract_gpmf_via_sample_table`] can't parse the container.
fn extract_gpmf_via_demux(path: &Path) -> Result<Vec<u8>> {
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = find_gpmd_stream(&ictx)
        .ok_or_else(|| Error::Ffmpeg(
            format!("no `gpmd` data stream in {path:?}")
        ))?;

    // Pre-size: GPMF telemetry is roughly 5-15 KB/s of footage. For a
    // 30-second clip that's ~300 KB; for an hour ~50 MB. Either way
    // it's fine to materialize in memory.
    let mut out = Vec::with_capacity(512 * 1024);
    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_idx {
            if let Some(data) = packet.data() {
                out.extend_from_slice(data);
            }
        }
    }

    if out.is_empty() {
        return Err(Error::Ffmpeg(format!(
            "GPMF data stream {} of {path:?} was empty", stream_idx
        )));
    }
    Ok(out)
}

/// Read only the `gpmd` telemetry track by walking the ISO-BMFF sample
/// table: locate `moov` (cheap — seeks past `mdat`), find the `trak` whose
/// `stsd` sample-entry format is `gpmd`, then concatenate that track's
/// samples chunk-by-chunk via direct file reads. The concatenation of the
/// samples is byte-identical to demuxing the `gpmd` packets in order.
fn extract_gpmf_via_sample_table(path: &Path) -> Result<Vec<u8>> {
    extract_data_track_via_sample_table(path, b"gpmd")
}

/// Read ONLY a data track's samples (GoPro `gpmd` telemetry / DJI `djmd`
/// metadata) by parsing the MP4 sample table — locate `moov` (seeking past
/// `mdat`), find EVERY `trak` whose `stsd` sample-entry format is `tag`,
/// concat each one's samples chunk-by-chunk, and return the LARGEST blob.
/// (DJI writes two `djmd` tracks; the big one carries the per-frame
/// protobuf — `extract_dji_meta_stream` picks the largest the same way.)
/// The concatenation is byte-identical to demuxing that track's packets.
fn extract_data_track_via_sample_table(path: &Path, tag: &[u8; 4]) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path)?;
    let fsize = f.metadata()?.len();

    // 1. Top-level scan for `moov` — reads box headers only, seeking past
    //    the multi-GB `mdat` without reading it.
    let mut moov_off: Option<u64> = None;
    let mut moov_sz: u64 = 0;
    let mut pos: u64 = 0;
    while pos + 8 <= fsize {
        let mut hdr = [0u8; 16];
        f.seek(SeekFrom::Start(pos))?;
        f.read_exact(&mut hdr[..8])?;
        let sz32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        let btag = [hdr[4], hdr[5], hdr[6], hdr[7]];
        let sz: u64 = if sz32 == 1 {
            f.read_exact(&mut hdr[8..16])?;
            u64::from_be_bytes(hdr[8..16].try_into().unwrap())
        } else if sz32 == 0 {
            fsize - pos
        } else {
            sz32 as u64
        };
        if sz < 8 { break; }
        if &btag == b"moov" { moov_off = Some(pos); moov_sz = sz; break; }
        pos = pos.checked_add(sz).ok_or_else(|| Error::Ffmpeg("mp4: box overflow".into()))?;
    }
    let moov_off = moov_off.ok_or_else(|| Error::Ffmpeg("mp4: no moov".into()))?;
    if moov_sz < 8 || moov_off + moov_sz > fsize {
        return Err(Error::Ffmpeg("mp4: bad moov size".into()));
    }

    // 2. Read moov into memory (small relative to the file).
    let mut moov = vec![0u8; moov_sz as usize];
    f.seek(SeekFrom::Start(moov_off))?;
    f.read_exact(&mut moov)?;
    let (mv0, mvsz) = find_atom(&moov, 0, moov.len(), b"moov")
        .ok_or_else(|| Error::Ffmpeg("mp4: moov header".into()))?;

    // 3. Read every matching track; keep the largest concatenation.
    let stbls = find_data_track_stbls(&moov, mv0, mvsz, tag);
    if stbls.is_empty() {
        return Err(Error::Ffmpeg(format!(
            "mp4: no {} trak", std::str::from_utf8(tag).unwrap_or("?"))));
    }
    let mut best: Vec<u8> = Vec::new();
    for stbl in stbls {
        let blob = read_stbl_samples(&mut f, fsize, &moov, stbl)?;
        if blob.len() > best.len() { best = blob; }
    }
    if best.is_empty() {
        return Err(Error::Ffmpeg("mp4: data track empty".into()));
    }
    Ok(best)
}

/// Concatenate ONE track's samples via direct file reads, using its
/// `stsz` (sizes) + `stsc` (sample→chunk) + `stco`/`co64` (chunk offsets).
/// Samples within a chunk are contiguous, so one read per chunk = the
/// concatenated samples.
fn read_stbl_samples(
    f: &mut std::fs::File,
    fsize: u64,
    moov: &[u8],
    stbl: (usize, usize),
) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let be32 = |b: &[u8], o: usize| -> Result<u32> {
        b.get(o..o + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap()))
            .ok_or_else(|| Error::Ffmpeg("mp4: short read (u32)".into()))
    };
    let be64 = |b: &[u8], o: usize| -> Result<u64> {
        b.get(o..o + 8).map(|s| u64::from_be_bytes(s.try_into().unwrap()))
            .ok_or_else(|| Error::Ffmpeg("mp4: short read (u64)".into()))
    };
    let (sb0, sbsz) = stbl;

    // Sample sizes (stsz: constant or per-sample).
    let (zo, zsz) = find_atom(moov, sb0 + 8, sb0 + sbsz, b"stsz")
        .ok_or_else(|| Error::Ffmpeg("mp4: stsz".into()))?;
    let const_size = be32(moov, zo + 12)?;
    let sample_count = be32(moov, zo + 16)? as usize;
    let sizes: Vec<u32> = if const_size != 0 {
        vec![const_size; sample_count]
    } else {
        let base = zo + 20;
        if base + sample_count * 4 > zo + zsz {
            return Err(Error::Ffmpeg("mp4: stsz entries".into()));
        }
        (0..sample_count).map(|i| be32(moov, base + i * 4)).collect::<Result<_>>()?
    };

    // Chunk offsets (stco 32-bit or co64 64-bit).
    let chunk_offsets: Vec<u64> = if let Some((o, s)) = find_atom(moov, sb0 + 8, sb0 + sbsz, b"stco") {
        let n = be32(moov, o + 12)? as usize;
        if o + 16 + n * 4 > o + s { return Err(Error::Ffmpeg("mp4: stco entries".into())); }
        (0..n).map(|i| be32(moov, o + 16 + i * 4).map(|v| v as u64)).collect::<Result<_>>()?
    } else if let Some((o, s)) = find_atom(moov, sb0 + 8, sb0 + sbsz, b"co64") {
        let n = be32(moov, o + 12)? as usize;
        if o + 16 + n * 8 > o + s { return Err(Error::Ffmpeg("mp4: co64 entries".into())); }
        (0..n).map(|i| be64(moov, o + 16 + i * 8)).collect::<Result<_>>()?
    } else {
        return Err(Error::Ffmpeg("mp4: no stco/co64".into()));
    };

    // Sample-to-chunk runs (first_chunk 1-based, samples_per_chunk).
    let (co, csz) = find_atom(moov, sb0 + 8, sb0 + sbsz, b"stsc")
        .ok_or_else(|| Error::Ffmpeg("mp4: stsc".into()))?;
    let stsc_n = be32(moov, co + 12)? as usize;
    if co + 16 + stsc_n * 12 > co + csz {
        return Err(Error::Ffmpeg("mp4: stsc entries".into()));
    }
    let runs: Vec<(u32, u32)> = (0..stsc_n)
        .map(|i| Ok((be32(moov, co + 16 + i * 12)?, be32(moov, co + 16 + i * 12 + 4)?)))
        .collect::<Result<_>>()?;
    if runs.is_empty() {
        return Err(Error::Ffmpeg("mp4: empty stsc".into()));
    }

    let total: usize = sizes.iter().map(|&s| s as usize).sum();
    let mut out = Vec::with_capacity(total);
    let mut sample_cursor = 0usize;
    for (c, &chunk_off) in chunk_offsets.iter().enumerate() {
        let chunk_1 = (c + 1) as u32;
        let mut spc = 0u32;
        for &(fc, s) in &runs {
            if fc <= chunk_1 { spc = s; } else { break; }
        }
        let mut chunk_len: u64 = 0;
        for k in 0..spc as usize {
            match sizes.get(sample_cursor + k) {
                Some(&s) => chunk_len += s as u64,
                None => break,
            }
        }
        if chunk_len > 0 {
            if chunk_off + chunk_len > fsize {
                return Err(Error::Ffmpeg("mp4: chunk past EOF".into()));
            }
            let mut buf = vec![0u8; chunk_len as usize];
            f.seek(SeekFrom::Start(chunk_off))?;
            f.read_exact(&mut buf)?;
            out.extend_from_slice(&buf);
        }
        sample_cursor += spc as usize;
        if sample_cursor >= sizes.len() { break; }
    }
    Ok(out)
}

#[cfg(test)]
mod gpmf_fast_tests {
    use super::*;
    /// The sample-table read must be byte-identical to demuxing, and far
    /// faster. Skips when the reference clip isn't mounted.
    #[test]
    fn fast_matches_demux() {
        let p = std::path::Path::new("/Volumes/Silver/0407rider/GS010191.360");
        if !p.exists() { eprintln!("skip fast_matches_demux: {p:?} not present"); return; }
        super::init();
        let t0 = std::time::Instant::now();
        let fast = extract_gpmf_via_sample_table(p).expect("sample-table");
        let t_fast = t0.elapsed();
        let t1 = std::time::Instant::now();
        let slow = extract_gpmf_via_demux(p).expect("demux");
        let t_slow = t1.elapsed();
        eprintln!("gpmf fast={} bytes in {t_fast:?}  demux={} bytes in {t_slow:?}",
            fast.len(), slow.len());
        assert_eq!(fast.len(), slow.len(), "gpmf length differs");
        assert!(fast == slow, "gpmf bytes differ");
    }

    /// DJI `djmd` sample-table read must equal the demux (largest track),
    /// and be far faster. Skips when no reference `.osv` is mounted.
    #[test]
    fn djmd_fast_matches_demux() {
        let p = std::path::Path::new("/Volumes/Silver/060613GC/CAM_20260613172639_0038_D.OSV");
        if !p.exists() { eprintln!("skip djmd test: no reference .osv"); return; }
        super::init();
        let fast = extract_data_track_via_sample_table(p, b"djmd").expect("sample-table");
        let slow = extract_dji_meta_stream_demux(p).expect("demux");
        assert_eq!(fast.len(), slow.len(), "djmd length differs");
        assert!(fast == slow, "djmd bytes differ");
    }
}

/// `stbl` (off, size) of EVERY track whose `stsd` sample-entry format is
/// `tag`, by walking `trak` siblings inside `moov`. (Usually one for
/// `gpmd`; two for DJI `djmd` — the caller picks the largest.)
fn find_data_track_stbls(
    moov: &[u8], moov_off: usize, moov_sz: usize, tag: &[u8; 4],
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let end = moov_off + moov_sz;
    let mut pos = moov_off + 8;
    while pos + 8 <= end {
        let sz32 = match moov.get(pos..pos + 4) {
            Some(b) => u32::from_be_bytes(b.try_into().unwrap()),
            None => break,
        };
        let btag = &moov[pos + 4..pos + 8];
        let sz = if sz32 == 1 {
            match moov.get(pos + 8..pos + 16) {
                Some(b) => u64::from_be_bytes(b.try_into().unwrap()) as usize,
                None => break,
            }
        } else if sz32 == 0 {
            end - pos
        } else {
            sz32 as usize
        };
        if sz < 8 || pos + sz > end { break; }
        if btag == b"trak" {
            // trak → mdia → minf → stbl → stsd; check the first entry fourcc.
            if let Some(mdia) = find_atom(moov, pos + 8, pos + sz, b"mdia") {
                if let Some(minf) = find_atom(moov, mdia.0 + 8, mdia.0 + mdia.1, b"minf") {
                    if let Some(stbl) = find_atom(moov, minf.0 + 8, minf.0 + minf.1, b"stbl") {
                        if let Some(stsd) = find_atom(moov, stbl.0 + 8, stbl.0 + stbl.1, b"stsd") {
                            // stsd: 8 hdr + 4 ver/flags + 4 entry_count, then
                            // entry: 4 size + 4 format fourcc.
                            if moov.get(stsd.0 + 20..stsd.0 + 24) == Some(tag.as_slice()) {
                                out.push(stbl);
                            }
                        }
                    }
                }
            }
        }
        pos += sz;
    }
    out
}

/// Find the stream index whose codec tag is `gpmd`. Returns the first
/// match in stream-index order. Returns `None` if no such stream exists.
fn find_gpmd_stream(ictx: &ffmpeg_next::format::context::Input) -> Option<usize> {
    for stream in ictx.streams() {
        // SAFETY: `AVCodecParameters` is exposed as a raw pointer; we
        // only read POD fields (`codec_tag`).
        let tag = unsafe { (*stream.parameters().as_ptr()).codec_tag };
        if tag == GPMD_TAG {
            return Some(stream.index());
        }
    }
    None
}

/// FourCC `'djmd'` — the codec tag DJI uses for the protobuf
/// metadata stream in `.osv` files. Letters d, j, m, d.
const DJMD_TAG: u32 = u32::from_le_bytes(*b"djmd");

/// Pull the DJI Osmo metadata blob from an `.osv` file. Equivalent to
/// the Python pipeline's `ffmpeg -i <file> -map 0:3 -c copy -f data
/// pipe:1` at `vr180_gui.py:364-370`.
///
/// The container typically has two `djmd` tracks (per the Python
/// survey of the sample file): the "CAM meta" track at index 3
/// (~226 kb/s) carries the full per-frame protobuf, while a second
/// "Gyro" track at index 4 (~36 kb/s) carries something else we don't
/// use. We pick the **largest** `djmd` stream (in bytes-yielded
/// terms), not the first — this is robust against firmware variants
/// where the track order might differ.
pub fn extract_dji_meta_stream(path: &Path) -> Result<Vec<u8>> {
    init();
    // Fast path: read only the largest `djmd` track's samples (the per-frame
    // protobuf) via the MP4 sample table, instead of demuxing the WHOLE OSV
    // file — several GB on a long clip → tens of seconds, and it ran on the
    // UI thread inside `DetailCache::new` (the slow clip-switch). Falls back
    // to the demuxer on a parse miss. Same fix as `extract_gpmf_stream`.
    match extract_data_track_via_sample_table(path, b"djmd") {
        Ok(blob) if !blob.is_empty() => return Ok(blob),
        Ok(_) => tracing::debug!("djmd: sample-table read empty for {path:?}; demuxing"),
        Err(e) => tracing::debug!("djmd: sample-table read failed for {path:?}: {e}; demuxing"),
    }
    extract_dji_meta_stream_demux(path)
}

/// Original full-demux DJI metadata extraction — reads the whole file and
/// keeps the largest `djmd` track. Fallback for `extract_dji_meta_stream`.
fn extract_dji_meta_stream_demux(path: &Path) -> Result<Vec<u8>> {
    init();

    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    // Collect all djmd stream indices first.
    let djmd_indices: Vec<usize> = ictx
        .streams()
        .filter(|s| {
            // SAFETY: same POD-field read pattern as find_gpmd_stream.
            let tag = unsafe { (*s.parameters().as_ptr()).codec_tag };
            tag == DJMD_TAG
        })
        .map(|s| s.index())
        .collect();
    if djmd_indices.is_empty() {
        return Err(Error::Ffmpeg(format!(
            "no `djmd` data stream in {path:?}"
        )));
    }
    tracing::debug!(
        "djmd track indices in {}: {:?}",
        path.display(), djmd_indices
    );

    // Buffer one Vec<u8> per djmd stream — keeps the option open to
    // pick a different track later (e.g. by handler_name) without
    // re-walking the container.
    let mut buffers: std::collections::HashMap<usize, Vec<u8>> =
        djmd_indices.iter().map(|&i| (i, Vec::with_capacity(64 * 1024))).collect();

    for (stream, packet) in ictx.packets() {
        if let Some(buf) = buffers.get_mut(&stream.index()) {
            if let Some(data) = packet.data() {
                buf.extend_from_slice(data);
            }
        }
    }

    let (best_idx, best_buf) = buffers
        .into_iter()
        .max_by_key(|(_, buf)| buf.len())
        .ok_or_else(|| Error::Ffmpeg("no djmd packets read".into()))?;

    if best_buf.is_empty() {
        return Err(Error::Ffmpeg(format!(
            "djmd stream {best_idx} of {path:?} was empty"
        )));
    }
    tracing::info!(
        "djmd: picked stream {} ({} bytes) from {}",
        best_idx, best_buf.len(), path.display()
    );
    Ok(best_buf)
}

/// Inspect basic video info for a .360 (or any container): fps,
/// duration, and the dimensions of the largest video stream. Used by
/// `probe-gyro` to convert sample counts → seconds in the printout.
#[derive(Debug, Clone, Copy)]
pub struct VideoProbe {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub duration_sec: f64,
}

pub fn probe_video(path: &Path) -> Result<VideoProbe> {
    init();
    let ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
    // Pick the LARGEST video stream by pixel area rather than ffmpeg's
    // `av_find_best_stream` (`.best()`). DJI OSV containers carry a
    // small MJPEG preview/thumbnail stream (e.g. 688×344) that `best()`
    // selects over the full-resolution HEVC fisheye eyes — it honours
    // the thumbnail's DEFAULT disposition — yielding a bogus 688×344 /
    // 0 fps probe that aborts the load. The real eyes are always the
    // biggest streams, so max-area selection is robust across cameras
    // and firmware revisions (and matches this fn's doc comment).
    let stream = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .max_by_key(|s| {
            // SAFETY: reading POD width/height from AVCodecParameters.
            let p = unsafe { &*s.parameters().as_ptr() };
            (p.width as i64) * (p.height as i64)
        })
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?;
    let params = stream.parameters();
    let avg = stream.avg_frame_rate();
    let fps = if avg.denominator() != 0 {
        avg.numerator() as f32 / avg.denominator() as f32
    } else {
        0.0
    };
    let tb = stream.time_base();
    let duration_sec = if tb.denominator() != 0 && stream.duration() != ffmpeg_next::ffi::AV_NOPTS_VALUE {
        stream.duration() as f64 * tb.numerator() as f64 / tb.denominator() as f64
    } else {
        ictx.duration() as f64 / ffmpeg_next::ffi::AV_TIME_BASE as f64
    };
    // SAFETY: AVCodecParameters is exposed by ffmpeg-next via as_ptr;
    // we read width/height which are POD fields.
    let raw = unsafe { &*params.as_ptr() };
    Ok(VideoProbe {
        width: raw.width as u32,
        height: raw.height as u32,
        fps,
        duration_sec,
    })
}

/// One decoded frame from each of the two HEVC streams in a .360 file,
/// converted to packed RGB8 host memory. The dimensions are probed from
/// the streams themselves — no hardcoded `5952×1920`.
#[derive(Debug)]
pub struct StreamPair {
    /// Packed RGB8 bytes, `stream_h × stream_w × 3`, from the first
    /// HEVC stream (`s0` in the Python code).
    pub s0: Vec<u8>,
    /// Same shape, from the second HEVC stream (`s4`).
    pub s4: Vec<u8>,
    /// Dimensions probed from the streams. Caller asserts the two
    /// streams agree on size — they always do on real GoPro footage.
    pub dims: vr180_core::eac::Dims,
    /// Which decode path actually produced this frame (after `Auto`
    /// resolution + any silent fallback).
    pub decode_path: DecodePath,
}

/// User-facing hwaccel selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwDecode {
    /// Pick the platform default: VideoToolbox on macOS, software
    /// elsewhere. Silently falls back to software if hwaccel setup fails.
    Auto,
    /// Force software decode (CPU only).
    Software,
    /// Force VideoToolbox (macOS only). Returns an error if hwaccel
    /// setup fails — use [`HwDecode::Auto`] for graceful fallback.
    VideoToolbox,
}

/// Which decode path actually ran (after fallback resolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    Software,
    VideoToolbox,
}

impl std::fmt::Display for DecodePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodePath::Software => f.write_str("software"),
            DecodePath::VideoToolbox => f.write_str("VideoToolbox"),
        }
    }
}

/// Extract the first decoded video frame from each of the two HEVC
/// streams. Defaults to platform-best hwaccel via [`HwDecode::Auto`].
pub fn extract_first_stream_pair(path: &Path) -> Result<StreamPair> {
    extract_first_stream_pair_with(path, HwDecode::Auto)
}

/// Same as [`extract_first_stream_pair`] but with explicit hwaccel
/// control. On macOS, `Auto` and `VideoToolbox` both wire VT decode
/// via `av_hwdevice_ctx_create` + a custom `get_format` callback that
/// returns `AV_PIX_FMT_VIDEOTOOLBOX`. Decoded hwframes are downloaded
/// to host memory via `av_hwframe_transfer_data` (typically NV12 for
/// 8-bit, P010 for 10-bit), then swscale converts to RGB24.
///
/// **Phase 0.6 scope:** decode path is hardware-accelerated end-to-end,
/// but frames still go through host memory before reaching the GPU
/// compositor. The zero-copy IOSurface↔Metal interop fast path
/// (skipping `av_hwframe_transfer_data` + the wgpu upload) is deferred
/// to Phase 0.6.5 — needs ~700 lines of objc/IOSurface bridging that's
/// its own focused work.
pub fn extract_first_stream_pair_with(path: &Path, hw: HwDecode) -> Result<StreamPair> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    // Find both HEVC video streams. GoPro .360 always has exactly two
    // (s0 + s4 in Python notation).
    let video_indices: Vec<usize> = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .map(|s| s.index())
        .collect();
    if video_indices.len() < 2 {
        return Err(Error::Ffmpeg(format!(
            "expected 2 video streams in .360 file, found {}",
            video_indices.len()
        )));
    }

    // Set up a decoder per stream + optionally attach the VT hwaccel.
    // We hold decoders as Vec<_> so borrow rules play nicely with the
    // packet-read loop below.
    let mut decoders: Vec<ffmpeg_next::codec::decoder::Video> = Vec::with_capacity(2);
    let mut hw_active_per_stream = [false, false];
    for (i, &idx) in video_indices.iter().take(2).enumerate() {
        let stream = ictx.stream(idx).unwrap();
        let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

        // Attempt to attach VT hwaccel BEFORE turning the Context into a
        // decoder — once the decoder is built we'd need its raw codec
        // context pointer to install `hw_device_ctx` / `get_format`.
        #[cfg(target_os = "macos")]
        {
            let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
            if want_vt {
                if try_enable_videotoolbox_decode(&mut codec_ctx) {
                    hw_active_per_stream[i] = true;
                } else if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg(format!(
                        "stream {idx}: --hw-accel videotoolbox requested but \
                         av_hwdevice_ctx_create failed (no VT support on this system)"
                    )));
                }
                // else: Auto + setup failed → silently fall back to sw
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (i, hw); // mark used so cfg gating doesn't warn
            if matches!(hw, HwDecode::VideoToolbox) {
                return Err(Error::Ffmpeg(
                    "--hw-accel videotoolbox is macOS-only".into()
                ));
            }
        }

        let decoder = codec_ctx.decoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;
        decoders.push(decoder);
    }

    // Both streams must agree on dimensions on real GoPro footage.
    let (w0, h0) = (decoders[0].width(), decoders[0].height());
    let (w1, h1) = (decoders[1].width(), decoders[1].height());
    if (w0, h0) != (w1, h1) {
        return Err(Error::Ffmpeg(format!(
            "video streams disagree on dimensions: {w0}x{h0} vs {w1}x{h1}"
        )));
    }
    let dims = vr180_core::eac::Dims::new(w0, h0);
    if !dims.is_valid() {
        return Err(Error::Ffmpeg(format!(
            "stream width {w0} not a valid EAC layout (need (w-1920) % 4 == 0)"
        )));
    }

    // Lazy scaler per stream: built once we see the actual sw-frame
    // pixel format (which differs depending on hwaccel: NV12/P010 after
    // VT transfer, yuv420p/p010le from sw decode).
    let mut scalers: Vec<Option<ffmpeg_next::software::scaling::Context>> = vec![None, None];

    let mut frames: [Option<Vec<u8>>; 2] = [None, None];
    'outer: for (stream, packet) in ictx.packets() {
        let pos = match video_indices.iter().position(|&i| i == stream.index()) {
            Some(p) if frames[p].is_none() => p,
            _ => continue,
        };
        let dec = &mut decoders[pos];
        if dec.send_packet(&packet).is_err() { continue; }
        let mut decoded = ffmpeg_next::frame::Video::empty();
        while dec.receive_frame(&mut decoded).is_ok() {
            // Hardware frame? Download to host memory via av_hwframe_transfer_data.
            // sw_frame ends up NV12 (8-bit) or P010 (10-bit) — swscale handles both.
            let mut sw_storage = ffmpeg_next::frame::Video::empty();
            let frame_ref: &ffmpeg_next::frame::Video = if hw_active_per_stream[pos]
                && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
            {
                download_hw_frame(&decoded, &mut sw_storage)?;
                &sw_storage
            } else {
                &decoded
            };

            let scaler = match &mut scalers[pos] {
                Some(s) => s,
                None => {
                    let s = ffmpeg_next::software::scaling::Context::get(
                        frame_ref.format(),
                        frame_ref.width(),
                        frame_ref.height(),
                        ffmpeg_next::format::Pixel::RGB24,
                        frame_ref.width(),
                        frame_ref.height(),
                        ffmpeg_next::software::scaling::Flags::BILINEAR,
                    ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?;
                    scalers[pos] = Some(s);
                    scalers[pos].as_mut().unwrap()
                }
            };
            let mut rgb = ffmpeg_next::frame::Video::empty();
            scaler.run(frame_ref, &mut rgb)
                .map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
            // swscale row stride may exceed width*3 (SIMD padding).
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            let src = rgb.data(0);
            let src_stride = rgb.stride(0);
            let mut out = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                let row = &src[y * src_stride..y * src_stride + w * 3];
                out.extend_from_slice(row);
            }
            frames[pos] = Some(out);
            if frames.iter().all(|f| f.is_some()) {
                break 'outer;
            }
            break;
        }
    }

    let (Some(s0), Some(s4)) = (frames[0].take(), frames[1].take()) else {
        return Err(Error::Ffmpeg(
            "EOF reached before both streams produced a decoded frame".into()
        ));
    };
    let decode_path = if hw_active_per_stream.iter().all(|&a| a) {
        DecodePath::VideoToolbox
    } else {
        DecodePath::Software
    };
    Ok(StreamPair { s0, s4, dims, decode_path })
}

/// Phase 0.6.6 streaming zero-copy decoder. Yields one
/// `(s0_iosurface, s4_iosurface)` tuple per frame from a `.360` file,
/// **without** crossing host memory at any point — every IOSurface is
/// wrapped as a `wgpu::Texture` ready for the `nv12_to_eac_cross`
/// kernel.
///
/// **macOS-only.** On Windows / Linux this struct can't be built;
/// callers should use [`StreamPairIter`] (CPU path) instead.
#[cfg(target_os = "macos")]
pub struct ZeroCopyStreamPairIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: Vec<usize>,
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    dims: vr180_core::eac::Dims,
    frame_limit: u32,
    frames_yielded: u32,
    /// s0 stream time_base as seconds-per-tick (for frame pts → seconds).
    tb0_s: f64,
    /// Nominal frame duration of s0 (1 / avg_frame_rate).
    dt_s: f64,
    /// Precise-seek state: after `seek(t)` the container lands on the
    /// keyframe ≤ t (GoPro HEVC GOP is 1-2 s ≈ 30-60 frames). `next_pair`
    /// decodes-and-discards until the pair's pts reaches this target, so
    /// the first pair RETURNED is the exact requested frame. Without
    /// this, the caller's frame indexing (stab/RS lookup by
    /// `time_offset/dt + n`) is off by up to a GOP after every scrub —
    /// the stabilization fights the content and the preview shakes.
    skip_until_s: Option<f64>,
}

/// One zero-copy NV12 plane tuple — two `IOSurfacePlaneTexture`s
/// per stream (Y + UV). Hold onto the returned tuple while the kernel
/// dispatch is in flight; dropping it releases the IOSurface retains.
#[cfg(target_os = "macos")]
pub struct ZeroCopyStreamPair {
    pub s0_y:  crate::interop_macos::IOSurfacePlaneTexture,
    pub s0_uv: crate::interop_macos::IOSurfacePlaneTexture,
    pub s4_y:  crate::interop_macos::IOSurfacePlaneTexture,
    pub s4_uv: crate::interop_macos::IOSurfacePlaneTexture,
    pub dims:  vr180_core::eac::Dims,
}

#[cfg(target_os = "macos")]
impl ZeroCopyStreamPairIter {
    /// Open `path`, force VideoToolbox decode, validate two HEVC streams
    /// of matching dimensions, return an iterator that produces zero-
    /// copy `(s0, s4)` IOSurface texture pairs.
    pub fn new(path: &Path, frame_limit: u32) -> Result<Self> {
        init();
        let mut ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

        let video_indices: Vec<usize> = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .map(|s| s.index())
            .collect();
        if video_indices.len() < 2 {
            return Err(Error::Ffmpeg(format!(
                "expected 2 video streams in .360, found {}", video_indices.len()
            )));
        }

        let mut decoders = Vec::with_capacity(2);
        for &idx in video_indices.iter().take(2) {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            if !try_enable_videotoolbox_decode(&mut codec_ctx) {
                return Err(Error::Ffmpeg(format!(
                    "zero-copy path requires VideoToolbox hwaccel — VT setup failed on stream {idx}"
                )));
            }
            decoders.push(codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?);
        }
        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "streams disagree on dims: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let dims = vr180_core::eac::Dims::new(w0, h0);
        if !dims.is_valid() {
            return Err(Error::Ffmpeg(format!("stream width {w0} not a valid EAC layout")));
        }
        let s0 = ictx.stream(video_indices[0]).unwrap();
        let tb = s0.time_base();
        let tb0_s = tb.numerator() as f64 / tb.denominator().max(1) as f64;
        let fr = s0.avg_frame_rate();
        let dt_s = if fr.numerator() > 0 {
            fr.denominator() as f64 / fr.numerator() as f64
        } else {
            1.0 / 30.0
        };
        Ok(Self {
            ictx, video_indices, decoders, dims, frame_limit,
            frames_yielded: 0, tb0_s, dt_s, skip_until_s: None,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.dims }

    /// PRECISE seek: the next `next_pair` call returns the frame AT
    /// `target_s` (±½ frame). Internally:
    /// 1. `av_seek_frame` to the nearest keyframe ≤ target_s.
    /// 2. Flush both decoder contexts so old packets are dropped.
    /// 3. Arm `skip_until_s` — `next_pair` decodes-and-discards the
    ///    keyframe→target run-in (≤ 1 GOP, 1-2 s on GoPro HEVC) before
    ///    returning. Callers index stab/RS bundles by
    ///    `round(target/dt) + frames_since_seek`; without the run-in
    ///    discard that index is off by up to a GOP after every scrub
    ///    and the stabilization visibly fights the content.
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let target_s = target_s.max(0.0);
        let ts = (target_s * 1_000_000.0) as i64; // AV_TIME_BASE
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek to {target_s:.3}s: {e}")))?;
        // Flush both decoders so leftover packets / frames from the
        // previous position don't poison the next next_pair call.
        for d in &mut self.decoders {
            d.flush();
        }
        self.frames_yielded = 0;
        self.skip_until_s = Some(target_s);
        Ok(())
    }

    /// Pull the next zero-copy `(s0, s4)` IOSurface pair. Returns
    /// `Ok(None)` at EOF or frame limit. Each call may issue multiple
    /// packets to the decoder before both streams yield a frame.
    /// Decode the next (s0, s4) frame pair, honouring the precise-seek
    /// run-in discard. Shared by [`next_pair`](Self::next_pair) (texture
    /// wrapping) and [`next_p010_streams`](Self::next_p010_streams) (raw
    /// `CVPixelBuffer`s for the zero-copy EAC denoise path).
    fn decode_stream_frame_pair(
        &mut self,
    ) -> Result<Option<[ffmpeg_next::frame::Video; 2]>> {
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut frames: [Option<ffmpeg_next::frame::Video>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();

        // Fill-and-maybe-discard loop. Each iteration assembles one
        // (s0, s4) pair; if a precise seek is pending and the pair is
        // still in the keyframe→target run-in, both frames are dropped
        // (decode cost only — no IOSurface wrapping) and we assemble
        // the next pair.
        let (f0, f4) = 'fill: loop {
            // Drain any pre-buffered frames first.
            for pos in 0..2 {
                if frames[pos].is_some() { continue; }
                if self.decoders[pos].receive_frame(&mut decoded).is_ok() {
                    frames[pos] = Some(std::mem::replace(&mut decoded, ffmpeg_next::frame::Video::empty()));
                }
            }

            // Then pump packets.
            //
            // Invariant: every packet for a video stream we care about
            // gets forwarded to its decoder, even if we already have a
            // frame for that stream this iteration. Otherwise the dropped
            // packet's frame is lost forever, and that stream falls behind
            // the other one — pairing (s0_K, s4_K+1) etc. — which shows up
            // visually as EAC faces sourced from different temporal frames.
            //
            // We only need to `receive_frame` from streams we're still
            // missing; the dropped packets pile up in the decoder's queue
            // and feed the NEXT iteration's `drain pre-buffered` step.
            if frames.iter().any(|f| f.is_none()) {
                loop {
                    let (stream, packet) = match self.ictx.packets().next() {
                        Some(x) => x, None => break,
                    };
                    let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                        Some(p) => p,
                        None => continue,  // not one of our video streams (audio/data/etc.)
                    };
                    // Always feed the packet to its decoder. Send errors
                    // are non-fatal — the decoder might be flushing.
                    if self.decoders[pos].send_packet(&packet).is_err() { continue; }
                    // Only consume a frame from this decoder if we still
                    // need it this iteration. Extras stay queued for next call.
                    if frames[pos].is_none()
                        && self.decoders[pos].receive_frame(&mut decoded).is_ok()
                    {
                        frames[pos] = Some(std::mem::replace(
                            &mut decoded, ffmpeg_next::frame::Video::empty()));
                    }
                    if frames.iter().all(|f| f.is_some()) { break; }
                }
            }

            let (Some(f0), Some(f4)) = (frames[0].take(), frames[1].take()) else {
                return Ok(None);
            };

            // Precise-seek run-in: discard pairs before the target.
            if let Some(target) = self.skip_until_s {
                let t = f0.pts().unwrap_or(0) as f64 * self.tb0_s;
                if t < target - 0.5 * self.dt_s {
                    // Not there yet — drop this pair and assemble the next.
                    continue 'fill;
                }
                self.skip_until_s = None;
            }
            break 'fill (f0, f4);
        };
        self.frames_yielded += 1;
        Ok(Some([f0, f4]))
    }

    /// Raw VT-decoded P010 `CVPixelBuffer`s for both streams (s0, s4),
    /// retained, for the zero-copy EAC denoise path. The caller transfers
    /// them into the denoiser and may drop them after.
    #[allow(clippy::type_complexity)]
    pub fn next_p010_streams(
        &mut self,
    ) -> Result<Option<(
        crate::interop_macos::RetainedCVPixelBuffer,
        crate::interop_macos::RetainedCVPixelBuffer,
    )>> {
        use crate::interop_macos::extract_cvpixelbuffer_from_vt_frame;
        let [f0, f4] = match self.decode_stream_frame_pair()? {
            Some(x) => x,
            None => return Ok(None),
        };
        Ok(Some((
            extract_cvpixelbuffer_from_vt_frame(&f0)?,
            extract_cvpixelbuffer_from_vt_frame(&f4)?,
        )))
    }

    pub fn next_pair(&mut self, wgpu_device: &wgpu::Device) -> Result<Option<ZeroCopyStreamPair>> {
        use crate::interop_macos::{
            extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
            IOSurfaceNv12Descriptor,
        };
        let [f0, f4] = match self.decode_stream_frame_pair()? {
            Some(x) => x,
            None => return Ok(None),
        };

        // Both frames are VT-format; extract IOSurfaces and wrap planes.
        let surf0 = extract_iosurface_from_vt_frame(&f0)?;
        let desc0 = IOSurfaceNv12Descriptor::new(surf0)?;
        let s0_y_surf  = unsafe { crate::interop_macos::RetainedIOSurface::retain(desc0.surface.as_raw()) };
        let s0_uv_surf = unsafe { crate::interop_macos::RetainedIOSurface::retain(desc0.surface.as_raw()) };
        // GoPro Max records 10-bit HEVC → VideoToolbox produces P010
        // IOSurfaces (16-bit per channel, 10 bits used). Wrap as
        // R16Unorm + Rg16Unorm. The nv12_to_eac_cross shader knows how
        // to expand 10-bits-in-upper-10 to BT.709 limited-range.
        // 8-bit (NV12) input would use R8Unorm + Rg8Unorm + a different
        // YUV→RGB scaler — not yet implemented (no 8-bit GoPro footage
        // in the test set).
        let s0_y = wgpu_texture_from_iosurface_plane(
            wgpu_device, s0_y_surf, 0,
            metal::MTLPixelFormat::R16Unorm,  wgpu::TextureFormat::R16Unorm,
            desc0.width, desc0.height, "s0_y",
        )?;
        let s0_uv = wgpu_texture_from_iosurface_plane(
            wgpu_device, s0_uv_surf, 1,
            metal::MTLPixelFormat::RG16Unorm, wgpu::TextureFormat::Rg16Unorm,
            desc0.width / 2, desc0.height / 2, "s0_uv",
        )?;
        // Drop the descriptor's retain explicitly so we don't leak — the
        // two plane textures each hold their own retain via `retain()` above.
        drop(desc0);

        let surf4 = extract_iosurface_from_vt_frame(&f4)?;
        let desc4 = IOSurfaceNv12Descriptor::new(surf4)?;
        let s4_y_surf  = unsafe { crate::interop_macos::RetainedIOSurface::retain(desc4.surface.as_raw()) };
        let s4_uv_surf = unsafe { crate::interop_macos::RetainedIOSurface::retain(desc4.surface.as_raw()) };
        let s4_y = wgpu_texture_from_iosurface_plane(
            wgpu_device, s4_y_surf, 0,
            metal::MTLPixelFormat::R16Unorm,  wgpu::TextureFormat::R16Unorm,
            desc4.width, desc4.height, "s4_y",
        )?;
        let s4_uv = wgpu_texture_from_iosurface_plane(
            wgpu_device, s4_uv_surf, 1,
            metal::MTLPixelFormat::RG16Unorm, wgpu::TextureFormat::Rg16Unorm,
            desc4.width / 2, desc4.height / 2, "s4_uv",
        )?;
        drop(desc4);

        Ok(Some(ZeroCopyStreamPair { s0_y, s0_uv, s4_y, s4_uv, dims: self.dims }))
    }
}

/// Plays an ordered list of GoPro `.360` segments (GS01…, GS02…, …) as
/// one continuous zero-copy stream. Transparent to the decode loop:
/// `next_pair` crosses segment boundaries without resetting, and `seek`
/// maps a global clip time to the right segment + local offset. A
/// single-element `segments` is a pure passthrough — behaviour is
/// byte-identical to using `ZeroCopyStreamPairIter` directly.
#[cfg(target_os = "macos")]
pub struct SegmentedZeroCopyPairIter {
    segments: Vec<std::path::PathBuf>,
    /// Cumulative clip-time start of each segment (seconds).
    seg_start_s: Vec<f64>,
    total_dur_s: f64,
    cur_idx: usize,
    cur: ZeroCopyStreamPairIter,
}

#[cfg(target_os = "macos")]
impl SegmentedZeroCopyPairIter {
    pub fn new(segments: &[std::path::PathBuf], frame_limit: u32) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for seg in segments {
            seg_start_s.push(acc);
            acc += probe_video(seg).map(|p| p.duration_sec).unwrap_or(0.0);
        }
        let cur = ZeroCopyStreamPairIter::new(&segments[0], frame_limit)?;
        Ok(Self {
            segments: segments.to_vec(),
            seg_start_s, total_dur_s: acc, cur_idx: 0, cur,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.cur.dims() }

    /// Total clip duration across all segments (seconds).
    pub fn total_duration_s(&self) -> f64 { self.total_dur_s }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let t = target_s.clamp(0.0, self.total_dur_s);
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            self.cur = ZeroCopyStreamPairIter::new(&self.segments[idx], 0)?;
            self.cur_idx = idx;
        }
        self.cur.seek(t - self.seg_start_s[idx])
    }

    pub fn next_pair(&mut self, device: &wgpu::Device) -> Result<Option<ZeroCopyStreamPair>> {
        loop {
            if let Some(p) = self.cur.next_pair(device)? {
                return Ok(Some(p));
            }
            // Current segment exhausted — advance to the next, if any.
            if self.cur_idx + 1 >= self.segments.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            self.cur = ZeroCopyStreamPairIter::new(&self.segments[self.cur_idx], 0)?;
        }
    }

    /// Segment-chaining counterpart of
    /// [`ZeroCopyStreamPairIter::next_p010_streams`] — raw s0/s4 P010
    /// buffers for the zero-copy EAC denoise path.
    #[allow(clippy::type_complexity)]
    pub fn next_p010_streams(
        &mut self,
    ) -> Result<Option<(
        crate::interop_macos::RetainedCVPixelBuffer,
        crate::interop_macos::RetainedCVPixelBuffer,
    )>> {
        loop {
            if let Some(p) = self.cur.next_p010_streams()? {
                return Ok(Some(p));
            }
            if self.cur_idx + 1 >= self.segments.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            self.cur = ZeroCopyStreamPairIter::new(&self.segments[self.cur_idx], 0)?;
        }
    }
}

/// Repackage two denoised stream [`EncodePixelBufferP010`]s as a
/// [`ZeroCopyStreamPair`] (s0 + s4, each split into Y + UV plane textures).
#[cfg(target_os = "macos")]
fn eac_epb_to_pair(
    s0: crate::interop_macos::EncodePixelBufferP010,
    s4: crate::interop_macos::EncodePixelBufferP010,
    dims: vr180_core::eac::Dims,
) -> ZeroCopyStreamPair {
    use crate::interop_macos::{EncodePixelBufferP010, IOSurfacePlaneTexture};
    fn split(epb: EncodePixelBufferP010) -> (IOSurfacePlaneTexture, IOSurfacePlaneTexture) {
        let (w, h) = (epb.width, epb.height);
        let y = IOSurfacePlaneTexture {
            surface: epb.iosurface.clone(),
            texture: epb.y_tex,
            width: w,
            height: h,
            format: wgpu::TextureFormat::R16Unorm,
        };
        let uv = IOSurfacePlaneTexture {
            surface: epb.iosurface,
            texture: epb.uv_tex,
            width: w / 2,
            height: h / 2,
            format: wgpu::TextureFormat::Rg16Unorm,
        };
        (y, uv)
    }
    let (s0_y, s0_uv) = split(s0);
    let (s4_y, s4_uv) = split(s4);
    ZeroCopyStreamPair { s0_y, s0_uv, s4_y, s4_uv, dims }
}

/// Wraps [`SegmentedZeroCopyPairIter`] and denoises the s0/s4 P010 streams on
/// the GPU (one [`P010Denoiser`](crate::vt_denoise::P010Denoiser) each) BEFORE
/// EAC cross assembly — the GoPro `.360` analogue of
/// [`DenoisingZeroCopyIter`](crate::fisheye_decode::DenoisingZeroCopyIter),
/// matching the Python app which denoises the source streams. No CPU readback.
/// Yields denoised [`ZeroCopyStreamPair`]s; count/order preserved 1:1.
#[cfg(target_os = "macos")]
pub struct DenoisingZeroCopyEacIter {
    inner: SegmentedZeroCopyPairIter,
    s0: crate::vt_denoise::P010Denoiser,
    s4: crate::vt_denoise::P010Denoiser,
    dims: vr180_core::eac::Dims,
    strength: f32,
    ready: std::collections::VecDeque<ZeroCopyStreamPair>,
    inner_done: bool,
}

#[cfg(target_os = "macos")]
impl DenoisingZeroCopyEacIter {
    pub fn new(inner: SegmentedZeroCopyPairIter, strength: f32) -> Result<Self> {
        let dims = inner.dims();
        let s0 = crate::vt_denoise::P010Denoiser::new(dims.stream_w, dims.stream_h, strength)?;
        let s4 = crate::vt_denoise::P010Denoiser::new(dims.stream_w, dims.stream_h, strength)?;
        Ok(Self {
            inner,
            s0,
            s4,
            dims,
            strength,
            ready: std::collections::VecDeque::new(),
            inner_done: false,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims {
        self.dims
    }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        self.inner.seek(target_s)?;
        // Temporal window meaningless across a seek — rebuild the denoisers.
        self.s0 = crate::vt_denoise::P010Denoiser::new(self.dims.stream_w, self.dims.stream_h, self.strength)?;
        self.s4 = crate::vt_denoise::P010Denoiser::new(self.dims.stream_w, self.dims.stream_h, self.strength)?;
        self.ready.clear();
        self.inner_done = false;
        Ok(())
    }

    pub fn next_pair(&mut self, device: &wgpu::Device) -> Result<Option<ZeroCopyStreamPair>> {
        loop {
            if let Some(p) = self.ready.pop_front() {
                return Ok(Some(p));
            }
            if self.inner_done {
                return Ok(None);
            }
            match self.inner.next_p010_streams()? {
                Some((s0_src, s4_src)) => {
                    let s0_outs = self.s0.push(device, s0_src.as_raw())?;
                    let s4_outs = self.s4.push(device, s4_src.as_raw())?;
                    drop(s0_src);
                    drop(s4_src);
                    for (a, b) in s0_outs.into_iter().zip(s4_outs.into_iter()) {
                        self.ready.push_back(eac_epb_to_pair(a, b, self.dims));
                    }
                }
                None => {
                    self.inner_done = true;
                    let s0_outs = self.s0.finish(device)?;
                    let s4_outs = self.s4.finish(device)?;
                    for (a, b) in s0_outs.into_iter().zip(s4_outs.into_iter()) {
                        self.ready.push_back(eac_epb_to_pair(a, b, self.dims));
                    }
                }
            }
        }
    }
}

/// Either the raw EAC zero-copy decoder or the denoising wrapper — lets
/// `export_eac_zerocopy_vt` call `next_pair`/`seek`/`dims` uniformly.
#[cfg(target_os = "macos")]
pub enum ZcEacDecoder {
    Raw(SegmentedZeroCopyPairIter),
    Denoising(DenoisingZeroCopyEacIter),
}

#[cfg(target_os = "macos")]
impl ZcEacDecoder {
    pub fn dims(&self) -> vr180_core::eac::Dims {
        match self {
            Self::Raw(d) => d.dims(),
            Self::Denoising(d) => d.dims(),
        }
    }
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        match self {
            Self::Raw(d) => d.seek(target_s),
            Self::Denoising(d) => d.seek(target_s),
        }
    }
    pub fn next_pair(&mut self, device: &wgpu::Device) -> Result<Option<ZeroCopyStreamPair>> {
        match self {
            Self::Raw(d) => d.next_pair(device),
            Self::Denoising(d) => d.next_pair(device),
        }
    }
}

/// Decode the first video frame using VideoToolbox hwaccel and return
/// it **without** running `av_hwframe_transfer_data`. The frame's
/// `format == AV_PIX_FMT_VIDEOTOOLBOX` and `data[3]` is the live
/// CVPixelBuffer the IOSurface lives inside.
///
/// Used by Phase 0.6.5's IOSurface demo (`probe-iosurface`) and is the
/// single-frame analogue of `ZeroCopyStreamPairIter::next_pair` above.
#[cfg(target_os = "macos")]
pub fn decode_first_vt_frame(path: &Path) -> Result<ffmpeg_next::frame::Video> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .next()
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?
        .index();

    let stream = ictx.stream(stream_idx).unwrap();
    let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

    if !try_enable_videotoolbox_decode(&mut codec_ctx) {
        return Err(Error::Ffmpeg(
            "VideoToolbox hwaccel setup failed — IOSurface path needs VT".into()
        ));
    }
    let mut decoder = codec_ctx.decoder().video()
        .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx { continue; }
        if decoder.send_packet(&packet).is_err() { continue; }
        let mut decoded = ffmpeg_next::frame::Video::empty();
        if decoder.receive_frame(&mut decoded).is_ok() {
            if decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX {
                return Ok(decoded);
            } else {
                return Err(Error::Ffmpeg(format!(
                    "expected VIDEOTOOLBOX pixel format, decoder produced {:?}",
                    decoded.format()
                )));
            }
        }
    }
    Err(Error::Ffmpeg("EOF before first frame decoded".into()))
}

// ─── VideoToolbox hwaccel plumbing (macOS only) ────────────────────────

/// Wire VideoToolbox hardware decode onto a codec context.
///
/// Returns `true` if the hwaccel was successfully attached. The codec
/// context takes ownership of the `AVBufferRef` and will free it; we
/// deliberately don't hold an independent clone here since this
/// Phase 0.6 pipeline only decodes one frame per stream and exits.
#[cfg(target_os = "macos")]
/// Visibility-bumped to `pub(crate)` so the fisheye-source decoders
/// can opt into VT decode too. The function itself is no-op on
/// non-macOS.
#[cfg(target_os = "macos")]
pub(crate) fn try_enable_videotoolbox_decode(
    dec_ctx: &mut ffmpeg_next::codec::context::Context,
) -> bool {
    try_enable_videotoolbox_decode_inner(dec_ctx)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn try_enable_videotoolbox_decode(
    _dec_ctx: &mut ffmpeg_next::codec::context::Context,
) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn try_enable_videotoolbox_decode_inner(
    dec_ctx: &mut ffmpeg_next::codec::context::Context,
) -> bool {
    use ffmpeg_next::ffi::*;
    let mut hw_device: *mut AVBufferRef = std::ptr::null_mut();
    let ret = unsafe {
        av_hwdevice_ctx_create(
            &mut hw_device,
            AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            std::ptr::null(),     // device name — VT picks the system device
            std::ptr::null_mut(), // options
            0,
        )
    };
    if ret < 0 || hw_device.is_null() {
        tracing::debug!("av_hwdevice_ctx_create VIDEOTOOLBOX returned {ret}");
        return false;
    }
    unsafe {
        let raw = dec_ctx.as_mut_ptr();
        (*raw).hw_device_ctx = hw_device;
        (*raw).get_format = Some(videotoolbox_get_format);
    }
    true
}

/// FFmpeg `get_format` callback: prefer `AV_PIX_FMT_VIDEOTOOLBOX` from
/// the offered list, fall through to the first software format otherwise
/// so a codec VT can't accelerate still decodes (just on the CPU).
#[cfg(target_os = "macos")]
unsafe extern "C" fn videotoolbox_get_format(
    _ctx: *mut ffmpeg_next::ffi::AVCodecContext,
    pix_fmts: *const ffmpeg_next::ffi::AVPixelFormat,
) -> ffmpeg_next::ffi::AVPixelFormat {
    use ffmpeg_next::ffi::AVPixelFormat::*;
    let mut p = pix_fmts;
    while unsafe { *p } != AV_PIX_FMT_NONE {
        if unsafe { *p } == AV_PIX_FMT_VIDEOTOOLBOX {
            return AV_PIX_FMT_VIDEOTOOLBOX;
        }
        p = unsafe { p.add(1) };
    }
    unsafe { *pix_fmts }
}

/// Download a hardware-resident frame (VideoToolbox CVPixelBuffer) to
/// host system memory via `av_hwframe_transfer_data`. The destination
/// frame's pixel format is determined by FFmpeg — typically `NV12` for
/// 8-bit HEVC/H.264 and `P010` for 10-bit Main10.
pub(crate) fn download_hw_frame(
    hw_frame: &ffmpeg_next::frame::Video,
    sw_frame: &mut ffmpeg_next::frame::Video,
) -> Result<()> {
    let ret = unsafe {
        ffmpeg_next::ffi::av_hwframe_transfer_data(
            sw_frame.as_mut_ptr(),
            hw_frame.as_ptr(),
            0,
        )
    };
    if ret < 0 {
        return Err(Error::Ffmpeg(format!("av_hwframe_transfer_data: {ret}")));
    }
    Ok(())
}

// ─── D3D11VA hwaccel plumbing (Windows only) ───────────────────────────

/// Wire D3D11VA hardware decode onto a codec context (Windows). Mirror
/// of [`try_enable_videotoolbox_decode`]: creates an FFmpeg-owned D3D11
/// device, installs it as `hw_device_ctx`, and points `get_format` at
/// [`d3d11va_get_format`] so the decoder emits `AV_PIX_FMT_D3D11`
/// surfaces. Returns `true` if the hwaccel attached.
///
/// This offloads HEVC decode to the GPU video engine (NVDEC on NVIDIA,
/// the equivalent block on Intel/AMD) — the single biggest win for
/// OSV's two 3840×3840 HEVC streams, which are ~an order of magnitude
/// too slow to decode in software. Decoded frames are GPU-resident;
/// callers download them to host memory with [`download_hw_frame`]
/// (NV12 for 8-bit, P010 for 10-bit Main10).
#[cfg(target_os = "windows")]
pub(crate) fn try_enable_d3d11va_decode(
    dec_ctx: &mut ffmpeg_next::codec::context::Context,
) -> bool {
    use ffmpeg_next::ffi::*;
    let mut hw_device: *mut AVBufferRef = std::ptr::null_mut();
    let ret = unsafe {
        av_hwdevice_ctx_create(
            &mut hw_device,
            AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
            std::ptr::null(),     // device name — picks the default adapter
            std::ptr::null_mut(), // options
            0,
        )
    };
    if ret < 0 || hw_device.is_null() {
        tracing::debug!("av_hwdevice_ctx_create D3D11VA returned {ret}");
        return false;
    }
    unsafe {
        let raw = dec_ctx.as_mut_ptr();
        (*raw).hw_device_ctx = hw_device;
        (*raw).get_format = Some(d3d11va_get_format);
    }
    true
}

/// FFmpeg `get_format` callback: prefer `AV_PIX_FMT_D3D11` from the
/// offered list (keeps decode on the GPU), else fall through to the
/// first software format so an unsupported profile still decodes on
/// the CPU rather than failing.
#[cfg(target_os = "windows")]
unsafe extern "C" fn d3d11va_get_format(
    _ctx: *mut ffmpeg_next::ffi::AVCodecContext,
    pix_fmts: *const ffmpeg_next::ffi::AVPixelFormat,
) -> ffmpeg_next::ffi::AVPixelFormat {
    use ffmpeg_next::ffi::AVPixelFormat::*;
    let mut p = pix_fmts;
    while unsafe { *p } != AV_PIX_FMT_NONE {
        if unsafe { *p } == AV_PIX_FMT_D3D11 {
            return AV_PIX_FMT_D3D11;
        }
        p = unsafe { p.add(1) };
    }
    unsafe { *pix_fmts }
}

/// Decode the first GPU-resident (`AV_PIX_FMT_D3D11`) frame from the largest
/// video stream of `path`. For interop testing of the zero-copy path —
/// returns the frame still on the GPU (not downloaded). Windows-only.
#[cfg(target_os = "windows")]
pub fn decode_first_d3d11_frame(path: &Path) -> Result<ffmpeg_next::frame::Video> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
    let (idx, params) = {
        let s = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .max_by_key(|s| {
                let p = unsafe { &*s.parameters().as_ptr() };
                (p.width as i64) * (p.height as i64)
            })
            .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?;
        (s.index(), s.parameters())
    };
    let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(params)
        .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
    if !try_enable_d3d11va_decode(&mut codec_ctx) {
        return Err(Error::Ffmpeg("d3d11va setup failed".into()));
    }
    let mut dec = codec_ctx
        .decoder()
        .video()
        .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;
    let mut frame = ffmpeg_next::frame::Video::empty();
    for (s, packet) in ictx.packets() {
        if s.index() != idx {
            continue;
        }
        if dec.send_packet(&packet).is_err() {
            continue;
        }
        if dec.receive_frame(&mut frame).is_ok() {
            return Ok(frame);
        }
    }
    Err(Error::Ffmpeg("no d3d11 frame decoded".into()))
}

/// Per-stream decode timing for a multi-frame benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct BenchResult {
    pub frames: u32,
    pub total: std::time::Duration,
    pub decode_path: DecodePath,
}

impl BenchResult {
    pub fn fps(&self) -> f32 {
        if self.total.is_zero() { 0.0 }
        else { self.frames as f32 / self.total.as_secs_f32() }
    }
}

/// Open a `.360`, return a streaming iterator that yields one
/// [`StreamPair`] per video-time-step (one frame from each of the two
/// HEVC streams, packed RGB8). Used by Phase 0.8's multi-frame `export`.
///
/// All the codec / hwaccel setup lives in [`StreamPairIter::new`]; the
/// iterator itself is just decoder.send_packet → receive_frame → swscale
/// → repack RGB8. Stops at EOF or after `n_frames` if non-zero.
pub fn iter_stream_pairs(path: &Path, hw: HwDecode, n_frames: u32) -> Result<StreamPairIter> {
    StreamPairIter::new(path, hw, n_frames)
}

pub struct StreamPairIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: Vec<usize>,
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    scalers: Vec<Option<ffmpeg_next::software::scaling::Context>>,
    hw_active: [bool; 2],
    dims: vr180_core::eac::Dims,
    decode_path: DecodePath,
    frame_limit: u32,
    frames_yielded: u32,
    /// Per-stream time_base (seconds per pts tick), for precise seek.
    tbs: [f64; 2],
    /// Nominal frame duration of s0 (1 / avg_frame_rate).
    dt_s: f64,
    /// Precise-seek state — see `ZeroCopyStreamPairIter::skip_until_s`.
    skip_until_s: Option<f64>,
}

impl StreamPairIter {
    pub fn new(path: &Path, hw: HwDecode, frame_limit: u32) -> Result<Self> {
        init();
        let mut ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

        let video_indices: Vec<usize> = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .map(|s| s.index())
            .collect();
        if video_indices.len() < 2 {
            return Err(Error::Ffmpeg(format!(
                "expected 2 video streams in .360 file, found {}",
                video_indices.len()
            )));
        }
        let mut decoders = Vec::with_capacity(2);
        let mut hw_active = [false, false];
        for (i, &idx) in video_indices.iter().take(2).enumerate() {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            #[cfg(target_os = "macos")]
            {
                let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
                if want_vt {
                    if try_enable_videotoolbox_decode(&mut codec_ctx) {
                        hw_active[i] = true;
                    } else if matches!(hw, HwDecode::VideoToolbox) {
                        return Err(Error::Ffmpeg("VT requested but unavailable".into()));
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (i, hw);
                if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg("VT is macOS-only".into()));
                }
            }
            decoders.push(codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?);
        }

        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "video streams disagree on dimensions: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let dims = vr180_core::eac::Dims::new(w0, h0);
        if !dims.is_valid() {
            return Err(Error::Ffmpeg(format!(
                "stream width {w0} not a valid EAC layout"
            )));
        }
        let decode_path = if hw_active.iter().all(|&a| a) {
            DecodePath::VideoToolbox
        } else {
            DecodePath::Software
        };
        let mut tbs = [0.0_f64; 2];
        for (i, &idx) in video_indices.iter().take(2).enumerate() {
            let tb = ictx.stream(idx).unwrap().time_base();
            tbs[i] = tb.numerator() as f64 / tb.denominator().max(1) as f64;
        }
        let fr = ictx.stream(video_indices[0]).unwrap().avg_frame_rate();
        let dt_s = if fr.numerator() > 0 {
            fr.denominator() as f64 / fr.numerator() as f64
        } else {
            1.0 / 30.0
        };
        Ok(Self {
            ictx, video_indices, decoders,
            scalers: vec![None, None],
            hw_active, dims, decode_path,
            frame_limit,
            frames_yielded: 0,
            tbs, dt_s, skip_until_s: None,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.dims }
    pub fn decode_path(&self) -> DecodePath { self.decode_path }

    /// CPU-path counterpart to `ZeroCopyStreamPairIter::seek` — same
    /// PRECISE semantics: the next `next_pair` returns the frame at
    /// `target_s` (±½ frame), decoding-and-discarding the keyframe
    /// run-in internally (repack to RGB is skipped for discarded
    /// frames, so the run-in costs decode only).
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let target_s = target_s.max(0.0);
        let ts = (target_s * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek to {target_s:.3}s: {e}")))?;
        for d in &mut self.decoders {
            d.flush();
        }
        self.frames_yielded = 0;
        self.skip_until_s = Some(target_s);
        Ok(())
    }

    /// Pull the next `StreamPair` (one frame from each video stream).
    /// Returns `Ok(None)` at EOF or after the frame limit is hit.
    pub fn next_pair(&mut self) -> Result<Option<StreamPair>> {
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut frames: [Option<Vec<u8>>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut sw_storage = ffmpeg_next::frame::Video::empty();

        // Fill-and-maybe-discard loop (precise seek): pairs in the
        // keyframe→target run-in are decoded but NOT repacked to RGB
        // (empty placeholder) and dropped; the first pair at/after the
        // target is repacked and returned. A frame is in the run-in iff
        // its OWN pts is below target − ½dt, so the target pair itself
        // always gets the real repack (both streams cross the threshold
        // on the same pair — the pairing invariant keeps them lockstep).
        let in_run_in = |skip: Option<f64>, pts: Option<i64>, tb: f64, dt: f64| -> bool {
            match skip {
                Some(target) => (pts.unwrap_or(0) as f64 * tb) < target - 0.5 * dt,
                None => false,
            }
        };
        let (s0, s4) = 'fill: loop {
            // Try receiving leftover-buffered frames first (the HEVC decoder
            // often has prepared frames from previous packets).
            for pos in 0..2 {
                if frames[pos].is_some() { continue; }
                let dec = &mut self.decoders[pos];
                if dec.receive_frame(&mut decoded).is_ok() {
                    frames[pos] = Some(
                        if in_run_in(self.skip_until_s, decoded.pts(), self.tbs[pos], self.dt_s) {
                            Vec::new()
                        } else {
                            repack_to_rgb8(
                                &mut decoded, &mut sw_storage,
                                self.hw_active[pos], &mut self.scalers[pos],
                            )?
                        });
                }
            }

            // Then drive the packet loop until both slots are filled or EOF.
            //
            // Same invariant as ZeroCopyStreamPairIter::next_pair: every
            // packet for a video stream we care about must be forwarded to
            // its decoder, even if we already have a frame for that stream
            // this iteration. Dropping packets causes that stream to fall
            // behind the other → pairing s0_K with s4_K+1 → EAC faces from
            // different temporal frames in the same output.
            if frames.iter().any(|f| f.is_none()) {
                loop {
                    let res = self.ictx.packets().next();
                    let (stream, packet) = match res {
                        Some(x) => x,
                        None => break,
                    };
                    let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                        Some(p) => p,
                        None => continue,
                    };
                    let dec = &mut self.decoders[pos];
                    if dec.send_packet(&packet).is_err() { continue; }
                    // Only repack a frame from this decoder if we still
                    // need it this iteration. Extras stay in the decoder's
                    // queue and feed the next next_pair call.
                    if frames[pos].is_none()
                        && dec.receive_frame(&mut decoded).is_ok()
                    {
                        frames[pos] = Some(
                            if in_run_in(self.skip_until_s, decoded.pts(), self.tbs[pos], self.dt_s) {
                                Vec::new()
                            } else {
                                repack_to_rgb8(
                                    &mut decoded, &mut sw_storage,
                                    self.hw_active[pos], &mut self.scalers[pos],
                                )?
                            });
                    }
                    if frames.iter().all(|f| f.is_some()) { break; }
                }
            }

            let (Some(s0), Some(s4)) = (frames[0].take(), frames[1].take()) else {
                return Ok(None);  // EOF before both streams had a frame
            };
            if self.skip_until_s.is_some() {
                if s0.is_empty() || s4.is_empty() {
                    // Run-in pair — discard and assemble the next one.
                    frames = [None, None];
                    continue 'fill;
                }
                self.skip_until_s = None;
            }
            break 'fill (s0, s4);
        };
        self.frames_yielded += 1;
        Ok(Some(StreamPair { s0, s4, dims: self.dims, decode_path: self.decode_path }))
    }
}

/// CPU-path counterpart to [`SegmentedZeroCopyPairIter`] — chains GoPro
/// `.360` segments through `StreamPairIter`. Single-element passthrough.
pub struct SegmentedStreamPairIter {
    segments: Vec<std::path::PathBuf>,
    seg_start_s: Vec<f64>,
    total_dur_s: f64,
    hw: HwDecode,
    cur_idx: usize,
    cur: StreamPairIter,
}

impl SegmentedStreamPairIter {
    pub fn new(segments: &[std::path::PathBuf], hw: HwDecode, frame_limit: u32) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for seg in segments {
            seg_start_s.push(acc);
            acc += probe_video(seg).map(|p| p.duration_sec).unwrap_or(0.0);
        }
        let cur = StreamPairIter::new(&segments[0], hw, frame_limit)?;
        Ok(Self {
            segments: segments.to_vec(),
            seg_start_s, total_dur_s: acc, hw, cur_idx: 0, cur,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.cur.dims() }
    pub fn decode_path(&self) -> DecodePath { self.cur.decode_path() }
    pub fn total_duration_s(&self) -> f64 { self.total_dur_s }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let t = target_s.clamp(0.0, self.total_dur_s);
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            self.cur = StreamPairIter::new(&self.segments[idx], self.hw, 0)?;
            self.cur_idx = idx;
        }
        self.cur.seek(t - self.seg_start_s[idx])
    }

    pub fn next_pair(&mut self) -> Result<Option<StreamPair>> {
        loop {
            if let Some(p) = self.cur.next_pair()? {
                return Ok(Some(p));
            }
            if self.cur_idx + 1 >= self.segments.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            self.cur = StreamPairIter::new(&self.segments[self.cur_idx], self.hw, 0)?;
        }
    }
}

/// Shared frame → packed-RGB8 conversion used by both single-frame and
/// streaming code paths.
fn repack_to_rgb8(
    decoded: &mut ffmpeg_next::frame::Video,
    sw_storage: &mut ffmpeg_next::frame::Video,
    hw_active: bool,
    scaler: &mut Option<ffmpeg_next::software::scaling::Context>,
) -> Result<Vec<u8>> {
    let frame_ref: &ffmpeg_next::frame::Video = if hw_active
        && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
    {
        download_hw_frame(decoded, sw_storage)?;
        sw_storage
    } else {
        decoded
    };
    let s = match scaler {
        Some(s) => s,
        None => {
            *scaler = Some(ffmpeg_next::software::scaling::Context::get(
                frame_ref.format(),
                frame_ref.width(), frame_ref.height(),
                ffmpeg_next::format::Pixel::RGB24,
                frame_ref.width(), frame_ref.height(),
                ffmpeg_next::software::scaling::Flags::BILINEAR,
            ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?);
            scaler.as_mut().unwrap()
        }
    };
    let mut rgb = ffmpeg_next::frame::Video::empty();
    s.run(frame_ref, &mut rgb).map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let src = rgb.data(0);
    let src_stride = rgb.stride(0);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &src[y * src_stride..y * src_stride + w * 3];
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// Multi-frame decode benchmark: decode `n_frames` from the FIRST video
/// stream as fast as possible, **no swscale RGB conversion**, returning
/// per-stream timing. This measures the steady-state decode throughput
/// where the single-frame cold-start cost has been amortized away —
/// which is where any hwaccel speedup actually shows up.
///
/// Doesn't assemble crosses; doesn't run the GPU pipeline. Pure decode
/// throughput test. Useful for the Phase 0.6 sw-vs-VT comparison.
pub fn bench_decode_throughput(
    path: &Path,
    n_frames: u32,
    hw: HwDecode,
) -> Result<BenchResult> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .next()
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?
        .index();

    let stream = ictx.stream(stream_idx).unwrap();
    let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

    let mut hw_active = false;
    #[cfg(target_os = "macos")]
    {
        let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
        if want_vt {
            if try_enable_videotoolbox_decode(&mut codec_ctx) {
                hw_active = true;
            } else if matches!(hw, HwDecode::VideoToolbox) {
                return Err(Error::Ffmpeg(
                    "VT hwaccel requested but unavailable on this system".into()
                ));
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = hw;
        if matches!(hw, HwDecode::VideoToolbox) {
            return Err(Error::Ffmpeg("VT is macOS-only".into()));
        }
    }

    let mut decoder = codec_ctx.decoder().video()
        .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;

    let mut decoded = ffmpeg_next::frame::Video::empty();
    let mut sw_storage = ffmpeg_next::frame::Video::empty();
    let mut count: u32 = 0;
    let t_start = std::time::Instant::now();

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx { continue; }
        if decoder.send_packet(&packet).is_err() { continue; }
        while decoder.receive_frame(&mut decoded).is_ok() {
            // For hwframes, do the transfer so we measure the full
            // "frames usable in host memory" throughput (matching real
            // pipeline usage). Skip if you want raw decode throughput.
            if hw_active
                && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
            {
                download_hw_frame(&decoded, &mut sw_storage)?;
            }
            count += 1;
            if count >= n_frames { break 'outer; }
        }
    }
    let total = t_start.elapsed();
    let decode_path = if hw_active { DecodePath::VideoToolbox } else { DecodePath::Software };
    Ok(BenchResult { frames: count, total, decode_path })
}
