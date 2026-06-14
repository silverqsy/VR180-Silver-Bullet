//! Native MP4 atom injector for VR180 metadata.
//!
//! Pure Rust port of `vr180_injector.py` from
//! <https://github.com/silverqsy/VR180Injector> — the user's standalone
//! injector tool. Two top-level operations:
//!
//! - [`inject_youtube_vr180`] — Google Spherical Video V2 (`st3d` + `sv3d`).
//!   YouTube and Quest Browser pick this up.
//! - [`inject_apmp_vr180`] — Apple Projected Media Profile (`vexu` + `hfov`).
//!   Vision Pro / visionOS 26+ native VR180 playback.
//!
//! Both write **directly into the existing MP4** — no re-encode, no
//! external dependency, ~100 bytes added. The shared
//! [`inject_atoms_into_visual_sample_entry`] helper:
//!   1. Locates `moov.trak (video).mdia.minf.stbl.stsd.{hvc1|hev1|...}`.
//!   2. Strips any conflicting metadata atoms (`{st3d, sv3d, vexu, hfov}`).
//!   3. Appends the new atom blob inside the visual sample entry.
//!   4. Updates parent atom sizes up to `moov`.
//!   5. Fixes `stco`/`co64` chunk offsets when `moov` precedes `mdat`.
//!
//! The Python helper hard-codes the `hvc1`/`hev1` lookup so it only
//! works on HEVC outputs; we extend it to also match ProRes
//! (`apch`/`apcn`/`apco`/`apcs`/`ap4h`/`ap4x`) and H.264 (`avc1`/`avc3`)
//! sample entries since the VR180 atoms are codec-agnostic.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{Error, Result};

/// Visual sample-entry tags we recognise. Order is preferred — first
/// match wins. HEVC first (most common), then ProRes, then H.264.
const VISUAL_SAMPLE_ENTRY_TAGS: &[&[u8; 4]] = &[
    b"hvc1", b"hev1",                            // HEVC
    b"apch", b"apcn", b"apco", b"apcs", b"ap4h", b"ap4x", // ProRes
    b"avc1", b"avc3",                            // H.264
];

/// Tags we strip from the visual sample entry before injection. Both
/// VR180 metadata flavours (YouTube and APMP) share these — they
/// conflict (Spherical V2 says 360° equirect, APMP says halfEquirect)
/// so an injector should never leave both in place.
const STRIP_TAGS: &[&[u8; 4]] = &[b"st3d", b"sv3d", b"vexu", b"hfov"];

/// Inject YouTube VR180 (Google Spherical Video V2) — `st3d` + `sv3d`.
///
/// Modifies `path` in place. After this call YouTube + Quest Browser
/// recognise the file as VR180 stereo.
pub fn inject_youtube_vr180(path: &Path) -> Result<()> {
    // st3d: stereo_mode = 2 (leftRight / side-by-side)
    let mut st3d = Vec::with_capacity(13);
    st3d.extend_from_slice(&13u32.to_be_bytes());
    st3d.extend_from_slice(b"st3d");
    st3d.extend_from_slice(&[0, 0, 0, 0]); // version + flags
    st3d.push(2);                          // stereo_mode = leftRight

    // svhd: SphericalVideoHeader (one null-terminated metadata source byte = 0)
    let mut svhd = Vec::with_capacity(13);
    svhd.extend_from_slice(&13u32.to_be_bytes());
    svhd.extend_from_slice(b"svhd");
    svhd.extend_from_slice(&[0, 0, 0, 0]);
    svhd.push(0);

    // prhd: ProjectionHeader (yaw/pitch/roll = 0)
    let mut prhd = Vec::with_capacity(24);
    prhd.extend_from_slice(&24u32.to_be_bytes());
    prhd.extend_from_slice(b"prhd");
    prhd.extend_from_slice(&[0u8; 16]);

    // equi: EquirectangularProjection — VR180 = crop 180° (0x3FFFFFFF) from left & right
    let mut equi = Vec::with_capacity(28);
    equi.extend_from_slice(&28u32.to_be_bytes());
    equi.extend_from_slice(b"equi");
    equi.extend_from_slice(&[0, 0, 0, 0]);       // version + flags
    equi.extend_from_slice(&0u32.to_be_bytes()); // top
    equi.extend_from_slice(&0u32.to_be_bytes()); // bottom
    equi.extend_from_slice(&0x3FFFFFFFu32.to_be_bytes()); // left
    equi.extend_from_slice(&0x3FFFFFFFu32.to_be_bytes()); // right

    // proj container
    let proj_size = (8 + prhd.len() + equi.len()) as u32;
    let mut proj = Vec::with_capacity(proj_size as usize);
    proj.extend_from_slice(&proj_size.to_be_bytes());
    proj.extend_from_slice(b"proj");
    proj.extend_from_slice(&prhd);
    proj.extend_from_slice(&equi);

    // sv3d container
    let sv3d_size = (8 + svhd.len() + proj.len()) as u32;
    let mut sv3d = Vec::with_capacity(sv3d_size as usize);
    sv3d.extend_from_slice(&sv3d_size.to_be_bytes());
    sv3d.extend_from_slice(b"sv3d");
    sv3d.extend_from_slice(&svhd);
    sv3d.extend_from_slice(&proj);

    let mut inject = Vec::with_capacity(st3d.len() + sv3d.len());
    inject.extend_from_slice(&st3d);
    inject.extend_from_slice(&sv3d);

    inject_atoms_into_visual_sample_entry(path, &inject)
}

/// Inject Apple Projected Media Profile (VR180 APMP) — `vexu` + `hfov`.
///
/// `baseline_mm` is the inter-pupillary distance the camera shot at —
/// typically 63 – 65 mm. Modifies `path` in place.
pub fn inject_apmp_vr180(path: &Path, baseline_mm: f32) -> Result<()> {
    // eyes/stri: stereo indication = 0x03 (side-by-side)
    let mut stri = Vec::with_capacity(13);
    stri.extend_from_slice(&13u32.to_be_bytes());
    stri.extend_from_slice(b"stri");
    stri.extend_from_slice(&[0, 0, 0, 0]);
    stri.push(0x03);

    let eyes_size = (8 + stri.len()) as u32;
    let mut eyes = Vec::with_capacity(eyes_size as usize);
    eyes.extend_from_slice(&eyes_size.to_be_bytes());
    eyes.extend_from_slice(b"eyes");
    eyes.extend_from_slice(&stri);

    // proj/prji: projection = halfEquirectangular (`hequ`)
    let mut prji = Vec::with_capacity(16);
    prji.extend_from_slice(&16u32.to_be_bytes());
    prji.extend_from_slice(b"prji");
    prji.extend_from_slice(&[0, 0, 0, 0]);
    prji.extend_from_slice(b"hequ");

    let proj_size = (8 + prji.len()) as u32;
    let mut proj = Vec::with_capacity(proj_size as usize);
    proj.extend_from_slice(&proj_size.to_be_bytes());
    proj.extend_from_slice(b"proj");
    proj.extend_from_slice(&prji);

    // pack/pkin: view packing = `side` (side-by-side)
    let mut pkin = Vec::with_capacity(16);
    pkin.extend_from_slice(&16u32.to_be_bytes());
    pkin.extend_from_slice(b"pkin");
    pkin.extend_from_slice(&[0, 0, 0, 0]);
    pkin.extend_from_slice(b"side");

    let pack_size = (8 + pkin.len()) as u32;
    let mut pack = Vec::with_capacity(pack_size as usize);
    pack.extend_from_slice(&pack_size.to_be_bytes());
    pack.extend_from_slice(b"pack");
    pack.extend_from_slice(&pkin);

    // cams/blin: baseline in mm × 65536 (16.16 fixed-point)
    let blin_val = (baseline_mm * 65536.0).round() as u32;
    let mut blin = Vec::with_capacity(12);
    blin.extend_from_slice(&12u32.to_be_bytes());
    blin.extend_from_slice(b"blin");
    blin.extend_from_slice(&blin_val.to_be_bytes());

    let cams_size = (8 + blin.len()) as u32;
    let mut cams = Vec::with_capacity(cams_size as usize);
    cams.extend_from_slice(&cams_size.to_be_bytes());
    cams.extend_from_slice(b"cams");
    cams.extend_from_slice(&blin);

    let vexu_size = (8 + eyes.len() + proj.len() + pack.len() + cams.len()) as u32;
    let mut vexu = Vec::with_capacity(vexu_size as usize);
    vexu.extend_from_slice(&vexu_size.to_be_bytes());
    vexu.extend_from_slice(b"vexu");
    vexu.extend_from_slice(&eyes);
    vexu.extend_from_slice(&proj);
    vexu.extend_from_slice(&pack);
    vexu.extend_from_slice(&cams);

    // hfov: horizontal field of view = 180° (stored as 180000 = 180.000°)
    let mut hfov = Vec::with_capacity(12);
    hfov.extend_from_slice(&12u32.to_be_bytes());
    hfov.extend_from_slice(b"hfov");
    hfov.extend_from_slice(&180_000u32.to_be_bytes());

    let mut inject = Vec::with_capacity(vexu.len() + hfov.len());
    inject.extend_from_slice(&vexu);
    inject.extend_from_slice(&hfov);

    inject_atoms_into_visual_sample_entry(path, &inject)
}

/// Find an atom with the given tag in `buf[start..end]`. Returns
/// `(offset, size)` relative to the start of `buf`. Handles 64-bit
/// extended-size atoms (size==1, real size in the next 8 bytes) and
/// extends-to-end atoms (size==0, runs to `end`).
pub(crate) fn find_atom(buf: &[u8], start: usize, end: usize, tag: &[u8; 4]) -> Option<(usize, usize)> {
    let mut pos = start;
    while pos + 8 <= end {
        let sz = u32::from_be_bytes(buf[pos..pos + 4].try_into().ok()?);
        let t = &buf[pos + 4..pos + 8];
        let sz = if sz == 1 {
            if pos + 16 > end { return None; }
            u64::from_be_bytes(buf[pos + 8..pos + 16].try_into().ok()?) as usize
        } else if sz == 0 {
            end - pos
        } else {
            sz as usize
        };
        if sz < 8 || pos + sz > end {
            return None;
        }
        if t == tag {
            return Some((pos, sz));
        }
        pos += sz;
    }
    None
}

/// Build the Google spatial-media `SA3D` box for an N-channel 1st-order
/// AmbiX ambisonic track (ACN ordering, SN3D normalization, identity
/// channel map). Goes inside the audio sample entry; VR-aware players
/// (YouTube VR, Quest Browser) read it to head-track the audio.
fn sa3d_box(num_channels: u32) -> Vec<u8> {
    let mut c = Vec::new();
    c.push(0u8);                                   // version
    c.push(0u8);                                   // ambisonic_type = 0 (periphonic)
    c.extend_from_slice(&1u32.to_be_bytes());      // ambisonic_order = 1
    c.push(0u8);                                   // channel_ordering = 0 (ACN)
    c.push(0u8);                                   // normalization = 0 (SN3D)
    c.extend_from_slice(&num_channels.to_be_bytes());
    for ch in 0..num_channels { c.extend_from_slice(&ch.to_be_bytes()); } // channel_map
    let mut b = Vec::with_capacity(8 + c.len());
    b.extend_from_slice(&((8 + c.len()) as u32).to_be_bytes());
    b.extend_from_slice(b"SA3D");
    b.extend_from_slice(&c);
    b
}

/// Find the audio track (`trak` containing a `soun` handler) inside `moov`.
fn find_audio_trak(buf: &[u8], moov_off: usize, moov_sz: usize) -> Option<(usize, usize)> {
    let mut pos = moov_off + 8;
    let end = moov_off + moov_sz;
    while pos + 8 <= end {
        let sz = u32::from_be_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        let t = &buf[pos + 4..pos + 8];
        if sz < 8 || pos + sz > end { return None; }
        if t == b"trak" && memmem(&buf[pos..pos + sz], b"soun").is_some() {
            return Some((pos, sz));
        }
        pos += sz;
    }
    None
}

/// Inject an `SA3D` ambisonic-spatial-audio box into the FIRST audio
/// sample entry, appended after the entry's existing child boxes (the
/// spatial-media convention). Mirrors the visual injector's moov
/// surgery: load moov → splice the larger entry → patch the parent size
/// chain + stco/co64. No-op-safe: returns an error (caller logs, keeps
/// the audio) if the file has no audio track / sample entry.
pub fn inject_sa3d_ambix(path: &Path, num_channels: u32) -> Result<()> {
    let mut f = OpenOptions::new().read(true).write(true).open(path).map_err(Error::Io)?;
    let fsize = f.metadata().map_err(Error::Io)?.len();

    // Locate moov.
    let (mut moov_off, mut moov_sz) = (None, None);
    let mut pos: u64 = 0;
    while pos + 8 <= fsize {
        let mut hdr = [0u8; 8];
        f.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
        f.read_exact(&mut hdr).map_err(Error::Io)?;
        let sz32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        let tag = &hdr[4..8];
        let sz: u64 = if sz32 == 1 {
            let mut ext = [0u8; 8]; f.read_exact(&mut ext).map_err(Error::Io)?;
            u64::from_be_bytes(ext)
        } else if sz32 == 0 { fsize - pos } else { sz32 as u64 };
        if sz < 8 { break; }
        if tag == b"moov" { moov_off = Some(pos); moov_sz = Some(sz); break; }
        pos += sz;
    }
    let moov_off = moov_off.ok_or_else(|| Error::Ffmpeg("moov not found".into()))?;
    let moov_sz = moov_sz.unwrap();

    let mut data = vec![0u8; moov_sz as usize];
    f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
    f.read_exact(&mut data).map_err(Error::Io)?;

    let moov = find_atom(&data, 0, data.len(), b"moov")
        .ok_or_else(|| Error::Ffmpeg("moov not in buffer".into()))?;
    let trak = find_audio_trak(&data, moov.0, moov.1)
        .ok_or_else(|| Error::Ffmpeg("audio trak not found".into()))?;
    let mdia = find_atom(&data, trak.0 + 8, trak.0 + trak.1, b"mdia")
        .ok_or_else(|| Error::Ffmpeg("audio mdia not found".into()))?;
    let minf = find_atom(&data, mdia.0 + 8, mdia.0 + mdia.1, b"minf")
        .ok_or_else(|| Error::Ffmpeg("audio minf not found".into()))?;
    let stbl = find_atom(&data, minf.0 + 8, minf.0 + minf.1, b"stbl")
        .ok_or_else(|| Error::Ffmpeg("audio stbl not found".into()))?;
    let stsd = find_atom(&data, stbl.0 + 8, stbl.0 + stbl.1, b"stsd")
        .ok_or_else(|| Error::Ffmpeg("audio stsd not found".into()))?;

    // First sample entry = first box after the stsd FullBox+count (16 bytes).
    let entry_off = stsd.0 + 16;
    if entry_off + 8 > stsd.0 + stsd.1 {
        return Err(Error::Ffmpeg("audio sample entry not found".into()));
    }
    let entry_sz = u32::from_be_bytes(data[entry_off..entry_off + 4].try_into().unwrap()) as usize;
    if entry_sz < 8 || entry_off + entry_sz > stsd.0 + stsd.1 {
        return Err(Error::Ffmpeg("bad audio sample entry size".into()));
    }
    // Bail if SA3D already present (idempotent).
    if memmem(&data[entry_off..entry_off + entry_sz], b"SA3D").is_some() {
        return Ok(());
    }

    let sa3d = sa3d_box(num_channels);
    let new_entry_sz = entry_sz + sa3d.len();
    let size_delta = sa3d.len() as i64;

    // Splice: entry grows by SA3D appended at its end.
    let mut new_data = Vec::with_capacity(data.len() + sa3d.len());
    new_data.extend_from_slice(&data[..entry_off]);
    new_data.extend_from_slice(&(new_entry_sz as u32).to_be_bytes());     // new entry size
    new_data.extend_from_slice(&data[entry_off + 4..entry_off + entry_sz]); // entry body
    new_data.extend_from_slice(&sa3d);                                     // SA3D at end
    new_data.extend_from_slice(&data[entry_off + entry_sz..]);
    data = new_data;

    // Patch parent size chain.
    for (off, _) in [stsd, stbl, minf, mdia, trak, moov] {
        let old = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as i64;
        data[off..off + 4].copy_from_slice(&((old + size_delta) as u32).to_be_bytes());
    }

    // Patch chunk offsets if moov precedes mdat.
    let moov_at_end = (moov_off + moov_sz) >= fsize;
    if !moov_at_end {
        for tag in [b"stco" as &[u8; 4], b"co64"] {
            let mut search = 0usize;
            while let Some(rel) = memmem(&data[search..], tag) {
                let idx = search + rel;
                if idx < 4 { break; }
                let n = u32::from_be_bytes(data[idx + 8..idx + 12].try_into().unwrap()) as usize;
                if tag == b"stco" {
                    for e in 0..n {
                        let eo = idx + 12 + e * 4;
                        if eo + 4 > data.len() { break; }
                        let v = u32::from_be_bytes(data[eo..eo + 4].try_into().unwrap()) as i64;
                        data[eo..eo + 4].copy_from_slice(&((v + size_delta) as u32).to_be_bytes());
                    }
                } else {
                    for e in 0..n {
                        let eo = idx + 12 + e * 8;
                        if eo + 8 > data.len() { break; }
                        let v = u64::from_be_bytes(data[eo..eo + 8].try_into().unwrap()) as i64;
                        data[eo..eo + 8].copy_from_slice(&((v + size_delta) as u64).to_be_bytes());
                    }
                }
                let atom_sz = u32::from_be_bytes(data[idx - 4..idx].try_into().unwrap()) as usize;
                search = idx + atom_sz.max(8);
            }
        }
    }

    // Write back.
    if moov_at_end {
        f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
        f.write_all(&data).map_err(Error::Io)?;
        f.set_len(moov_off + data.len() as u64).map_err(Error::Io)?;
    } else {
        let mut after = Vec::new();
        f.seek(SeekFrom::Start(moov_off + moov_sz)).map_err(Error::Io)?;
        f.read_to_end(&mut after).map_err(Error::Io)?;
        f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
        f.write_all(&data).map_err(Error::Io)?;
        f.write_all(&after).map_err(Error::Io)?;
        f.set_len(moov_off + data.len() as u64 + after.len() as u64).map_err(Error::Io)?;
    }
    tracing::info!(path=%path.display(), num_channels, "SA3D ambisonic metadata injected");
    Ok(())
}

/// Find the video track (`trak` containing `vide` handler) inside `moov`.
fn find_video_trak(buf: &[u8], moov_off: usize, moov_sz: usize) -> Option<(usize, usize)> {
    let mut pos = moov_off + 8;
    let end = moov_off + moov_sz;
    while pos + 8 <= end {
        let sz = u32::from_be_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        let t = &buf[pos + 4..pos + 8];
        if sz < 8 || pos + sz > end {
            return None;
        }
        if t == b"trak" && memmem(&buf[pos..pos + sz], b"vide").is_some() {
            return Some((pos, sz));
        }
        pos += sz;
    }
    None
}

fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() { return None; }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Core injector — `inject_data` is appended verbatim inside the
/// visual sample entry. Any of `STRIP_TAGS` already present inside
/// the entry are removed first.
fn inject_atoms_into_visual_sample_entry(path: &Path, inject_data: &[u8]) -> Result<()> {
    // Step 1: scan the file from the top to find the `moov` atom
    // offset and size, without reading the whole file into memory.
    let mut f = OpenOptions::new().read(true).write(true).open(path)
        .map_err(|e| Error::Io(e))?;
    let fsize = f.metadata().map_err(Error::Io)?.len();
    let mut moov_off: Option<u64> = None;
    let mut moov_sz: Option<u64> = None;

    f.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
    let mut pos: u64 = 0;
    while pos + 8 <= fsize {
        let mut hdr = [0u8; 8];
        f.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
        f.read_exact(&mut hdr).map_err(Error::Io)?;
        let sz32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        let tag = &hdr[4..8];
        let sz: u64 = if sz32 == 1 {
            let mut ext = [0u8; 8];
            f.read_exact(&mut ext).map_err(Error::Io)?;
            u64::from_be_bytes(ext)
        } else if sz32 == 0 {
            fsize - pos
        } else {
            sz32 as u64
        };
        if sz < 8 { break; }
        if tag == b"moov" {
            moov_off = Some(pos);
            moov_sz = Some(sz);
            break;
        }
        pos += sz;
    }
    let moov_off = moov_off.ok_or_else(|| Error::Ffmpeg("moov atom not found".into()))?;
    let moov_sz  = moov_sz.unwrap();

    // Step 2: pull the moov atom into memory and walk down to the
    // visual sample entry.
    let mut data = vec![0u8; moov_sz as usize];
    f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
    f.read_exact(&mut data).map_err(Error::Io)?;

    let moov = find_atom(&data, 0, data.len(), b"moov")
        .ok_or_else(|| Error::Ffmpeg("moov not found in buffer".into()))?;
    let trak = find_video_trak(&data, moov.0, moov.1)
        .ok_or_else(|| Error::Ffmpeg("video trak not found".into()))?;
    let mdia = find_atom(&data, trak.0 + 8, trak.0 + trak.1, b"mdia")
        .ok_or_else(|| Error::Ffmpeg("mdia not found".into()))?;
    let minf = find_atom(&data, mdia.0 + 8, mdia.0 + mdia.1, b"minf")
        .ok_or_else(|| Error::Ffmpeg("minf not found".into()))?;
    let stbl = find_atom(&data, minf.0 + 8, minf.0 + minf.1, b"stbl")
        .ok_or_else(|| Error::Ffmpeg("stbl not found".into()))?;
    let stsd = find_atom(&data, stbl.0 + 8, stbl.0 + stbl.1, b"stsd")
        .ok_or_else(|| Error::Ffmpeg("stsd not found".into()))?;

    // The first 8 bytes of stsd body are the FullBox header (version +
    // flags + entry_count), then the entries start. Search from
    // stsd.off + 16 to skip past those.
    let mut entry: Option<(usize, usize)> = None;
    for &tag in VISUAL_SAMPLE_ENTRY_TAGS {
        if let Some(e) = find_atom(&data, stsd.0 + 16, stsd.0 + stsd.1, tag) {
            entry = Some(e);
            break;
        }
    }
    let entry = entry.ok_or_else(|| Error::Ffmpeg(
        "visual sample entry not found (need hvc1/hev1/apch/apcn/avc1)".into(),
    ))?;
    let entry_off = entry.0;
    let entry_sz  = entry.1;

    // Step 3: strip conflicting atoms from inside the visual sample
    // entry (the first 86 bytes are the entry header + VisualSampleEntry
    // fields; child atoms start at offset 86).
    let entry_header_len = 86usize;
    let header = data[entry_off..entry_off + entry_header_len].to_vec();
    let mut kept: Vec<u8> = Vec::new();
    let mut p = entry_off + entry_header_len;
    let end = entry_off + entry_sz;
    while p + 8 <= end {
        let asz = u32::from_be_bytes(data[p..p + 4].try_into().unwrap()) as usize;
        let atag: [u8; 4] = data[p + 4..p + 8].try_into().unwrap();
        if asz < 8 || p + asz > end {
            break;
        }
        let stripped = STRIP_TAGS.iter().any(|t| **t == atag);
        if !stripped {
            kept.extend_from_slice(&data[p..p + asz]);
        }
        p += asz;
    }

    // Step 4: rebuild the visual sample entry = header + kept + inject.
    let new_body_len = kept.len() + inject_data.len();
    let new_entry_sz = entry_header_len + new_body_len;
    let mut new_entry = Vec::with_capacity(new_entry_sz);
    new_entry.extend_from_slice(&(new_entry_sz as u32).to_be_bytes()); // size
    new_entry.extend_from_slice(&header[4..entry_header_len]);         // header tail
    new_entry.extend_from_slice(&kept);
    new_entry.extend_from_slice(inject_data);
    let size_delta: i64 = new_entry_sz as i64 - entry_sz as i64;

    // Splice the new entry into the moov buffer.
    let mut new_data: Vec<u8> = Vec::with_capacity(data.len() + size_delta.max(0) as usize);
    new_data.extend_from_slice(&data[..entry_off]);
    new_data.extend_from_slice(&new_entry);
    new_data.extend_from_slice(&data[entry_off + entry_sz..]);
    data = new_data;

    // Step 5: walk back up the parent chain, adding `size_delta` to
    // each container's size field.
    for (off, _) in [stsd, stbl, minf, mdia, trak, moov] {
        let old = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as i64;
        let new = (old + size_delta) as u32;
        data[off..off + 4].copy_from_slice(&new.to_be_bytes());
    }

    // Step 6: if `moov` was before `mdat`, increasing moov's size
    // pushes mdat farther into the file. Patch every stco / co64
    // entry in the moov buffer so chunk offsets stay correct.
    let moov_at_end = (moov_off + moov_sz) >= fsize;
    if size_delta != 0 && !moov_at_end {
        for tag in [b"stco" as &[u8; 4], b"co64"] {
            let mut search = 0usize;
            while let Some(idx_rel) = memmem(&data[search..], tag) {
                let idx = search + idx_rel;
                if idx < 4 { break; }
                let n = u32::from_be_bytes(data[idx + 8..idx + 12].try_into().unwrap()) as usize;
                if tag == b"stco" {
                    for e in 0..n {
                        let eo = idx + 12 + e * 4;
                        if eo + 4 > data.len() { break; }
                        let v = u32::from_be_bytes(data[eo..eo + 4].try_into().unwrap()) as i64;
                        let v = (v + size_delta) as u32;
                        data[eo..eo + 4].copy_from_slice(&v.to_be_bytes());
                    }
                } else {
                    for e in 0..n {
                        let eo = idx + 12 + e * 8;
                        if eo + 8 > data.len() { break; }
                        let v = u64::from_be_bytes(data[eo..eo + 8].try_into().unwrap()) as i64;
                        let v = (v + size_delta) as u64;
                        data[eo..eo + 8].copy_from_slice(&v.to_be_bytes());
                    }
                }
                let atom_sz = u32::from_be_bytes(data[idx - 4..idx].try_into().unwrap()) as usize;
                search = idx + atom_sz.max(8);
            }
        }
    }

    // Step 7: write back. If moov is at the end of the file we can just
    // overwrite from moov_off and truncate. Otherwise we have to read
    // everything after the old moov, then write [data][after_moov].
    if moov_at_end {
        f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
        f.write_all(&data).map_err(Error::Io)?;
        let new_len = moov_off + data.len() as u64;
        f.set_len(new_len).map_err(Error::Io)?;
    } else {
        let mut after_moov = Vec::with_capacity((fsize - moov_off - moov_sz) as usize);
        f.seek(SeekFrom::Start(moov_off + moov_sz)).map_err(Error::Io)?;
        f.read_to_end(&mut after_moov).map_err(Error::Io)?;
        f.seek(SeekFrom::Start(moov_off)).map_err(Error::Io)?;
        f.write_all(&data).map_err(Error::Io)?;
        f.write_all(&after_moov).map_err(Error::Io)?;
        let new_len = moov_off + data.len() as u64 + after_moov.len() as u64;
        f.set_len(new_len).map_err(Error::Io)?;
    }

    tracing::info!(
        path = %path.display(),
        bytes_added = inject_data.len(),
        size_delta,
        "VR180 metadata injected"
    );
    Ok(())
}
