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
//! - [`inject_apmp_lens`] — the same profile's parametric immersive
//!   projection: the frame is the image of a lens, and the file states the
//!   lens (`vexu` with a lens collection).
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
    inject_atoms_into_visual_sample_entry(path, &youtube_vr180_atoms(true))
}

/// [`inject_youtube_vr180`] for a frame that holds ONE eye: the same 180°
/// equirectangular bounds, declared monoscopic.
pub fn inject_youtube_vr180_mono(path: &Path) -> Result<()> {
    inject_atoms_into_visual_sample_entry(path, &youtube_vr180_atoms(false))
}

/// The `st3d` + `sv3d` boxes of a 180° half-equirect file — two eyes side by
/// side (`stereo`), or one eye as the whole frame.
fn youtube_vr180_atoms(stereo: bool) -> Vec<u8> {
    // st3d: stereo_mode = 2 (leftRight / side-by-side) or 0 (monoscopic)
    let mut st3d = Vec::with_capacity(13);
    st3d.extend_from_slice(&13u32.to_be_bytes());
    st3d.extend_from_slice(b"st3d");
    st3d.extend_from_slice(&[0, 0, 0, 0]); // version + flags
    st3d.push(if stereo { 2 } else { 0 });

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
    inject
}

/// Inject Apple Projected Media Profile (VR180 APMP) — `vexu` + `hfov`.
///
/// What a Vision Pro needs to play the file as VR180: both eye views, their
/// side-by-side packing, the half-equirectangular projection and the 180°
/// horizontal field of view. The camera baseline is optional for projected
/// media and the player does not use it, so none is written. (Up to 2.5.5
/// a `cams/blin` box was written here, but directly under `vexu` and not in
/// Apple's format, so readers skipped it.)
pub fn inject_apmp_vr180(path: &Path) -> Result<()> {
    inject_atoms_into_visual_sample_entry(path, &apmp_vr180_atoms(true))
}

/// [`inject_apmp_vr180`] for a frame that holds ONE eye: the projection and
/// the 180° field of view, with no eye views and no packing (monoscopic
/// projected media).
pub fn inject_apmp_vr180_mono(path: &Path) -> Result<()> {
    inject_atoms_into_visual_sample_entry(path, &apmp_vr180_atoms(false))
}

/// The `vexu` and `hfov` boxes of a VR180 file — two eyes side by side
/// (`stereo`), or one eye as the whole frame — laid out as in
/// Apple's "QuickTime and ISO Base Media File Formats and Spatial and
/// Immersive Media" (the same bytes Apple's own writer produces).
fn apmp_vr180_atoms(stereo: bool) -> Vec<u8> {
    /// A box holding `payload`.
    fn atom(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + payload.len());
        b.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
        b.extend_from_slice(tag);
        b.extend_from_slice(payload);
        b
    }
    /// A full box (version 0, no flags) holding `payload`.
    fn full_atom(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = vec![0u8; 4];
        body.extend_from_slice(payload);
        atom(tag, &body)
    }

    // eyes ▷ stri: the frame holds a left and a right eye view (0x03).
    let eyes = atom(b"eyes", &full_atom(b"stri", &[0x03]));
    // proj ▷ prji: half-equirectangular.
    let proj = atom(b"proj", &full_atom(b"prji", b"hequ"));
    // pack ▷ pkin: the two views sit side by side.
    let pack = atom(b"pack", &full_atom(b"pkin", b"side"));
    let boxes = if stereo { [eyes, proj, pack].concat() } else { proj };
    let mut out = atom(b"vexu", &boxes);
    // hfov: 180° in thousandths of a degree (a plain box, no version/flags).
    out.extend_from_slice(&atom(b"hfov", &180_000u32.to_be_bytes()));
    out
}

/// The lens of a lens frame, as an Apple Projected Media Profile file states
/// it (parametric immersive projection, lens algorithm `prim`): the model of
/// Mei and Rives, centred in the view. A ray `(X, Y, Z)` — x right, y down,
/// z along the axis, unit length — lands on the plane point
/// `m = (X, Y) / (Z + xi)`, is scaled by `1 + k1·|m|² + k2·|m|⁴`, and reaches
/// the pixel `(width/2, height/2) + focal · m`, counted from the view's left
/// and top edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApmpLens {
    /// One view's size in pixels (an eye; the whole frame of a single view).
    pub width: u32,
    pub height: u32,
    /// The model's ξ (positive for a fisheye). The file carries it negated:
    /// see [`apmp_lens_atoms`].
    pub xi: f32,
    /// Focal length in pixels of the view.
    pub focal: f32,
    pub k1: f32,
    pub k2: f32,
    /// How far off the axis the lens is stated for, in degrees: the player
    /// builds its surface out to here.
    pub limit_deg: f32,
}

/// Camera baseline written with a two-view lens frame, micrometres. The lens
/// boxes name the baseline as the origin of each lens, so the box has to be
/// there; the player draws both views on one surface and does not move
/// them by it.
pub const APMP_LENS_BASELINE_UM: u32 = 65_000;

/// Inject Apple Projected Media Profile metadata for a LENS FRAME: the
/// parametric immersive projection with the lens in the file, in place of
/// any VR180 metadata. `stereo`: the frame holds a left and a right view
/// side by side, both through `lens`; otherwise one view fills it.
///
/// Apple Vision Pro (visionOS 26 or later) plays such a file immersively at
/// the lens's own angles. Nothing else reads it.
pub fn inject_apmp_lens(path: &Path, lens: &ApmpLens, stereo: bool) -> Result<()> {
    inject_atoms_into_visual_sample_entry(path, &apmp_lens_atoms(lens, stereo))
}

/// The `vexu` box of a lens frame, byte for byte what Apple's writer
/// (AVFoundation, macOS 27) produces from the same lens.
///
/// Three things in it are not what the model's textbook form suggests; each
/// was measured against the system's own player mesh:
/// - the **projection offset is stored negated** (`-xi`): the system divides
///   by `Z - offset`;
/// - the principal point is counted from the view's edges, so a centred lens
///   is exactly one half;
/// - the player builds ONE surface, from the first lens of the collection —
///   which is why both views are rendered through the same centred lens and
///   the two lens boxes differ in their role alone.
fn apmp_lens_atoms(lens: &ApmpLens, stereo: bool) -> Vec<u8> {
    fn atom(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + payload.len());
        b.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
        b.extend_from_slice(tag);
        b.extend_from_slice(payload);
        b
    }
    /// A full box with the given flags word (version in the top byte).
    fn full_atom(tag: &[u8; 4], flags: u32, payload: &[u8]) -> Vec<u8> {
        let mut body = flags.to_be_bytes().to_vec();
        body.extend_from_slice(payload);
        atom(tag, &body)
    }
    // Focal length and principal point are fractions of the view's width /
    // height in 2⁻²⁹ steps, computed in single precision as the writer does.
    const SHIFT: i16 = 29;
    let fixed = |v: f32, of: u32| -> i32 { ((v / of as f32) * (1u32 << SHIFT) as f32) as i32 };
    let one_lens = |id: u32, role: &[u8; 4]| -> Vec<u8> {
        let mut lnhd = id.to_be_bytes().to_vec();
        lnhd.extend_from_slice(b"prim"); // the lens algorithm
        lnhd.extend_from_slice(b"colr"); // a lens that captures colour
        lnhd.extend_from_slice(role);
        let mut rdim = lens.width.to_be_bytes().to_vec();
        rdim.extend_from_slice(&lens.height.to_be_bytes());
        // k1, k2, then no tangential terms, then the angle limit (flag 1).
        let mut ldst = Vec::with_capacity(20);
        for v in [lens.k1, lens.k2, 0.0, 0.0, lens.limit_deg] {
            ldst.extend_from_slice(&v.to_be_bytes());
        }
        // No frame adjustment: x' = 0 + 1·x + 0·x³, the same for y.
        let mut lfad = Vec::with_capacity(24);
        for v in [0.0_f32, 1.0, 0.0, 0.0, 1.0, 0.0] {
            lfad.extend_from_slice(&v.to_be_bytes());
        }
        // Flags 1 | 2: a vertical focal length and skew follow, then the
        // projection offset.
        let mut lnin = SHIFT.to_be_bytes().to_vec();
        lnin.extend_from_slice(&0_i16.to_be_bytes()); // skew denominator shift
        lnin.extend_from_slice(&fixed(lens.focal, lens.width).to_be_bytes());
        lnin.extend_from_slice(&fixed(lens.width as f32 * 0.5, lens.width).to_be_bytes());
        lnin.extend_from_slice(&fixed(lens.height as f32 * 0.5, lens.height).to_be_bytes());
        lnin.extend_from_slice(&fixed(lens.focal, lens.height).to_be_bytes());
        lnin.extend_from_slice(&0_i32.to_be_bytes()); // skew
        lnin.extend_from_slice(&(-lens.xi).to_be_bytes());
        // The lens sits half the camera baseline from the system's origin.
        let lnex = atom(b"lnex", &full_atom(b"corg", 0, b"blin"));
        atom(b"lens", &[
            full_atom(b"lnhd", 0, &lnhd),
            full_atom(b"rdim", 0, &rdim),
            full_atom(b"ldst", 1, &ldst),
            full_atom(b"lfad", 0, &lfad),
            full_atom(b"lnin", 3, &lnin),
            lnex,
        ].concat())
    };
    let proj = atom(b"proj", &full_atom(b"prji", 0, b"prim"));
    let boxes = if stereo {
        let blin = full_atom(b"blin", 0, &APMP_LENS_BASELINE_UM.to_be_bytes());
        let eyes = atom(b"eyes", &[full_atom(b"stri", 0, &[0x03]), atom(b"cams", &blin)].concat());
        let pack = atom(b"pack", &full_atom(b"pkin", 0, b"side"));
        let lnsc = atom(b"lnsc", &[one_lens(1, b"left"), one_lens(2, b"rght")].concat());
        [eyes, proj, pack, lnsc].concat()
    } else {
        [proj, atom(b"lnsc", &one_lens(1, b"mono"))].concat()
    };
    atom(b"vexu", &boxes)
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

#[cfg(test)]
mod apmp_lens_tests {
    use super::{apmp_lens_atoms, ApmpLens};

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// Two views side by side through one lens: the bytes AVFoundation's
    /// writer produced from the same numbers (a 4912 × 4912 view, focal
    /// 7223.1, ξ 2.45543, k1 1.2035, k2 0.0249, stated to 89°), a file the
    /// system reads as parametric immersive and stereo, and whose player
    /// mesh follows this lens to a thousandth of a pixel.
    #[test]
    fn a_two_view_lens_frame_is_written_as_apples_writer_writes_it() {
        let lens = ApmpLens { width: 4912, height: 4912, xi: 2.45543, focal: 7223.1, k1: 1.2035, k2: 0.0249, limit_deg: 89.0 };
        let want = hex(concat!(
            "000001e576657875",
            "0000002d65796573", "0000000d737472690000000003",
            "0000001863616d73", "00000010626c696e000000000000fde8",
            "0000001870726f6a", "0000001070726a69000000007072696d",
            "000000187061636b", "00000010706b696e0000000073696465",
            "000001806c6e7363",
            "000000bc6c656e73",
            "0000001c6c6e686400000000000000017072696d636f6c726c656674",
            "000000147264696d000000000000133000001330",
            "000000206c647374000000013f9a0c4a3ccbfb16000000000000000042b20000",
            "000000246c66616400000000000000003f80000000000000000000003f80000000000000",
            "000000286c6e696e00000003001d00002f0e57c010000000100000002f0e57c000000000c01d25c4",
            "000000186c6e657800000010636f726700000000626c696e",
            "000000bc6c656e73",
            "0000001c6c6e686400000000000000027072696d636f6c7272676874",
            "000000147264696d000000000000133000001330",
            "000000206c647374000000013f9a0c4a3ccbfb16000000000000000042b20000",
            "000000246c66616400000000000000003f80000000000000000000003f80000000000000",
            "000000286c6e696e00000003001d00002f0e57c010000000100000002f0e57c000000000c01d25c4",
            "000000186c6e657800000010636f726700000000626c696e",
        ));
        assert_eq!(apmp_lens_atoms(&lens, true), want);
    }

    /// One view as the whole frame: the projection and one lens, no eye
    /// views, no packing, no baseline. Again the writer's own bytes (512 ×
    /// 512, focal 752.875).
    #[test]
    fn a_single_view_lens_frame_is_written_as_apples_writer_writes_it() {
        let lens = ApmpLens { width: 512, height: 512, xi: 2.45543, focal: 752.875, k1: 1.2035, k2: 0.0249, limit_deg: 89.0 };
        let want = hex(concat!(
            "000000e476657875",
            "0000001870726f6a", "0000001070726a69000000007072696d",
            "000000c46c6e7363",
            "000000bc6c656e73",
            "0000001c6c6e686400000000000000017072696d636f6c726d6f6e6f",
            "000000147264696d000000000000020000000200",
            "000000206c647374000000013f9a0c4a3ccbfb16000000000000000042b20000",
            "000000246c66616400000000000000003f80000000000000000000003f80000000000000",
            "000000286c6e696e00000003001d00002f0e000010000000100000002f0e000000000000c01d25c4",
            "000000186c6e657800000010636f726700000000626c696e",
        ));
        assert_eq!(apmp_lens_atoms(&lens, false), want);
    }

    /// The offset in the file is the model's ξ negated, and nothing of the
    /// half-equirect signalling rides along.
    #[test]
    fn the_projection_offset_is_stored_negated() {
        let lens = ApmpLens { width: 3712, height: 3712, xi: 2.45543, focal: 5458.0, k1: 1.2, k2: 0.02, limit_deg: 68.0 };
        for stereo in [true, false] {
            let got = apmp_lens_atoms(&lens, stereo);
            let at = got.windows(4).position(|w| w == b"lnin").expect("an intrinsics box");
            // tag, flags, two shifts, five 32-bit numbers, then the offset.
            let offset = f32::from_be_bytes(got[at + 4 + 4 + 4 + 20..at + 4 + 4 + 4 + 24].try_into().unwrap());
            assert_eq!(offset, -2.45543);
            assert!(!got.windows(4).any(|w| w == b"hequ" || w == b"hfov"));
            assert_eq!(got.windows(4).filter(|w| *w == b"lens").count(), if stereo { 2 } else { 1 });
        }
    }
}

#[cfg(test)]
mod apmp_tests {
    use super::apmp_vr180_atoms;

    /// The exact bytes: what Apple's reader takes as stereo, side by side,
    /// half-equirectangular, 180° (checked with AVFoundation on a real file).
    #[test]
    fn vr180_boxes_are_laid_out_as_apple_reads_them() {
        let got = apmp_vr180_atoms(true);
        let mut want: Vec<u8> = Vec::new();
        want.extend_from_slice(&[0, 0, 0, 77]); want.extend_from_slice(b"vexu");
        want.extend_from_slice(&[0, 0, 0, 21]); want.extend_from_slice(b"eyes");
        want.extend_from_slice(&[0, 0, 0, 13]); want.extend_from_slice(b"stri");
        want.extend_from_slice(&[0, 0, 0, 0, 0x03]);
        want.extend_from_slice(&[0, 0, 0, 24]); want.extend_from_slice(b"proj");
        want.extend_from_slice(&[0, 0, 0, 16]); want.extend_from_slice(b"prji");
        want.extend_from_slice(&[0, 0, 0, 0]); want.extend_from_slice(b"hequ");
        want.extend_from_slice(&[0, 0, 0, 24]); want.extend_from_slice(b"pack");
        want.extend_from_slice(&[0, 0, 0, 16]); want.extend_from_slice(b"pkin");
        want.extend_from_slice(&[0, 0, 0, 0]); want.extend_from_slice(b"side");
        want.extend_from_slice(&[0, 0, 0, 12]); want.extend_from_slice(b"hfov");
        want.extend_from_slice(&180_000u32.to_be_bytes());
        assert_eq!(got, want);
        // No camera baseline: optional for projected media, and a box that
        // is not in Apple's format is worse than none.
        assert!(!got.windows(4).any(|w| w == b"cams" || w == b"blin"));
    }

    /// One eye as the whole frame: the projection and the field of view
    /// alone — no eye views, no packing.
    #[test]
    fn a_single_eye_frame_declares_no_views_and_no_packing() {
        let got = apmp_vr180_atoms(false);
        let mut want: Vec<u8> = Vec::new();
        want.extend_from_slice(&[0, 0, 0, 32]); want.extend_from_slice(b"vexu");
        want.extend_from_slice(&[0, 0, 0, 24]); want.extend_from_slice(b"proj");
        want.extend_from_slice(&[0, 0, 0, 16]); want.extend_from_slice(b"prji");
        want.extend_from_slice(&[0, 0, 0, 0]); want.extend_from_slice(b"hequ");
        want.extend_from_slice(&[0, 0, 0, 12]); want.extend_from_slice(b"hfov");
        want.extend_from_slice(&180_000u32.to_be_bytes());
        assert_eq!(got, want);
        // The Spherical Video boxes differ in the stereo mode byte alone.
        let (two, one) = (super::youtube_vr180_atoms(true), super::youtube_vr180_atoms(false));
        assert_eq!((two.len(), two[12], one[12]), (one.len(), 2, 0));
        assert_eq!(two[13..], one[13..]);
    }
}
