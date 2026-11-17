//! Linear DNG (DNG 1.4 LinearRaw, 3×16-bit) writer.
//!
//! Writes a camera-linear RGB image as a DNG that Adobe Camera Raw /
//! Lightroom, Adobe DNG Converter, macOS ImageIO (Finder, Preview, `sips`)
//! and exiftool all accept.  The tag recipe is the one validated by the
//! Python reference writer (`lindng.py --layout thumb --comp 7 --tile 256x256
//! --ljpeg-ids 0 --ljpeg-per-comp-dht`):
//!
//! * little-endian TIFF; IFD0 = 8-bit RGB thumbnail (NewSubfileType 1,
//!   uncompressed strips) carrying the camera / colour tags (Make, Model,
//!   Software, DateTime, DNGVersion 1.4.0.0, DNGBackwardVersion 1.1.0.0,
//!   UniqueCameraModel, ColorMatrix1, CalibrationIlluminant1, AsShotNeutral,
//!   BaselineExposure, optional BaselineNoise / BaselineSharpness /
//!   ProfileHueSatMap*), `SubIFDs` → the main image, `ExifIFD` → EXIF;
//! * main IFD: NewSubfileType 0, 3×16-bit, PhotometricInterpretation 34892
//!   (LinearRaw), Compression 7 (lossless JPEG, 256×256 tiles, partial edge
//!   tiles padded by edge replication) or 1 (uncompressed, 64-row strips),
//!   BlackLevel 0, WhiteLevel, DefaultScale/CropOrigin/CropSize, ActiveArea;
//!   no CFA tags, no opcode lists;
//! * EXIF IFD: ExposureTime, FNumber, ISO, ExifVersion, DateTimeOriginal /
//!   DateTimeDigitized, FocalLength, FocalLengthIn35mmFormat.
//!
//! IFD entries are sorted by tag, values of four bytes or fewer are inlined,
//! out-of-line values and every data chunk start 4-byte aligned, and the file
//! is laid out in two passes (sizes first, pointers patched second) exactly
//! like the reference.  Tile data is streamed: tiles are encoded on all
//! cores (`std::thread::scope`), written in order as they complete, and the
//! `TileOffsets` / `TileByteCounts` arrays are patched in place afterwards,
//! so the compressed output is never held in memory as a whole (the window
//! of tiles in flight is bounded).
//!
//! Note: ImageIO refuses DNG files smaller than 102400 bytes; tiny images
//! write fine but will not preview on macOS.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Condvar, Mutex};

use crate::ljpeg;

/// Tile edge for [`DngCompression::LosslessJpegTiles`].
pub const TILE_SIZE: u32 = 256;
/// Rows per strip for [`DngCompression::Uncompressed`].
pub const ROWS_PER_STRIP: u32 = 64;
/// Widest thumbnail accepted / generated (DNG recommends ≤ 256 px).
pub const MAX_THUMBNAIL_WIDTH: u32 = 256;

/// Camera and colour metadata written into the DNG.
#[derive(Clone, Debug)]
pub struct LinearDngMeta {
    pub make: String,
    pub model: String,
    pub unique_camera_model: String,
    pub software: String,
    /// XYZ → camera, row-major 3×3 (ColorMatrix1).
    pub color_matrix1: [f32; 9],
    /// EXIF light-source code (21 = D65).
    pub calibration_illuminant1: u16,
    pub as_shot_neutral: [f32; 3],
    pub baseline_exposure: f32,
    pub baseline_noise: Option<f32>,
    pub baseline_sharpness: Option<f32>,
    /// `[hue divisions, saturation divisions, value divisions]`.
    pub profile_hue_sat_map_dims: Option<[u32; 3]>,
    /// `dims[0] * dims[1] * dims[2] * 3` floats (hue shift, sat scale, val scale).
    pub profile_hue_sat_map_data1: Option<Vec<f32>>,
    pub profile_hue_sat_map_encoding: Option<u32>,
    /// Camera-profile name shown by raw developers (ACR: the "Embedded"
    /// profile entry). Written when `Some`.
    pub profile_name: Option<String>,
    /// `ProfileEmbedPolicy` (0 = allow copying). Written when `Some`.
    pub profile_embed_policy: Option<u32>,
    /// `ProfileToneCurve`: linear-in → linear-out `(x, y)` pairs, first
    /// `(0,0)`, last `(1,1)`, monotone. REPLACES the developer's default tone
    /// curve (ACR and Apple both honour it) — write it only when a specific
    /// tone is intended.
    pub profile_tone_curve: Option<Vec<[f32; 2]>>,
    /// `ProfileLookTable` (dims hue×sat×val, data (hueShift°, satScale,
    /// valScale) per node in val/hue/sat nesting, encoding 0 = linear).
    /// Applied by DNG-SDK developers after exposure, before the tone curve.
    pub profile_look_table: Option<LookTable>,
    pub exposure_s: Option<f64>,
    pub f_number: Option<f64>,
    pub iso: Option<u32>,
    pub focal_mm: Option<f64>,
    pub focal_35mm: Option<u32>,
    /// `"YYYY:MM:DD HH:MM:SS"`; also used for IFD0 `DateTime`.
    pub datetime_original: Option<String>,
    /// `UserComment` text (written with the `ASCII\0\0\0` prefix). The X6
    /// still's comment carries the factory lens strings — provenance of
    /// the projection.
    pub user_comment: Option<String>,
    pub exposure_program: Option<u16>,
    pub metering_mode: Option<u16>,
    pub white_balance: Option<u16>,
    pub exposure_compensation: Option<f64>,
    pub shutter_speed_value: Option<f64>,
    pub image_description: Option<String>,
    /// An XMP packet (tag 700), e.g. a raw converter's own settings for the
    /// picture (`crs:` properties), which it then opens the file with.
    pub xmp: Option<String>,
}

/// `ProfileLookTable` payload — see [`LinearDngMeta::profile_look_table`].
#[derive(Clone, Debug, PartialEq)]
pub struct LookTable {
    pub dims: [u32; 3],
    pub data: Vec<f32>,
    pub encoding: u32,
}

/// Main-image encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DngCompression {
    /// Compression 1, 64-row strips, little-endian u16.
    Uncompressed,
    /// Compression 7, 256×256 lossless-JPEG tiles (see [`crate::ljpeg`]).
    LosslessJpegTiles,
}

/// The image to write: interleaved camera-linear RGB, `width * height * 3`
/// samples in `0..=65535`.
pub struct LinearDngImage<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb16: &'a [u16],
    /// Code value of scene white (normally 65535).
    pub white_level: u32,
}

impl fmt::Debug for LinearDngImage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearDngImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("samples", &self.rgb16.len())
            .field("white_level", &self.white_level)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TIFF primitives
// ---------------------------------------------------------------------------

const TYPE_BYTE: u16 = 1;
const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;
const TYPE_UNDEFINED: u16 = 7;
const TYPE_SRATIONAL: u16 = 10;
const TYPE_FLOAT: u16 = 11;

/// TIFF / EXIF / DNG tag numbers used here.
mod tag {
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const MAKE: u16 = 271;
    pub const MODEL: u16 = 272;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const ORIENTATION: u16 = 274;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const PLANAR_CONFIGURATION: u16 = 284;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const SOFTWARE: u16 = 305;
    pub const XMP: u16 = 700;
    pub const DATE_TIME: u16 = 306;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SUB_IFDS: u16 = 330;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const EXIF_IFD: u16 = 34665;
    pub const EXPOSURE_TIME: u16 = 33434;
    pub const F_NUMBER: u16 = 33437;
    pub const ISO: u16 = 34855;
    pub const EXIF_VERSION: u16 = 36864;
    pub const DATE_TIME_ORIGINAL: u16 = 36867;
    pub const DATE_TIME_DIGITIZED: u16 = 36868;
    pub const FOCAL_LENGTH: u16 = 37386;
    pub const FOCAL_LENGTH_35MM: u16 = 41989;
    pub const EXPOSURE_PROGRAM: u16 = 34850;
    pub const SHUTTER_SPEED_VALUE: u16 = 37377;
    pub const EXPOSURE_COMPENSATION: u16 = 37380;
    pub const METERING_MODE: u16 = 37383;
    pub const USER_COMMENT: u16 = 37510;
    pub const WHITE_BALANCE: u16 = 41987;
    pub const DNG_VERSION: u16 = 50706;
    pub const DNG_BACKWARD_VERSION: u16 = 50707;
    pub const UNIQUE_CAMERA_MODEL: u16 = 50708;
    pub const BLACK_LEVEL: u16 = 50714;
    pub const WHITE_LEVEL: u16 = 50717;
    pub const DEFAULT_SCALE: u16 = 50718;
    pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
    pub const DEFAULT_CROP_SIZE: u16 = 50720;
    pub const COLOR_MATRIX1: u16 = 50721;
    pub const AS_SHOT_NEUTRAL: u16 = 50728;
    pub const BASELINE_EXPOSURE: u16 = 50730;
    pub const BASELINE_NOISE: u16 = 50731;
    pub const BASELINE_SHARPNESS: u16 = 50732;
    pub const CALIBRATION_ILLUMINANT1: u16 = 50778;
    pub const ACTIVE_AREA: u16 = 50829;
    pub const PROFILE_HUE_SAT_MAP_DIMS: u16 = 50937;
    pub const PROFILE_HUE_SAT_MAP_DATA1: u16 = 50938;
    pub const PROFILE_HUE_SAT_MAP_ENCODING: u16 = 51107;
    pub const PROFILE_NAME: u16 = 50936;
    pub const PROFILE_TONE_CURVE: u16 = 50940;
    pub const PROFILE_EMBED_POLICY: u16 = 50941;
    pub const PROFILE_LOOK_TABLE_DIMS: u16 = 50981;
    pub const PROFILE_LOOK_TABLE_DATA: u16 = 50982;
    pub const PROFILE_LOOK_TABLE_ENCODING: u16 = 51108; // 0xC7A4 (DNG 1.4)
}

/// PhotometricInterpretation = LinearRaw.
const PHOTOMETRIC_LINEAR_RAW: u16 = 34892;

#[derive(Clone, Debug)]
struct Entry {
    typ: u16,
    count: u32,
    data: Vec<u8>,
}

/// One TIFF IFD; entries are kept sorted by tag.
#[derive(Clone, Debug, Default)]
struct Ifd {
    entries: BTreeMap<u16, Entry>,
}

impl Ifd {
    fn set(&mut self, tag: u16, typ: u16, count: u32, data: Vec<u8>) {
        self.entries.insert(tag, Entry { typ, count, data });
    }

    /// 7-bit ASCII, NUL-terminated.  Non-ASCII characters become `?` and
    /// embedded NULs are dropped (TIFF ASCII is 7-bit).
    fn set_ascii(&mut self, tag: u16, s: &str) {
        let mut b: Vec<u8> = s
            .chars()
            .filter(|&c| c != '\0')
            .map(|c| if c.is_ascii() { c as u8 } else { b'?' })
            .collect();
        b.push(0);
        let n = b.len() as u32;
        self.set(tag, TYPE_ASCII, n, b);
    }

    fn set_byte(&mut self, tag: u16, v: &[u8]) {
        self.set(tag, TYPE_BYTE, v.len() as u32, v.to_vec());
    }

    fn set_undefined(&mut self, tag: u16, v: &[u8]) {
        self.set(tag, TYPE_UNDEFINED, v.len() as u32, v.to_vec());
    }

    fn set_short(&mut self, tag: u16, v: &[u16]) {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.set(tag, TYPE_SHORT, v.len() as u32, data);
    }

    fn set_long(&mut self, tag: u16, v: &[u32]) {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.set(tag, TYPE_LONG, v.len() as u32, data);
    }

    fn set_rational(&mut self, tag: u16, v: &[(u32, u32)]) {
        let data = v
            .iter()
            .flat_map(|(n, d)| [n.to_le_bytes(), d.to_le_bytes()].concat())
            .collect();
        self.set(tag, TYPE_RATIONAL, v.len() as u32, data);
    }

    fn set_srational(&mut self, tag: u16, v: &[(i32, i32)]) {
        let data = v
            .iter()
            .flat_map(|(n, d)| [n.to_le_bytes(), d.to_le_bytes()].concat())
            .collect();
        self.set(tag, TYPE_SRATIONAL, v.len() as u32, data);
    }

    fn set_float(&mut self, tag: u16, v: &[f32]) {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.set(tag, TYPE_FLOAT, v.len() as u32, data);
    }

    /// Serialize at absolute file offset `base` (4-byte aligned).  Returns the
    /// bytes and, per tag, the absolute offset of its value bytes (inline in
    /// the entry or out of line after the next-IFD pointer, 4-byte aligned).
    fn serialize(&self, base: u32) -> (Vec<u8>, BTreeMap<u16, u32>) {
        let n = self.entries.len() as u32;
        let head = base + 2 + 12 * n + 4;
        let mut ent = Vec::with_capacity(12 * n as usize);
        let mut data: Vec<u8> = Vec::new();
        let mut positions = BTreeMap::new();
        for (i, (&tag, e)) in self.entries.iter().enumerate() {
            ent.extend_from_slice(&tag.to_le_bytes());
            ent.extend_from_slice(&e.typ.to_le_bytes());
            ent.extend_from_slice(&e.count.to_le_bytes());
            if e.data.len() <= 4 {
                positions.insert(tag, base + 2 + 12 * i as u32 + 8);
                let mut v = [0u8; 4];
                v[..e.data.len()].copy_from_slice(&e.data);
                ent.extend_from_slice(&v);
            } else {
                while (head as usize + data.len()) % 4 != 0 {
                    data.push(0);
                }
                let off = head + data.len() as u32;
                positions.insert(tag, off);
                ent.extend_from_slice(&off.to_le_bytes());
                data.extend_from_slice(&e.data);
            }
        }
        let mut out = Vec::with_capacity(6 + ent.len() + data.len());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&ent);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&data);
        (out, positions)
    }

    fn serialized_len(&self, base: u32) -> u32 {
        self.serialize(base).0.len() as u32
    }
}

fn align4(n: u32) -> u32 {
    (n + 3) & !3
}

// ---------------------------------------------------------------------------
// Rational conversion: exact port of Python `Fraction(v).limit_denominator(N)`
// ---------------------------------------------------------------------------

const MAX_DENOMINATOR: i128 = 1 << 20;

/// `v` as an exact rational `m * 2^e` reduced to lowest terms, as
/// `(numerator, denominator)`; `None` for non-finite values.
fn exact_fraction(v: f64) -> Option<(i128, i128)> {
    if !v.is_finite() {
        return None;
    }
    let bits = v.to_bits();
    let neg = (bits >> 63) != 0;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = (bits & ((1u64 << 52) - 1)) as i128;
    let (mut m, mut e) = if exp == 0 { (frac, -1074) } else { (frac | (1i128 << 52), exp - 1075) };
    if m == 0 {
        return Some((0, 1));
    }
    while m % 2 == 0 && e < 0 {
        m /= 2;
        e += 1;
    }
    let (num, den) = if e >= 0 {
        // |v| >= 2^52 here; cap the shift so the product stays in range (such
        // values are clamped to the TIFF field width later anyway).
        (m << e.min(70), 1i128)
    } else if e < -100 {
        // |v| < 2^-48: the closest fraction with a small denominator is 0.
        (0, 1)
    } else {
        (m, 1i128 << (-e))
    };
    Some((if neg { -num } else { num }, den))
}

/// Python 3.12+ `Fraction.limit_denominator` on `n / d` (`d > 0`, lowest
/// terms): the closest fraction with denominator ≤ `max_den`, ties resolved
/// the way CPython does.
fn limit_denominator(n: i128, d: i128, max_den: i128) -> (i128, i128) {
    if d <= max_den {
        return (n, d);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (n, d);
    let den0 = d;
    loop {
        let a = n.div_euclid(d); // Python floor division (d > 0)
        let q2 = q0 + a * q1;
        if q2 > max_den {
            break;
        }
        let p2 = p0 + a * p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let r = n - a * d;
        n = d;
        d = r;
    }
    let k = (max_den - q0).div_euclid(q1);
    if 2 * d * (q0 + k * q1) <= den0 {
        (p1, q1)
    } else {
        (p0 + k * p1, q0 + k * q1)
    }
}

/// `v` → `(num, den)` with `den ≤ 2^20`, as `lindng.py`'s `rat()`.  The
/// denominator bound is lowered until the numerator fits `limit`.
fn rat_bounded(v: f64, limit: i128) -> (i128, i128) {
    let Some((n, d)) = exact_fraction(v) else {
        return (0, 1);
    };
    let mut max_den = MAX_DENOMINATOR;
    loop {
        let (p, q) = limit_denominator(n, d, max_den);
        if p.abs() <= limit || max_den == 1 {
            return (p.clamp(-limit, limit), q.max(1));
        }
        max_den = (max_den / 2).max(1);
    }
}

/// Unsigned RATIONAL; negative values become 0/1.
fn urat(v: f64) -> (u32, u32) {
    let (n, d) = rat_bounded(v, u32::MAX as i128);
    if n <= 0 {
        (0, 1)
    } else {
        (n as u32, d as u32)
    }
}

/// Signed SRATIONAL.
fn srat(v: f64) -> (i32, i32) {
    let (n, d) = rat_bounded(v, i32::MAX as i128);
    (n as i32, d as i32)
}

// ---------------------------------------------------------------------------
// Output stream with position tracking
// ---------------------------------------------------------------------------

struct Out<W: Write + Seek> {
    w: W,
    pos: u64,
}

impl<W: Write + Seek> Out<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<()> {
        self.w.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }

    fn pad_to(&mut self, target: u64) -> io::Result<()> {
        if target < self.pos {
            return Err(io::Error::other("dng: layout went backwards"));
        }
        const ZEROS: [u8; 64] = [0; 64];
        let mut left = target - self.pos;
        while left > 0 {
            let n = left.min(ZEROS.len() as u64) as usize;
            self.write(&ZEROS[..n])?;
            left -= n as u64;
        }
        Ok(())
    }

    fn align4(&mut self) -> io::Result<()> {
        let t = (self.pos + 3) & !3;
        self.pad_to(t)
    }

    /// Append one tile / strip (4-byte aligned) and record its offset + count.
    fn put_chunk(&mut self, bytes: &[u8], offsets: &mut Vec<u32>, counts: &mut Vec<u32>) -> io::Result<()> {
        self.align4()?;
        let off = u32::try_from(self.pos).map_err(|_| too_large())?;
        let len = u32::try_from(bytes.len()).map_err(|_| too_large())?;
        self.pos.checked_add(bytes.len() as u64).filter(|&p| p <= u32::MAX as u64).ok_or_else(too_large)?;
        self.write(bytes)?;
        offsets.push(off);
        counts.push(len);
        Ok(())
    }

    /// Overwrite `values` (LONG array) at absolute offset `at`.
    fn patch_longs(&mut self, at: u32, values: &[u32]) -> io::Result<()> {
        self.w.seek(SeekFrom::Start(at as u64))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.w.write_all(&bytes)
    }
}

fn too_large() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "dng: output exceeds the 4 GiB TIFF limit")
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, format!("dng: {msg}"))
}

// ---------------------------------------------------------------------------
// Thumbnail
// ---------------------------------------------------------------------------

/// 8-bit RGB thumbnail, at most [`MAX_THUMBNAIL_WIDTH`] wide, box-filtered
/// from the image, then white-balanced by AsShotNeutral, lifted by
/// BaselineExposure and gamma-encoded (1/2.2) so it previews sensibly.
fn generate_thumbnail(img: &LinearDngImage, meta: &LinearDngMeta) -> (u32, u32, Vec<u8>) {
    let (w, h) = (img.width as u64, img.height as u64);
    let tw = w.min(MAX_THUMBNAIL_WIDTH as u64);
    let th = ((h * tw + w / 2) / w).max(1);
    let (twu, thu) = (tw as usize, th as usize);
    let mut acc = vec![0u64; twu * thu * 3];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, thu);
    let band = thu.div_ceil(nthreads);
    std::thread::scope(|s| {
        for (b, chunk) in acc.chunks_mut(band * twu * 3).enumerate() {
            s.spawn(move || {
                let ty0 = b * band;
                for (i, cell_row) in chunk.chunks_mut(twu * 3).enumerate() {
                    let ty = (ty0 + i) as u64;
                    let (ya, yb) = (ty * h / th, (ty + 1) * h / th);
                    for y in ya..yb {
                        let row = &img.rgb16[(y * w * 3) as usize..((y + 1) * w * 3) as usize];
                        for tx in 0..tw {
                            let (xa, xb) = ((tx * w / tw) as usize, ((tx + 1) * w / tw) as usize);
                            let mut sum = [0u64; 3];
                            for px in row[xa * 3..xb * 3].chunks_exact(3) {
                                sum[0] += px[0] as u64;
                                sum[1] += px[1] as u64;
                                sum[2] += px[2] as u64;
                            }
                            let cell = &mut cell_row[tx as usize * 3..tx as usize * 3 + 3];
                            cell[0] += sum[0];
                            cell[1] += sum[1];
                            cell[2] += sum[2];
                        }
                    }
                }
            });
        }
    });
    let white = img.white_level.max(1) as f64;
    let lift = 2f64.powf(meta.baseline_exposure as f64) / white;
    let gain: [f64; 3] = std::array::from_fn(|c| {
        let n = meta.as_shot_neutral[c] as f64;
        lift / if n > 0.0 && n.is_finite() { n } else { 1.0 }
    });
    let mut out = Vec::with_capacity(twu * thu * 3);
    for ty in 0..th {
        let rows = ((ty + 1) * h / th - ty * h / th).max(1);
        for tx in 0..tw {
            let cols = ((tx + 1) * w / tw - tx * w / tw).max(1);
            let count = (rows * cols) as f64;
            let cell = &acc[((ty * tw + tx) * 3) as usize..((ty * tw + tx) * 3 + 3) as usize];
            for c in 0..3 {
                let v = (cell[c] as f64 / count * gain[c]).clamp(0.0, 1.0).powf(1.0 / 2.2);
                out.push((v * 255.0).round() as u8);
            }
        }
    }
    (tw as u32, th as u32, out)
}

// ---------------------------------------------------------------------------
// Main image data
// ---------------------------------------------------------------------------

/// Copy tile `k` of the grid into `buf` (`TILE_SIZE²×3`), replicating the
/// last valid column / row into the padding of partial edge tiles.
fn gather_tile(img: &LinearDngImage, k: usize, tiles_x: usize, buf: &mut [u16]) {
    let t = TILE_SIZE as usize;
    let (w, h) = (img.width as usize, img.height as usize);
    let (x0, y0) = ((k % tiles_x) * t, (k / tiles_x) * t);
    let vis_w = t.min(w - x0);
    let vis_h = t.min(h - y0);
    for r in 0..t {
        let sy = y0 + r.min(vis_h - 1);
        let src = &img.rgb16[(sy * w + x0) * 3..(sy * w + x0 + vis_w) * 3];
        let dst = &mut buf[r * t * 3..(r + 1) * t * 3];
        dst[..vis_w * 3].copy_from_slice(src);
        if vis_w < t {
            let last = [src[vis_w * 3 - 3], src[vis_w * 3 - 2], src[vis_w * 3 - 1]];
            for px in dst[vis_w * 3..].chunks_exact_mut(3) {
                px.copy_from_slice(&last);
            }
        }
    }
}

/// Encode all tiles in parallel and stream them to `out` in tile order.
///
/// Workers claim tile indices from an atomic counter but may not run more
/// than `window` tiles ahead of the writer, which bounds memory; completed
/// tiles arrive over a channel and are re-ordered in a small map.
fn write_tiles<W: Write + Seek>(
    out: &mut Out<W>,
    img: &LinearDngImage,
    tiles_x: usize,
    tiles_y: usize,
    offsets: &mut Vec<u32>,
    counts: &mut Vec<u32>,
) -> io::Result<()> {
    let ntiles = tiles_x * tiles_y;
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, ntiles);
    let window = nthreads * 4;
    let next_claim = AtomicUsize::new(0);
    let written = Mutex::new(0usize);
    let cv = Condvar::new();
    let (tx, rx) = mpsc::channel::<(usize, Vec<u8>)>();
    std::thread::scope(|s| {
        for _ in 0..nthreads {
            let tx = tx.clone();
            let (next_claim, written, cv) = (&next_claim, &written, &cv);
            s.spawn(move || {
                let t = TILE_SIZE as usize;
                let mut tile = vec![0u16; t * t * 3];
                let mut diffs: Vec<i16> = Vec::with_capacity(t * t * 3);
                loop {
                    let k = next_claim.fetch_add(1, Ordering::Relaxed);
                    if k >= ntiles {
                        break;
                    }
                    {
                        let mut g = written.lock().unwrap_or_else(|e| e.into_inner());
                        while k >= *g + window {
                            g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                        }
                    }
                    gather_tile(img, k, tiles_x, &mut tile);
                    let mut enc = Vec::new();
                    ljpeg::encode_tile_rgb16_into(&tile, t, t, &mut diffs, &mut enc);
                    if tx.send((k, enc)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut next = 0usize;
        let mut res = Ok(());
        'outer: while next < ntiles {
            let (k, enc) = match rx.recv() {
                Ok(v) => v,
                Err(_) => {
                    res = Err(io::Error::other("dng: tile encoder stopped early"));
                    break;
                }
            };
            pending.insert(k, enc);
            while let Some(enc) = pending.remove(&next) {
                if let Err(e) = out.put_chunk(&enc, offsets, counts) {
                    res = Err(e);
                    break 'outer;
                }
                next += 1;
                *written.lock().unwrap_or_else(|e| e.into_inner()) = next;
                cv.notify_all();
            }
        }
        // Error path: unpark any worker waiting on the window; with the
        // receiver dropped their next send fails and they exit.
        *written.lock().unwrap_or_else(|e| e.into_inner()) = usize::MAX / 2;
        cv.notify_all();
        drop(rx);
        res
    })
}

/// Stream the image as uncompressed little-endian 64-row strips.
fn write_strips<W: Write + Seek>(
    out: &mut Out<W>,
    img: &LinearDngImage,
    offsets: &mut Vec<u32>,
    counts: &mut Vec<u32>,
) -> io::Result<()> {
    let (w, h) = (img.width as usize, img.height as usize);
    let rps = ROWS_PER_STRIP as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(rps * w * 6);
    for y0 in (0..h).step_by(rps) {
        let rows = (h - y0).min(rps);
        let src = &img.rgb16[y0 * w * 3..(y0 + rows) * w * 3];
        buf.clear();
        buf.resize(src.len() * 2, 0);
        for (dst, &v) in buf.chunks_exact_mut(2).zip(src) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        out.put_chunk(&buf, offsets, counts)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write `image` as a linear DNG at `path`.
///
/// `thumbnail`: `(width, height, rgb8)` with `width ≤ 256` and
/// `rgb8.len() == width * height * 3`; `None` generates a 256-wide
/// box-filtered thumbnail from the image (white balance + BaselineExposure +
/// gamma 1/2.2).  On error the partially written file is removed.
pub fn write_linear_dng(
    path: &Path,
    image: &LinearDngImage,
    meta: &LinearDngMeta,
    compression: DngCompression,
    thumbnail: Option<(u32, u32, &[u8])>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let w = BufWriter::with_capacity(1 << 20, file);
    match write_linear_dng_to(w, image, meta, compression, thumbnail) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(path);
            Err(e)
        }
    }
}

/// [`write_linear_dng`] into any seekable writer (the stream is flushed).
pub fn write_linear_dng_to<W: Write + Seek>(
    w: W,
    image: &LinearDngImage,
    meta: &LinearDngMeta,
    compression: DngCompression,
    thumbnail: Option<(u32, u32, &[u8])>,
) -> io::Result<()> {
    // ---- validation ------------------------------------------------------
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 {
        return Err(invalid("image dimensions must be non-zero"));
    }
    let expect = (width as u64) * (height as u64) * 3;
    if image.rgb16.len() as u64 != expect {
        return Err(invalid(&format!(
            "rgb16 has {} samples, expected {}x{}x3 = {}",
            image.rgb16.len(),
            width,
            height,
            expect
        )));
    }
    if let Some((tw, th, bytes)) = thumbnail {
        if tw == 0 || th == 0 || tw > MAX_THUMBNAIL_WIDTH {
            return Err(invalid("thumbnail must be 1..=256 px wide and non-empty"));
        }
        if bytes.len() as u64 != tw as u64 * th as u64 * 3 {
            return Err(invalid("thumbnail byte length must be width*height*3"));
        }
    }
    if let Some(lt) = &meta.profile_look_table {
        let n = lt.dims.iter().map(|&d| d as u64).product::<u64>() * 3;
        if lt.data.len() as u64 != n || lt.dims[0] < 1 || lt.dims[1] < 2 || lt.dims[2] < 1 {
            return Err(invalid(&format!("ProfileLookTable has {} floats, dims {:?} need {} (hue≥1, sat≥2, val≥1)", lt.data.len(), lt.dims, n)));
        }
    }
    if let Some(tc) = &meta.profile_tone_curve {
        let ok = tc.len() >= 2 && tc[0] == [0.0, 0.0] && *tc.last().unwrap() == [1.0, 1.0]
            && tc.windows(2).all(|w| w[1][0] > w[0][0] && w[1][1] >= w[0][1]);
        if !ok {
            return Err(invalid("ProfileToneCurve must run (0,0) → (1,1) with increasing x and non-decreasing y"));
        }
    }
    if let (Some(dims), Some(data)) = (&meta.profile_hue_sat_map_dims, &meta.profile_hue_sat_map_data1) {
        let n = dims.iter().map(|&d| d as u64).product::<u64>() * 3;
        if data.len() as u64 != n {
            return Err(invalid(&format!(
                "ProfileHueSatMapData1 has {} floats, dims {:?} need {}",
                data.len(),
                dims,
                n
            )));
        }
    }

    // ---- thumbnail -------------------------------------------------------
    let generated;
    let (thumb_w, thumb_h, thumb_bytes): (u32, u32, &[u8]) = match thumbnail {
        Some((tw, th, b)) => (tw, th, b),
        None => {
            generated = generate_thumbnail(image, meta);
            (generated.0, generated.1, &generated.2)
        }
    };

    // ---- main-image IFD ----------------------------------------------------
    let mut main = Ifd::default();
    main.set_long(tag::NEW_SUBFILE_TYPE, &[0]);
    main.set_long(tag::IMAGE_WIDTH, &[width]);
    main.set_long(tag::IMAGE_LENGTH, &[height]);
    main.set_short(tag::BITS_PER_SAMPLE, &[16, 16, 16]);
    main.set_short(tag::PHOTOMETRIC, &[PHOTOMETRIC_LINEAR_RAW]);
    main.set_short(tag::ORIENTATION, &[1]);
    main.set_short(tag::SAMPLES_PER_PIXEL, &[3]);
    main.set_short(tag::PLANAR_CONFIGURATION, &[1]);
    main.set_short(tag::SAMPLE_FORMAT, &[1, 1, 1]);
    main.set_short(tag::BLACK_LEVEL, &[0, 0, 0]);
    main.set_long(tag::WHITE_LEVEL, &[image.white_level; 3]);
    main.set_rational(tag::DEFAULT_SCALE, &[(1, 1), (1, 1)]);
    main.set_long(tag::DEFAULT_CROP_ORIGIN, &[0, 0]);
    main.set_long(tag::DEFAULT_CROP_SIZE, &[width, height]);
    main.set_long(tag::ACTIVE_AREA, &[0, 0, height, width]);
    let (tiles_x, tiles_y) = (width.div_ceil(TILE_SIZE) as usize, height.div_ceil(TILE_SIZE) as usize);
    let nchunks = match compression {
        DngCompression::LosslessJpegTiles => tiles_x * tiles_y,
        DngCompression::Uncompressed => height.div_ceil(ROWS_PER_STRIP) as usize,
    };
    let (offsets_tag, counts_tag) = match compression {
        DngCompression::LosslessJpegTiles => {
            main.set_short(tag::COMPRESSION, &[7]);
            main.set_long(tag::TILE_WIDTH, &[TILE_SIZE]);
            main.set_long(tag::TILE_LENGTH, &[TILE_SIZE]);
            (tag::TILE_OFFSETS, tag::TILE_BYTE_COUNTS)
        }
        DngCompression::Uncompressed => {
            main.set_short(tag::COMPRESSION, &[1]);
            main.set_long(tag::ROWS_PER_STRIP, &[ROWS_PER_STRIP]);
            (tag::STRIP_OFFSETS, tag::STRIP_BYTE_COUNTS)
        }
    };
    // Placeholders: patched in place once the data has been streamed.
    main.set_long(offsets_tag, &vec![0u32; nchunks]);
    main.set_long(counts_tag, &vec![0u32; nchunks]);

    // ---- EXIF IFD ---------------------------------------------------------
    let mut exif = Ifd::default();
    if let Some(t) = meta.exposure_s {
        exif.set_rational(tag::EXPOSURE_TIME, &[urat(t)]);
    }
    if let Some(f) = meta.f_number {
        exif.set_rational(tag::F_NUMBER, &[urat(f)]);
    }
    if let Some(iso) = meta.iso {
        exif.set_short(tag::ISO, &[iso.min(u16::MAX as u32) as u16]);
    }
    exif.set_undefined(tag::EXIF_VERSION, b"0230");
    if let Some(dt) = &meta.datetime_original {
        exif.set_ascii(tag::DATE_TIME_ORIGINAL, dt);
        exif.set_ascii(tag::DATE_TIME_DIGITIZED, dt);
    }
    if let Some(f) = meta.focal_mm {
        exif.set_rational(tag::FOCAL_LENGTH, &[urat(f)]);
    }
    if let Some(f) = meta.focal_35mm {
        exif.set_short(tag::FOCAL_LENGTH_35MM, &[f.min(u16::MAX as u32) as u16]);
    }
    if let Some(v) = meta.exposure_program { exif.set_short(tag::EXPOSURE_PROGRAM, &[v]); }
    if let Some(v) = meta.metering_mode { exif.set_short(tag::METERING_MODE, &[v]); }
    if let Some(v) = meta.white_balance { exif.set_short(tag::WHITE_BALANCE, &[v]); }
    if let Some(v) = meta.exposure_compensation { exif.set_srational(tag::EXPOSURE_COMPENSATION, &[srat(v)]); }
    if let Some(v) = meta.shutter_speed_value { exif.set_srational(tag::SHUTTER_SPEED_VALUE, &[srat(v)]); }
    if let Some(c) = &meta.user_comment {
        let mut v = b"ASCII\0\0\0".to_vec();
        v.extend(c.bytes().map(|b| if b.is_ascii() && b != 0 { b } else { b'?' }));
        exif.set_undefined(tag::USER_COMMENT, &v);
    }

    // ---- IFD0: thumbnail + camera / colour tags ---------------------------
    let mut ifd0 = Ifd::default();
    ifd0.set_long(tag::NEW_SUBFILE_TYPE, &[1]);
    ifd0.set_long(tag::IMAGE_WIDTH, &[thumb_w]);
    ifd0.set_long(tag::IMAGE_LENGTH, &[thumb_h]);
    ifd0.set_short(tag::BITS_PER_SAMPLE, &[8, 8, 8]);
    ifd0.set_short(tag::COMPRESSION, &[1]);
    ifd0.set_short(tag::PHOTOMETRIC, &[2]);
    ifd0.set_long(tag::STRIP_OFFSETS, &[0]); // patched
    ifd0.set_short(tag::ORIENTATION, &[1]);
    ifd0.set_short(tag::SAMPLES_PER_PIXEL, &[3]);
    ifd0.set_long(tag::ROWS_PER_STRIP, &[thumb_h]);
    ifd0.set_long(tag::STRIP_BYTE_COUNTS, &[thumb_bytes.len() as u32]);
    ifd0.set_short(tag::PLANAR_CONFIGURATION, &[1]);
    ifd0.set_long(tag::SUB_IFDS, &[0]); // patched
    if let Some(d) = &meta.image_description {
        ifd0.set_ascii(tag::IMAGE_DESCRIPTION, d);
    }
    if let Some(x) = &meta.xmp {
        ifd0.set_byte(tag::XMP, x.as_bytes());
    }
    ifd0.set_ascii(tag::MAKE, &meta.make);
    ifd0.set_ascii(tag::MODEL, &meta.model);
    ifd0.set_rational(tag::X_RESOLUTION, &[(72, 1)]);
    ifd0.set_rational(tag::Y_RESOLUTION, &[(72, 1)]);
    ifd0.set_short(tag::RESOLUTION_UNIT, &[2]);
    ifd0.set_ascii(tag::SOFTWARE, &meta.software);
    if let Some(dt) = &meta.datetime_original {
        ifd0.set_ascii(tag::DATE_TIME, dt);
    }
    ifd0.set_long(tag::EXIF_IFD, &[0]); // patched
    ifd0.set_byte(tag::DNG_VERSION, &[1, 4, 0, 0]);
    ifd0.set_byte(tag::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
    ifd0.set_ascii(tag::UNIQUE_CAMERA_MODEL, &meta.unique_camera_model);
    let cm: Vec<(i32, i32)> = meta.color_matrix1.iter().map(|&v| srat(v as f64)).collect();
    ifd0.set_srational(tag::COLOR_MATRIX1, &cm);
    let asn: Vec<(u32, u32)> = meta.as_shot_neutral.iter().map(|&v| urat(v as f64)).collect();
    ifd0.set_rational(tag::AS_SHOT_NEUTRAL, &asn);
    ifd0.set_srational(tag::BASELINE_EXPOSURE, &[srat(meta.baseline_exposure as f64)]);
    if let Some(v) = meta.baseline_noise {
        ifd0.set_rational(tag::BASELINE_NOISE, &[urat(v as f64)]);
    }
    if let Some(v) = meta.baseline_sharpness {
        ifd0.set_rational(tag::BASELINE_SHARPNESS, &[urat(v as f64)]);
    }
    ifd0.set_short(tag::CALIBRATION_ILLUMINANT1, &[meta.calibration_illuminant1]);
    if let (Some(dims), Some(data)) = (&meta.profile_hue_sat_map_dims, &meta.profile_hue_sat_map_data1) {
        ifd0.set_long(tag::PROFILE_HUE_SAT_MAP_DIMS, dims);
        ifd0.set_float(tag::PROFILE_HUE_SAT_MAP_DATA1, data);
        if let Some(enc) = meta.profile_hue_sat_map_encoding {
            ifd0.set_long(tag::PROFILE_HUE_SAT_MAP_ENCODING, &[enc]);
        }
    }
    if let Some(n) = &meta.profile_name { ifd0.set_ascii(tag::PROFILE_NAME, n); }
    if let Some(p) = meta.profile_embed_policy { ifd0.set_long(tag::PROFILE_EMBED_POLICY, &[p]); }
    if let Some(tc) = &meta.profile_tone_curve {
        let flat: Vec<f32> = tc.iter().flat_map(|p| [p[0], p[1]]).collect();
        ifd0.set_float(tag::PROFILE_TONE_CURVE, &flat);
    }
    if let Some(lt) = &meta.profile_look_table {
        ifd0.set_long(tag::PROFILE_LOOK_TABLE_DIMS, &lt.dims);
        ifd0.set_float(tag::PROFILE_LOOK_TABLE_DATA, &lt.data);
        ifd0.set_long(tag::PROFILE_LOOK_TABLE_ENCODING, &[lt.encoding]);
    }

    // ---- pass 1: layout (lengths do not depend on pointer values) ---------
    let ifd0_off: u32 = 8;
    let main_off = align4(ifd0_off + ifd0.serialized_len(ifd0_off));
    let exif_off = align4(main_off + main.serialized_len(main_off));
    let thumb_off = align4(exif_off + exif.serialized_len(exif_off));
    let data_off = thumb_off
        .checked_add(thumb_bytes.len() as u32)
        .map(align4)
        .ok_or_else(too_large)?;

    // ---- pass 2: patch pointers, write header + IFDs + thumbnail ----------
    ifd0.set_long(tag::STRIP_OFFSETS, &[thumb_off]);
    ifd0.set_long(tag::SUB_IFDS, &[main_off]);
    ifd0.set_long(tag::EXIF_IFD, &[exif_off]);

    let mut out = Out { w, pos: 0 };
    out.write(b"II*\0")?;
    out.write(&8u32.to_le_bytes())?;
    let (b, _) = ifd0.serialize(ifd0_off);
    out.write(&b)?;
    out.pad_to(main_off as u64)?;
    let (b, main_pos) = main.serialize(main_off);
    out.write(&b)?;
    out.pad_to(exif_off as u64)?;
    let (b, _) = exif.serialize(exif_off);
    out.write(&b)?;
    out.pad_to(thumb_off as u64)?;
    out.write(thumb_bytes)?;
    out.pad_to(data_off as u64)?;

    // ---- image data, then patch the offset / count arrays -----------------
    let mut offsets = Vec::with_capacity(nchunks);
    let mut counts = Vec::with_capacity(nchunks);
    match compression {
        DngCompression::LosslessJpegTiles => write_tiles(&mut out, image, tiles_x, tiles_y, &mut offsets, &mut counts)?,
        DngCompression::Uncompressed => write_strips(&mut out, image, &mut offsets, &mut counts)?,
    }
    debug_assert_eq!(offsets.len(), nchunks);
    out.patch_longs(main_pos[&offsets_tag], &offsets)?;
    out.patch_longs(main_pos[&counts_tag], &counts)?;
    out.w.flush()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ljpeg::test_support::{ljpeg_decode, synth_image, Rng};
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    // ---- minimal TIFF reader ---------------------------------------------

    const TYPE_SIZE: [usize; 13] = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8];

    struct TEntry {
        typ: u16,
        count: u32,
        data: Vec<u8>,
        value_pos: u32,
    }

    struct TIfd {
        entries: BTreeMap<u16, TEntry>,
        next: u32,
    }

    fn le16(b: &[u8], p: usize) -> u16 {
        u16::from_le_bytes([b[p], b[p + 1]])
    }
    fn le32(b: &[u8], p: usize) -> u32 {
        u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
    }

    /// Parse one IFD, checking the invariants this writer guarantees
    /// (sorted tags, 4-byte aligned out-of-line values).
    fn read_ifd(buf: &[u8], off: u32) -> TIfd {
        read_ifd_opts(buf, off, true)
    }

    /// `strict`: also require 4-byte alignment of out-of-line values (TIFF
    /// itself only requires word alignment; Adobe's files use 2).
    fn read_ifd_opts(buf: &[u8], off: u32, strict: bool) -> TIfd {
        assert_eq!(off % 2, 0, "IFD offset {off} is not word aligned");
        let off = off as usize;
        let n = le16(buf, off) as usize;
        let mut entries = BTreeMap::new();
        let mut last_tag = 0u32;
        for i in 0..n {
            let e = off + 2 + 12 * i;
            let (tag, typ, count, val) = (le16(buf, e), le16(buf, e + 2), le32(buf, e + 4), le32(buf, e + 8));
            assert!(i == 0 || tag as u32 > last_tag, "entries not sorted ascending at tag {tag}");
            last_tag = tag as u32;
            let size = TYPE_SIZE[typ as usize] * count as usize;
            let (data, value_pos) = if size <= 4 {
                (buf[e + 8..e + 8 + size].to_vec(), (e + 8) as u32)
            } else {
                if strict {
                    assert_eq!(val % 4, 0, "out-of-line value of tag {tag} not 4-byte aligned");
                }
                (buf[val as usize..val as usize + size].to_vec(), val)
            };
            entries.insert(tag, TEntry { typ, count, data, value_pos });
        }
        TIfd { entries, next: le32(buf, off + 2 + 12 * n) }
    }

    impl TIfd {
        fn e(&self, tag: u16) -> &TEntry {
            self.entries.get(&tag).unwrap_or_else(|| panic!("tag {tag} missing"))
        }
        fn longs(&self, tag: u16) -> Vec<u32> {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_LONG, "tag {tag} type");
            e.data.chunks_exact(4).map(|c| le32(c, 0)).collect()
        }
        fn shorts(&self, tag: u16) -> Vec<u16> {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_SHORT, "tag {tag} type");
            e.data.chunks_exact(2).map(|c| le16(c, 0)).collect()
        }
        fn bytes(&self, tag: u16, typ: u16) -> Vec<u8> {
            let e = self.e(tag);
            assert_eq!(e.typ, typ, "tag {tag} type");
            e.data.clone()
        }
        fn ascii(&self, tag: u16) -> String {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_ASCII, "tag {tag} type");
            assert_eq!(e.data.last(), Some(&0), "ASCII tag {tag} not NUL-terminated");
            assert_eq!(e.count as usize, e.data.len());
            String::from_utf8(e.data[..e.data.len() - 1].to_vec()).unwrap()
        }
        fn rationals(&self, tag: u16) -> Vec<(u32, u32)> {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_RATIONAL, "tag {tag} type");
            e.data.chunks_exact(8).map(|c| (le32(c, 0), le32(c, 4))).collect()
        }
        fn srationals(&self, tag: u16) -> Vec<(i32, i32)> {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_SRATIONAL, "tag {tag} type");
            e.data.chunks_exact(8).map(|c| (le32(c, 0) as i32, le32(c, 4) as i32)).collect()
        }
        fn floats(&self, tag: u16) -> Vec<f32> {
            let e = self.e(tag);
            assert_eq!(e.typ, TYPE_FLOAT, "tag {tag} type");
            e.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    }

    // ---- fixtures ----------------------------------------------------------

    fn x6_meta() -> LinearDngMeta {
        LinearDngMeta {
            make: "Arashi Vision".into(),
            model: "Insta360 X6".into(),
            unique_camera_model: "Insta360 X6".into(),
            software: "VR180 Silver Bullet dng_write test".into(),
            color_matrix1: [0.71875, -0.1484375, -0.09375, -0.609375, 1.453125, 0.109375, -0.15625, 0.296875, 0.4765625],
            calibration_illuminant1: 21,
            as_shot_neutral: [0.4599609375, 1.0, 0.537109375],
            baseline_exposure: 2.6,
            baseline_noise: Some(1.0),
            baseline_sharpness: Some(1.0),
            profile_hue_sat_map_dims: Some([1, 2, 1]),
            profile_hue_sat_map_data1: Some(vec![1.0, 0.93, 1.0, 1.0, 0.93, 1.0]),
            profile_hue_sat_map_encoding: Some(1),
            profile_name: Some("VR180 Silver Bullet grade".into()),
            profile_embed_policy: Some(0),
            profile_tone_curve: Some(vec![[0.0, 0.0], [0.25, 0.18], [0.5, 0.45], [0.75, 0.74], [1.0, 1.0]]),
            profile_look_table: Some(LookTable { dims: [1, 2, 1], data: vec![0.0, 1.3, 1.0, 0.0, 1.3, 1.0], encoding: 0 }),
            exposure_s: Some(1.0 / 5072.0),
            f_number: Some(2.0),
            iso: Some(120),
            focal_mm: Some(2.54),
            focal_35mm: Some(8),
            datetime_original: Some("2026:10:07 13:31:05".into()),
            user_comment: Some("INSTA3602_2.455430_7214.730".into()),
            exposure_program: Some(2), metering_mode: Some(5), white_balance: Some(0),
            exposure_compensation: Some(0.0), shutter_speed_value: Some(12.31),
            image_description: Some("VR180 left eye".into()),
            xmp: None,
        }
    }

    /// Smooth gradient + deterministic noise + a bright disc: compresses like
    /// a real photograph (keeps the LJPEG file comfortably above ImageIO's
    /// 100 KiB minimum at 512x256).
    fn realish_image(w: usize, h: usize, seed: u64) -> Vec<u16> {
        let mut rng = Rng(seed);
        let mut img = Vec::with_capacity(w * h * 3);
        let (cx, cy, r2) = (w as f64 * 0.6, h as f64 * 0.45, (h as f64 * 0.3).powi(2));
        for y in 0..h {
            for x in 0..w {
                let base = 4000.0 + 40000.0 * x as f64 / w as f64 + 8000.0 * y as f64 / h as f64;
                let disc = if (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) < r2 { 12000.0 } else { 0.0 };
                for c in 0..3 {
                    let noise = (rng.next_u32() % 2048) as f64 - 1024.0;
                    let tint = [1.0, 0.8, 0.6][c];
                    let v = (base * tint + disc + noise).round();
                    img.push(v.clamp(0.0, 65535.0) as u16);
                }
            }
        }
        img
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("vr180_dng_write_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_to_vec(
        img: &LinearDngImage,
        meta: &LinearDngMeta,
        comp: DngCompression,
        thumb: Option<(u32, u32, &[u8])>,
    ) -> Vec<u8> {
        let mut cur = Cursor::new(Vec::new());
        write_linear_dng_to(&mut cur, img, meta, comp, thumb).unwrap();
        cur.into_inner()
    }

    /// Full structural + pixel verification of a written DNG.
    fn check_dng(buf: &[u8], img: &LinearDngImage, meta: &LinearDngMeta, comp: DngCompression, thumb: Option<(u32, u32, &[u8])>) {
        assert_eq!(&buf[..4], b"II*\0");
        assert_eq!(le32(buf, 4), 8);
        let ifd0 = read_ifd(buf, 8);
        assert_eq!(ifd0.next, 0, "IFD0 must be the only IFD in the chain");

        // -- IFD0: thumbnail + colour tags
        assert_eq!(ifd0.longs(tag::NEW_SUBFILE_TYPE), [1]);
        let (tw, th) = (ifd0.longs(tag::IMAGE_WIDTH)[0], ifd0.longs(tag::IMAGE_LENGTH)[0]);
        assert_eq!(ifd0.shorts(tag::BITS_PER_SAMPLE), [8, 8, 8]);
        assert_eq!(ifd0.shorts(tag::COMPRESSION), [1]);
        assert_eq!(ifd0.shorts(tag::PHOTOMETRIC), [2]);
        assert_eq!(ifd0.shorts(tag::ORIENTATION), [1]);
        assert_eq!(ifd0.shorts(tag::SAMPLES_PER_PIXEL), [3]);
        assert_eq!(ifd0.shorts(tag::PLANAR_CONFIGURATION), [1]);
        assert_eq!(ifd0.longs(tag::ROWS_PER_STRIP), [th]);
        let (so, sc) = (ifd0.longs(tag::STRIP_OFFSETS)[0], ifd0.longs(tag::STRIP_BYTE_COUNTS)[0]);
        assert_eq!(sc, tw * th * 3);
        assert_eq!(so % 4, 0);
        let thumb_data = &buf[so as usize..(so + sc) as usize];
        match thumb {
            Some((ew, eh, eb)) => {
                assert_eq!((tw, th), (ew, eh));
                assert_eq!(thumb_data, eb);
            }
            None => {
                let (ew, eh, eb) = generate_thumbnail(img, meta);
                assert_eq!((tw, th), (ew, eh));
                assert!(tw <= MAX_THUMBNAIL_WIDTH);
                assert_eq!(thumb_data, &eb[..]);
            }
        }
        assert_eq!(ifd0.ascii(tag::MAKE), meta.make);
        assert_eq!(ifd0.ascii(tag::MODEL), meta.model);
        assert_eq!(ifd0.ascii(tag::SOFTWARE), meta.software);
        assert_eq!(ifd0.ascii(tag::UNIQUE_CAMERA_MODEL), meta.unique_camera_model);
        assert_eq!(ifd0.rationals(tag::X_RESOLUTION), [(72, 1)]);
        assert_eq!(ifd0.rationals(tag::Y_RESOLUTION), [(72, 1)]);
        assert_eq!(ifd0.shorts(tag::RESOLUTION_UNIT), [2]);
        assert_eq!(ifd0.bytes(tag::DNG_VERSION, TYPE_BYTE), [1, 4, 0, 0]);
        assert_eq!(ifd0.bytes(tag::DNG_BACKWARD_VERSION, TYPE_BYTE), [1, 1, 0, 0]);
        assert_eq!(ifd0.shorts(tag::CALIBRATION_ILLUMINANT1), [meta.calibration_illuminant1]);
        let cm: Vec<(i32, i32)> = meta.color_matrix1.iter().map(|&v| srat(v as f64)).collect();
        assert_eq!(ifd0.srationals(tag::COLOR_MATRIX1), cm);
        let asn: Vec<(u32, u32)> = meta.as_shot_neutral.iter().map(|&v| urat(v as f64)).collect();
        assert_eq!(ifd0.rationals(tag::AS_SHOT_NEUTRAL), asn);
        assert_eq!(ifd0.srationals(tag::BASELINE_EXPOSURE), [srat(meta.baseline_exposure as f64)]);
        match meta.baseline_noise {
            Some(v) => assert_eq!(ifd0.rationals(tag::BASELINE_NOISE), [urat(v as f64)]),
            None => assert!(!ifd0.entries.contains_key(&tag::BASELINE_NOISE)),
        }
        match meta.baseline_sharpness {
            Some(v) => assert_eq!(ifd0.rationals(tag::BASELINE_SHARPNESS), [urat(v as f64)]),
            None => assert!(!ifd0.entries.contains_key(&tag::BASELINE_SHARPNESS)),
        }
        match &meta.profile_tone_curve {
            Some(tc) => assert_eq!(ifd0.floats(tag::PROFILE_TONE_CURVE), tc.iter().flat_map(|p| [p[0], p[1]]).collect::<Vec<f32>>()),
            None => assert!(!ifd0.entries.contains_key(&tag::PROFILE_TONE_CURVE)),
        }
        match &meta.profile_look_table {
            Some(lt) => {
                assert_eq!(ifd0.longs(tag::PROFILE_LOOK_TABLE_DIMS), lt.dims);
                assert_eq!(ifd0.floats(tag::PROFILE_LOOK_TABLE_DATA), lt.data);
                assert_eq!(ifd0.longs(tag::PROFILE_LOOK_TABLE_ENCODING), [lt.encoding]);
            }
            None => assert!(!ifd0.entries.contains_key(&tag::PROFILE_LOOK_TABLE_DIMS)),
        }
        match (&meta.profile_hue_sat_map_dims, &meta.profile_hue_sat_map_data1) {
            (Some(d), Some(data)) => {
                assert_eq!(ifd0.longs(tag::PROFILE_HUE_SAT_MAP_DIMS), d);
                assert_eq!(ifd0.floats(tag::PROFILE_HUE_SAT_MAP_DATA1), *data);
                if let Some(e) = meta.profile_hue_sat_map_encoding {
                    assert_eq!(ifd0.longs(tag::PROFILE_HUE_SAT_MAP_ENCODING), [e]);
                }
            }
            _ => {
                assert!(!ifd0.entries.contains_key(&tag::PROFILE_HUE_SAT_MAP_DIMS));
                assert!(!ifd0.entries.contains_key(&tag::PROFILE_HUE_SAT_MAP_DATA1));
                assert!(!ifd0.entries.contains_key(&tag::PROFILE_HUE_SAT_MAP_ENCODING));
            }
        }
        match &meta.datetime_original {
            Some(dt) => assert_eq!(ifd0.ascii(tag::DATE_TIME), *dt),
            None => assert!(!ifd0.entries.contains_key(&tag::DATE_TIME)),
        }
        match &meta.image_description {
            Some(d) => assert_eq!(ifd0.ascii(tag::IMAGE_DESCRIPTION), *d),
            None => assert!(!ifd0.entries.contains_key(&tag::IMAGE_DESCRIPTION)),
        }
        // No CFA / opcode tags anywhere.
        for t in [33421u16, 33422, 50710, 50711, 51008, 51009, 51022] {
            assert!(!ifd0.entries.contains_key(&t), "unexpected tag {t} in IFD0");
        }

        // -- EXIF IFD
        let exif = read_ifd(buf, ifd0.longs(tag::EXIF_IFD)[0]);
        assert_eq!(exif.next, 0);
        assert_eq!(exif.bytes(tag::EXIF_VERSION, TYPE_UNDEFINED), b"0230");
        match meta.exposure_s {
            Some(v) => assert_eq!(exif.rationals(tag::EXPOSURE_TIME), [urat(v)]),
            None => assert!(!exif.entries.contains_key(&tag::EXPOSURE_TIME)),
        }
        match meta.f_number {
            Some(v) => assert_eq!(exif.rationals(tag::F_NUMBER), [urat(v)]),
            None => assert!(!exif.entries.contains_key(&tag::F_NUMBER)),
        }
        match meta.iso {
            Some(v) => assert_eq!(exif.shorts(tag::ISO), [v as u16]),
            None => assert!(!exif.entries.contains_key(&tag::ISO)),
        }
        match &meta.datetime_original {
            Some(dt) => {
                assert_eq!(exif.ascii(tag::DATE_TIME_ORIGINAL), *dt);
                assert_eq!(exif.ascii(tag::DATE_TIME_DIGITIZED), *dt);
            }
            None => assert!(!exif.entries.contains_key(&tag::DATE_TIME_ORIGINAL)),
        }
        match meta.focal_mm {
            Some(v) => assert_eq!(exif.rationals(tag::FOCAL_LENGTH), [urat(v)]),
            None => assert!(!exif.entries.contains_key(&tag::FOCAL_LENGTH)),
        }
        match meta.focal_35mm {
            Some(v) => assert_eq!(exif.shorts(tag::FOCAL_LENGTH_35MM), [v as u16]),
            None => assert!(!exif.entries.contains_key(&tag::FOCAL_LENGTH_35MM)),
        }

        // -- main image IFD
        let main = read_ifd(buf, ifd0.longs(tag::SUB_IFDS)[0]);
        assert_eq!(main.next, 0);
        let (w, h) = (img.width, img.height);
        assert_eq!(main.longs(tag::NEW_SUBFILE_TYPE), [0]);
        assert_eq!(main.longs(tag::IMAGE_WIDTH), [w]);
        assert_eq!(main.longs(tag::IMAGE_LENGTH), [h]);
        assert_eq!(main.shorts(tag::BITS_PER_SAMPLE), [16, 16, 16]);
        assert_eq!(main.shorts(tag::PHOTOMETRIC), [PHOTOMETRIC_LINEAR_RAW]);
        assert_eq!(main.shorts(tag::SAMPLES_PER_PIXEL), [3]);
        assert_eq!(main.shorts(tag::PLANAR_CONFIGURATION), [1]);
        assert_eq!(main.shorts(tag::ORIENTATION), [1]);
        assert_eq!(main.shorts(tag::SAMPLE_FORMAT), [1, 1, 1]);
        assert_eq!(main.shorts(tag::BLACK_LEVEL), [0, 0, 0]);
        assert_eq!(main.longs(tag::WHITE_LEVEL), [img.white_level; 3]);
        assert_eq!(main.rationals(tag::DEFAULT_SCALE), [(1, 1), (1, 1)]);
        assert_eq!(main.longs(tag::DEFAULT_CROP_ORIGIN), [0, 0]);
        assert_eq!(main.longs(tag::DEFAULT_CROP_SIZE), [w, h]);
        assert_eq!(main.longs(tag::ACTIVE_AREA), [0, 0, h, w]);
        for t in [33421u16, 33422, 50710, 50711, 51008, 51009, 51022, tag::MAKE, tag::MODEL, tag::DNG_VERSION] {
            assert!(!main.entries.contains_key(&t), "unexpected tag {t} in main IFD");
        }

        // -- pixel data
        let mut got = vec![0u16; (w * h * 3) as usize];
        match comp {
            DngCompression::LosslessJpegTiles => {
                assert_eq!(main.shorts(tag::COMPRESSION), [7]);
                assert_eq!(main.longs(tag::TILE_WIDTH), [TILE_SIZE]);
                assert_eq!(main.longs(tag::TILE_LENGTH), [TILE_SIZE]);
                assert!(!main.entries.contains_key(&tag::STRIP_OFFSETS));
                assert!(!main.entries.contains_key(&tag::ROWS_PER_STRIP));
                let (offs, cnts) = (main.longs(tag::TILE_OFFSETS), main.longs(tag::TILE_BYTE_COUNTS));
                let across = w.div_ceil(TILE_SIZE);
                let down = h.div_ceil(TILE_SIZE);
                assert_eq!(offs.len() as u32, across * down);
                assert_eq!(cnts.len(), offs.len());
                let mut prev_end = 0u32;
                for (k, (&off, &cnt)) in offs.iter().zip(&cnts).enumerate() {
                    assert_eq!(off % 4, 0, "tile {k} not 4-byte aligned");
                    assert!(off >= prev_end, "tile {k} overlaps the previous one");
                    assert!(cnt > 0 && (off + cnt) as usize <= buf.len());
                    prev_end = off + cnt;
                    let dec = ljpeg_decode(&buf[off as usize..(off + cnt) as usize]);
                    assert_eq!((dec.height, dec.width, dec.ncomp), (TILE_SIZE as usize, TILE_SIZE as usize, 3));
                    let (tx, ty) = (k as u32 % across, k as u32 / across);
                    let (x0, y0) = (tx * TILE_SIZE, ty * TILE_SIZE);
                    let vis_w = TILE_SIZE.min(w - x0) as usize;
                    let vis_h = TILE_SIZE.min(h - y0) as usize;
                    for r in 0..TILE_SIZE as usize {
                        let trow = &dec.samples[r * TILE_SIZE as usize * 3..(r + 1) * TILE_SIZE as usize * 3];
                        // Padding must replicate the last valid column / row.
                        for x in vis_w..TILE_SIZE as usize {
                            assert_eq!(&trow[x * 3..x * 3 + 3], &trow[(vis_w - 1) * 3..vis_w * 3], "column padding tile {k}");
                        }
                        if r >= vis_h {
                            let last = &dec.samples[(vis_h - 1) * TILE_SIZE as usize * 3..vis_h * TILE_SIZE as usize * 3];
                            assert_eq!(trow, last, "row padding tile {k}");
                            continue;
                        }
                        let y = y0 as usize + r;
                        let dst = &mut got[(y * w as usize + x0 as usize) * 3..(y * w as usize + x0 as usize + vis_w) * 3];
                        dst.copy_from_slice(&trow[..vis_w * 3]);
                    }
                }
                assert_eq!(prev_end as usize, buf.len(), "file must end after the last tile");
            }
            DngCompression::Uncompressed => {
                assert_eq!(main.shorts(tag::COMPRESSION), [1]);
                assert_eq!(main.longs(tag::ROWS_PER_STRIP), [ROWS_PER_STRIP]);
                assert!(!main.entries.contains_key(&tag::TILE_OFFSETS));
                assert!(!main.entries.contains_key(&tag::TILE_WIDTH));
                let (offs, cnts) = (main.longs(tag::STRIP_OFFSETS), main.longs(tag::STRIP_BYTE_COUNTS));
                assert_eq!(offs.len() as u32, h.div_ceil(ROWS_PER_STRIP));
                let mut prev_end = 0u32;
                for (k, (&off, &cnt)) in offs.iter().zip(&cnts).enumerate() {
                    assert_eq!(off % 4, 0, "strip {k} not 4-byte aligned");
                    assert!(off >= prev_end);
                    let y0 = k as u32 * ROWS_PER_STRIP;
                    let rows = ROWS_PER_STRIP.min(h - y0);
                    assert_eq!(cnt, rows * w * 6, "strip {k} byte count");
                    prev_end = off + cnt;
                    let data = &buf[off as usize..(off + cnt) as usize];
                    let dst = &mut got[(y0 * w * 3) as usize..((y0 + rows) * w * 3) as usize];
                    for (d, s) in dst.iter_mut().zip(data.chunks_exact(2)) {
                        *d = u16::from_le_bytes([s[0], s[1]]);
                    }
                }
                assert_eq!(prev_end as usize, buf.len());
            }
        }
        assert!(got == img.rgb16, "decoded samples differ from the source image");
    }

    // ---- tests -------------------------------------------------------------

    #[test]
    fn rational_conversion_matches_python_fraction() {
        // Values and results from `Fraction(v).limit_denominator(1 << 20)`.
        assert_eq!(srat(0.71875), (23, 32));
        assert_eq!(srat(-0.1484375), (-19, 128));
        assert_eq!(srat(2.6), (13, 5));
        assert_eq!(srat(2.6f32 as f64), (13, 5));
        assert_eq!(urat(0.4599609375), (471, 1024));
        assert_eq!(urat(0.537109375), (275, 512));
        assert_eq!(urat(1.0 / 5072.0), (1, 5072));
        assert_eq!(urat(2.54), (127, 50));
        assert_eq!(urat(2.0), (2, 1));
        assert_eq!(urat(1.0 / 3.0), (1, 3));
        assert_eq!(urat(0.1), (1, 10));
        assert_eq!(urat(0.0), (0, 1));
        assert_eq!(urat(-5.0), (0, 1));
        assert_eq!(srat(-0.0), (0, 1));
        assert_eq!(urat(f64::NAN), (0, 1));
        assert_eq!(urat(1e-30), (0, 1));
        assert_eq!(srat(1.0 / 1048577.0), (1, 1048576)); // best fraction at the denominator cap
        // Numerator bound: 1e12 cannot be 1e12/1 in SRATIONAL -> smaller denominator path.
        let (n, d) = srat(3e9);
        assert!(n == i32::MAX && d == 1);
        let (n, d) = urat(3e9);
        assert_eq!((n, d), (3_000_000_000, 1));
    }

    #[test]
    fn ifd_serializer_inlines_and_aligns() {
        let mut ifd = Ifd::default();
        ifd.set_short(300, &[1]);
        ifd.set_ascii(200, "abc"); // 4 bytes with NUL -> inline
        ifd.set_ascii(100, "abcd"); // 5 bytes -> out of line
        ifd.set_long(400, &[1, 2, 3]); // 12 bytes
        let (b, pos) = ifd.serialize(8);
        // 2 + 4*12 + 4 = 54 bytes of table; first out-of-line value padded to 64.
        assert_eq!(le16(&b, 0), 4);
        assert_eq!(&b[2..4], &100u16.to_le_bytes());
        assert_eq!(pos[&100], 64);
        assert_eq!(&b[(64 - 8)..(64 - 8 + 5)], b"abcd\0");
        assert_eq!(pos[&400], 72); // 69 -> aligned 72
        assert_eq!(pos[&200], 8 + 2 + 12 + 8); // inline, second entry
        assert_eq!(&b[(pos[&200] - 8) as usize..(pos[&200] - 8 + 4) as usize], b"abc\0");
        assert_eq!(pos[&300], 8 + 2 + 24 + 8);
        assert_eq!(b.len(), 72 + 12 - 8);
        // The positions the writer patches through must be where an
        // independent reader finds the values.
        let mut file = vec![0u8; 8];
        file.extend_from_slice(&b);
        let parsed = read_ifd(&file, 8);
        for (tag, e) in &parsed.entries {
            assert_eq!(e.value_pos, pos[tag], "value position of tag {tag}");
            assert_eq!(e.count as usize * TYPE_SIZE[e.typ as usize], e.data.len());
        }
        assert_eq!(parsed.ascii(100), "abcd");
        assert_eq!(parsed.longs(400), [1, 2, 3]);
    }

    /// Decode the main image of any thumbnail+SubIFD or single-IFD DNG with
    /// lossless-JPEG tiles (what Adobe DNG Converter writes) into interleaved
    /// samples, for comparing a converter's output with the source.
    fn decode_any_ljpeg_dng(buf: &[u8]) -> (u32, u32, u32, Vec<u16>) {
        assert_eq!(&buf[..4], b"II*\0");
        let mut ifd = read_ifd_opts(buf, le32(buf, 4), false);
        if ifd.longs(tag::NEW_SUBFILE_TYPE) == [1] && ifd.entries.contains_key(&tag::SUB_IFDS) {
            ifd = read_ifd_opts(buf, ifd.longs(tag::SUB_IFDS)[0], false);
        }
        let (w, h) = (ifd.longs(tag::IMAGE_WIDTH)[0], ifd.longs(tag::IMAGE_LENGTH)[0]);
        let spp = ifd.shorts(tag::SAMPLES_PER_PIXEL)[0] as u32;
        assert_eq!(ifd.shorts(tag::COMPRESSION), [7], "expected lossless JPEG");
        let (offs, cnts, tw, tl) = if ifd.entries.contains_key(&tag::TILE_OFFSETS) {
            (ifd.longs(tag::TILE_OFFSETS), ifd.longs(tag::TILE_BYTE_COUNTS), ifd.longs(tag::TILE_WIDTH)[0], ifd.longs(tag::TILE_LENGTH)[0])
        } else {
            (ifd.longs(tag::STRIP_OFFSETS), ifd.longs(tag::STRIP_BYTE_COUNTS), w, ifd.longs(tag::ROWS_PER_STRIP)[0])
        };
        let across = w.div_ceil(tw);
        let mut img = vec![0u16; (w * h * spp) as usize];
        for (k, (&off, &cnt)) in offs.iter().zip(&cnts).enumerate() {
            let dec = ljpeg_decode(&buf[off as usize..(off + cnt) as usize]);
            assert_eq!(dec.width as u32 * dec.ncomp as u32, tw * spp, "tile {k} geometry");
            let (tx, ty) = (k as u32 % across, k as u32 / across);
            let (x0, y0) = (tx * tw, ty * tl);
            let n = (tw.min(w - x0) * spp) as usize;
            for (r, trow) in dec.samples.chunks_exact(dec.width * dec.ncomp).enumerate() {
                let y = y0 + r as u32;
                if y >= h {
                    break;
                }
                let dst = &mut img[((y * w + x0) * spp) as usize..((y * w + x0) * spp) as usize + n];
                dst.copy_from_slice(&trow[..n]);
            }
        }
        (w, h, spp, img)
    }

    #[test]
    fn write_512x256_both_compressions() {
        let (w, h) = (512u32, 256u32);
        let rgb = realish_image(w as usize, h as usize, 1);
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 65535 };
        let meta = x6_meta();
        for comp in [DngCompression::LosslessJpegTiles, DngCompression::Uncompressed] {
            let buf = write_to_vec(&img, &meta, comp, None);
            check_dng(&buf, &img, &meta, comp, None);
            if comp == DngCompression::LosslessJpegTiles {
                assert!(buf.len() > 102400, "LJPEG file too small for ImageIO: {}", buf.len());
                assert!(buf.len() < (w * h * 6) as usize, "LJPEG did not compress");
            }
        }
        // Provided thumbnail + the path-based entry point.
        let thumb: Vec<u8> = (0..4 * 2 * 3).map(|i| i as u8 * 10).collect();
        let dir = temp_dir("rt");
        let path = dir.join("t.dng");
        write_linear_dng(&path, &img, &meta, DngCompression::LosslessJpegTiles, Some((4, 2, &thumb))).unwrap();
        let buf = std::fs::read(&path).unwrap();
        check_dng(&buf, &img, &meta, DngCompression::LosslessJpegTiles, Some((4, 2, &thumb)));
        // Identical bytes from the Cursor and File paths.
        assert_eq!(buf, write_to_vec(&img, &meta, DngCompression::LosslessJpegTiles, Some((4, 2, &thumb))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_partial_tiles_strips_and_sparse_meta() {
        // 300x130: 2x1 tile grid with a 44-px-wide / 130-row partial tile;
        // strips 64, 64, 2.
        let (w, h) = (300u32, 130u32);
        let rgb = realish_image(w as usize, h as usize, 2);
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 60000 };
        let mut meta = x6_meta();
        meta.baseline_noise = None;
        meta.baseline_sharpness = None;
        meta.profile_hue_sat_map_dims = None;
        meta.profile_hue_sat_map_data1 = None;
        meta.profile_hue_sat_map_encoding = None;
        meta.exposure_s = None;
        meta.f_number = None;
        meta.iso = None;
        meta.focal_mm = None;
        meta.focal_35mm = None;
        meta.datetime_original = None;
        meta.image_description = None;
        for comp in [DngCompression::LosslessJpegTiles, DngCompression::Uncompressed] {
            let buf = write_to_vec(&img, &meta, comp, None);
            check_dng(&buf, &img, &meta, comp, None);
        }
        // Synthetic reference content too (smooth, tiny streams).
        let syn = synth_image(257, 3);
        let img = LinearDngImage { width: 257, height: 3, rgb16: &syn, white_level: 65535 };
        for comp in [DngCompression::LosslessJpegTiles, DngCompression::Uncompressed] {
            let buf = write_to_vec(&img, &meta, comp, None);
            check_dng(&buf, &img, &meta, comp, None);
        }
    }

    #[test]
    fn write_single_tile_inline_offsets() {
        // One tile / one strip: TileOffsets and TileByteCounts are inlined in
        // the entry, exercising the inline patch path.
        let rgb = realish_image(16, 16, 3);
        let img = LinearDngImage { width: 16, height: 16, rgb16: &rgb, white_level: 65535 };
        let meta = x6_meta();
        for comp in [DngCompression::LosslessJpegTiles, DngCompression::Uncompressed] {
            let buf = write_to_vec(&img, &meta, comp, None);
            check_dng(&buf, &img, &meta, comp, None);
        }
        // 1x1 image.
        let px = [1u16, 2, 3];
        let img = LinearDngImage { width: 1, height: 1, rgb16: &px, white_level: 65535 };
        for comp in [DngCompression::LosslessJpegTiles, DngCompression::Uncompressed] {
            let buf = write_to_vec(&img, &meta, comp, None);
            check_dng(&buf, &img, &meta, comp, None);
        }
    }

    #[test]
    fn generated_thumbnail_is_box_filtered_and_gamma_encoded() {
        // 512x256 constant image -> 256x128 thumbnail of one value.
        let (w, h) = (512u32, 256u32);
        let mut meta = x6_meta();
        meta.baseline_exposure = 0.0;
        meta.as_shot_neutral = [1.0, 1.0, 1.0];
        let rgb = vec![65535u16 / 4; (w * h * 3) as usize];
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 65535 };
        let (tw, th, t) = generate_thumbnail(&img, &meta);
        assert_eq!((tw, th), (256, 128));
        let expect = ((16383.0f64 / 65535.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        assert!(t.iter().all(|&v| v == expect), "thumb value {} != {expect}", t[0]);
        // Checkerboard of 0 / 65535 at 2x2 -> every thumb cell averages 0.5 (gamma'd).
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0 } else { 65535 };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 65535 };
        let (_, _, t) = generate_thumbnail(&img, &meta);
        let expect = (0.5f64.powf(1.0 / 2.2) * 255.0).round() as u8;
        assert!(t.iter().all(|&v| v == expect));
        // Narrow image keeps its width; white balance and exposure apply.
        meta.baseline_exposure = 1.0;
        meta.as_shot_neutral = [0.5, 1.0, 0.25];
        let rgb = vec![8192u16; 10 * 7 * 3];
        let img = LinearDngImage { width: 10, height: 7, rgb16: &rgb, white_level: 65535 };
        let (tw, th, t) = generate_thumbnail(&img, &meta);
        assert_eq!((tw, th), (10, 7));
        let lin: f64 = 8192.0 / 65535.0 * 2.0;
        let e = [lin / 0.5, lin, lin / 0.25].map(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8);
        assert_eq!(&t[..3], &e);
    }

    #[test]
    fn rejects_bad_input() {
        let rgb = vec![0u16; 4 * 4 * 3];
        let meta = x6_meta();
        let bad = LinearDngImage { width: 4, height: 5, rgb16: &rgb, white_level: 65535 };
        let err = write_linear_dng_to(Cursor::new(Vec::new()), &bad, &meta, DngCompression::Uncompressed, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let ok = LinearDngImage { width: 4, height: 4, rgb16: &rgb, white_level: 65535 };
        let t = [0u8; 9];
        let err = write_linear_dng_to(Cursor::new(Vec::new()), &ok, &meta, DngCompression::Uncompressed, Some((2, 2, &t))).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let wide = vec![0u8; 300 * 3];
        let err = write_linear_dng_to(Cursor::new(Vec::new()), &ok, &meta, DngCompression::Uncompressed, Some((300, 1, &wide))).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let mut m = meta.clone();
        m.profile_hue_sat_map_data1 = Some(vec![1.0; 5]);
        let err = write_linear_dng_to(Cursor::new(Vec::new()), &ok, &m, DngCompression::Uncompressed, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        // Non-ASCII text is sanitised, not rejected.
        let mut m = meta.clone();
        m.software = "Silver Bullet — ünïcode\0x".into();
        let buf = write_to_vec(&ok, &m, DngCompression::Uncompressed, None);
        let ifd0 = read_ifd(&buf, 8);
        assert_eq!(ifd0.ascii(tag::SOFTWARE), "Silver Bullet ? ?n?codex");
    }

    /// System validation with the external readers the recipe was validated
    /// against: exiftool, macOS ImageIO (`sips`) and Adobe DNG Converter.
    /// `cargo test -p vr180-pipeline --lib dng_write -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn system_validate_exiftool_sips_adobe() {
        let (w, h) = (512u32, 256u32);
        let rgb = realish_image(w as usize, h as usize, 11);
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 65535 };
        let meta = x6_meta();
        // VR180_DNG_KEEP_DIR=<dir> writes there and keeps the files (for
        // checking them with other tools afterwards).
        let keep = std::env::var_os("VR180_DNG_KEEP_DIR").map(PathBuf::from);
        let dir = match &keep {
            Some(d) => {
                std::fs::create_dir_all(d).unwrap();
                d.clone()
            }
            None => temp_dir("sys"),
        };
        let have = |cmd: &str| Command::new("which").arg(cmd).output().map(|o| o.status.success()).unwrap_or(false);
        let adobe = Path::new("/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter");
        for (comp, name) in [(DngCompression::LosslessJpegTiles, "ljpeg"), (DngCompression::Uncompressed, "uncomp")] {
            let path = dir.join(format!("{name}.dng"));
            write_linear_dng(&path, &img, &meta, comp, None).unwrap();
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(size >= 102400, "{name}: {size} bytes is below ImageIO's minimum");
            if have("exiftool") {
                let out = Command::new("exiftool").args(["-validate", "-warning", "-a"]).arg(&path).output().unwrap();
                let s = String::from_utf8_lossy(&out.stdout);
                eprintln!("[{name}] exiftool: {}", s.trim());
                assert!(out.status.success());
                assert!(s.lines().any(|l| l.starts_with("Validate") && l.contains("OK")), "{name}: exiftool validate: {s}");
                assert!(!s.lines().any(|l| l.starts_with("Warning")), "{name}: exiftool warnings: {s}");
            } else {
                eprintln!("[{name}] exiftool not installed — skipped");
            }
            if have("sips") {
                let out = Command::new("sips").args(["-g", "pixelWidth", "-g", "pixelHeight"]).arg(&path).output().unwrap();
                let s = String::from_utf8_lossy(&out.stdout);
                eprintln!("[{name}] sips: {}", s.trim());
                assert!(out.status.success(), "{name}: sips failed: {}", String::from_utf8_lossy(&out.stderr));
                assert!(s.contains(&format!("pixelWidth: {w}")), "{name}: sips width: {s}");
                assert!(s.contains(&format!("pixelHeight: {h}")), "{name}: sips height: {s}");
            } else {
                eprintln!("[{name}] sips not available — skipped");
            }
            if adobe.is_file() {
                let out_name = format!("{name}_adobe.dng");
                let out = Command::new(adobe).args(["-c", "-p0", "-d"]).arg(&dir).arg("-o").arg(&out_name).arg(&path).output().unwrap();
                eprintln!("[{name}] Adobe DNG Converter: exit {:?}", out.status.code());
                assert!(out.status.success(), "{name}: DNG Converter failed: {}", String::from_utf8_lossy(&out.stderr));
                let produced = dir.join(&out_name);
                assert!(produced.is_file(), "{name}: DNG Converter wrote no output");
                let adobe_buf = std::fs::read(&produced).unwrap();
                eprintln!("[{name}] converter output {} bytes", adobe_buf.len());
                // The converter re-encodes the samples it decoded: they must
                // be exactly the ones we wrote.
                let (aw, ah, aspp, adobe_samples) = decode_any_ljpeg_dng(&adobe_buf);
                assert_eq!((aw, ah, aspp), (w, h, 3), "{name}: converter output geometry");
                assert!(adobe_samples == rgb, "{name}: Adobe DNG Converter output samples differ from the source");
                eprintln!("[{name}] converter output decodes sample-identical to the source");
            } else {
                eprintln!("[{name}] Adobe DNG Converter not installed — skipped");
            }
        }
        if keep.is_none() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// Timing of a full-size (15488x7744) write in both compressions.
    /// `cargo test --release -p vr180-pipeline --lib dng_write::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_full_size_write() {
        let (w, h) = (15488u32, 7744u32);
        let t0 = Instant::now();
        let rgb = realish_image(w as usize, h as usize, 5);
        eprintln!("synthetic {w}x{h} generated in {:.2}s", t0.elapsed().as_secs_f64());
        let img = LinearDngImage { width: w, height: h, rgb16: &rgb, white_level: 65535 };
        let meta = x6_meta();
        let dir = temp_dir("bench");
        for (comp, name) in [(DngCompression::LosslessJpegTiles, "ljpeg"), (DngCompression::Uncompressed, "uncomp")] {
            let path = dir.join(format!("{name}.dng"));
            let t = Instant::now();
            write_linear_dng(&path, &img, &meta, comp, None).unwrap();
            let dt = t.elapsed().as_secs_f64();
            let size = std::fs::metadata(&path).unwrap().len();
            eprintln!("{name}: {dt:.2}s, {size} bytes ({:.1} MB, {:.0}% of raw)", size as f64 / 1e6, 100.0 * size as f64 / (w as f64 * h as f64 * 6.0));
            // Spot-check: decode the last tile / strip back.
            let buf = std::fs::read(&path).unwrap();
            let ifd0 = read_ifd(&buf, 8);
            let main = read_ifd(&buf, ifd0.longs(tag::SUB_IFDS)[0]);
            if comp == DngCompression::LosslessJpegTiles {
                let (offs, cnts) = (main.longs(tag::TILE_OFFSETS), main.longs(tag::TILE_BYTE_COUNTS));
                assert_eq!(offs.len(), 61 * 31);
                let k = offs.len() - 1;
                let dec = ljpeg_decode(&buf[offs[k] as usize..(offs[k] + cnts[k]) as usize]);
                let (x0, y0) = (60 * 256usize, 30 * 256usize);
                assert_eq!(&dec.samples[..3], &rgb[(y0 * w as usize + x0) * 3..(y0 * w as usize + x0) * 3 + 3]);
            }
            std::fs::remove_file(&path).unwrap();
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
