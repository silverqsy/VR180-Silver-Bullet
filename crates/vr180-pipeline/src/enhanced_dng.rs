//! The picture a raw converter leaves in a DNG it has enhanced: Adobe
//! Camera Raw's Super Resolution, applied in place.
//!
//! Camera Raw 18 no longer writes a second file for an enhanced frame. It
//! keeps the DNG as it was — the mosaic, its tags — and appends the
//! enhanced picture to it: the frame demosaiced and enlarged twofold, in
//! the camera's own linear RGB. Read here from what such a file holds (a
//! frame of an X6 raw take enhanced by Camera Raw 18.7):
//!
//! - Three tags join IFD0: `0xCD3C` (16 bytes, a digest), `0xCD3D` (the
//!   offset of a block of the file) and `0xCD3E` (its length).
//! - The block: 60 bytes of header (two words, the digest, width, height,
//!   planes, sample type, a length, four more words), then a TIFF of its
//!   own with one IFD: width × height, three 16-bit samples a pixel,
//!   `Compression` 52546 (JPEG XL), tiles (768 × 864 in the sample), each
//!   tile a bare JPEG XL codestream.
//! - The codes are not linear. The file's XMP says how they are (under the
//!   `Enhance` filter of `crs:FilterList`): a black and a white code for
//!   each channel (`crs:Black="1353,65535"`) and a polynomial
//!   (`crs:MapPolynomial`, four fractions `c0 … c3`). With
//!   `x = (code − black) / (white − black)` the camera value is
//!   `c0 + c1·x + c2·x² + c3·x³`, the sensor's white level at 1. Checked
//!   against Camera Raw's own demosaic of the same frame: block means
//!   agree to within 0.1 % (0.3 % in blue).
//!
//! [`find`] reads that description; [`EnhancedPicture::decode_rgba16`]
//! decodes the picture into camera values. Decoding needs a JPEG XL
//! decoder: ImageIO's on macOS; this build has none elsewhere.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{Error, Result};

const TAG_XMP: u16 = 700;
const TAG_BLOCK_OFFSET: u16 = 0xCD3D;
const TAG_BLOCK_LENGTH: u16 = 0xCD3E;
const HEADER: usize = 60;
const JPEG_XL: u32 = 52546;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::Ffmpeg(format!("enhanced picture: {msg}"))
}

/// A TIFF's byte order and one IFD's entries.
struct Ifd {
    little: bool,
    /// tag → (type, count, the entry's four value bytes).
    entries: Vec<(u16, u16, u32, [u8; 4])>,
}

impl Ifd {
    fn u16_of(&self, b: &[u8]) -> u16 {
        if self.little { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) }
    }
    fn u32_of(&self, b: &[u8]) -> u32 {
        if self.little { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) }
    }
    fn entry(&self, tag: u16) -> Option<&(u16, u16, u32, [u8; 4])> {
        self.entries.iter().find(|e| e.0 == tag)
    }
    /// A single SHORT or LONG value held in the entry itself.
    fn value(&self, tag: u16) -> Option<u32> {
        let e = self.entry(tag)?;
        match (e.1, e.2) {
            (3, 1) => Some(self.u16_of(&e.3) as u32),
            (4, 1) | (13, 1) => Some(self.u32_of(&e.3)),
            _ => None,
        }
    }
    /// Where an entry's values are when they do not fit it, and how many.
    fn array(&self, tag: u16) -> Option<(u16, u32, u32)> {
        let e = self.entry(tag)?;
        Some((e.1, e.2, self.u32_of(&e.3)))
    }
}

/// Parse a TIFF header and its first IFD from `bytes` (the TIFF starts at
/// `bytes[0]`; the IFD must lie inside).
fn first_ifd(bytes: &[u8]) -> Option<Ifd> {
    let little = match bytes.get(..4)? {
        [b'I', b'I', 42, 0] => true,
        [b'M', b'M', 0, 42] => false,
        _ => return None,
    };
    let mut ifd = Ifd { little, entries: Vec::new() };
    let at = ifd.u32_of(bytes.get(4..8)?) as usize;
    let n = ifd.u16_of(bytes.get(at..at + 2)?) as usize;
    for i in 0..n {
        let e = bytes.get(at + 2 + 12 * i..at + 14 + 12 * i)?;
        ifd.entries.push((ifd.u16_of(&e[0..2]), ifd.u16_of(&e[2..4]), ifd.u32_of(&e[4..8]), [e[8], e[9], e[10], e[11]]));
    }
    Some(ifd)
}

fn read_at(file: &mut File, at: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(&mut buf).map_err(|_| bad(format!("the file ends before byte {}", at + len as u64)))?;
    Ok(buf)
}

/// How one channel's codes become camera values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMap {
    pub black: f64,
    pub white: f64,
    /// `c0 … c3` of the polynomial over `(code − black) / (white − black)`.
    pub poly: [f64; 4],
}

impl ChannelMap {
    /// The camera value of a code (the sensor's white level = 1).
    pub fn camera(&self, code: u16) -> f64 {
        let x = (code as f64 - self.black) / (self.white - self.black);
        self.poly[0] + x * (self.poly[1] + x * (self.poly[2] + x * self.poly[3]))
    }

    /// Every code's camera value as 16 bits (0..1 → 0..65535, held there).
    fn table(&self) -> Vec<u16> {
        (0..=u16::MAX).map(|c| (self.camera(c).clamp(0.0, 1.0) * 65535.0).round() as u16).collect()
    }
}

/// The enhanced picture of a DNG: where it is and how it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedPicture {
    pub width: u32,
    pub height: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    /// Each tile's place in the file and its length, row-major.
    pub tiles: Vec<(u64, u64)>,
    /// Red, green, blue.
    pub channels: [ChannelMap; 3],
}

/// A fraction as XMP writes it (`"11772426/1073741824"`), or a plain number.
fn fraction(text: &str) -> Option<f64> {
    let t = text.trim();
    match t.split_once('/') {
        Some((n, d)) => {
            let d: f64 = d.trim().parse().ok()?;
            (d != 0.0).then(|| n.trim().parse::<f64>().ok().map(|n| n / d))?
        }
        None => t.parse().ok(),
    }
}

/// The three channels' maps from the XMP packet of an enhanced DNG.
pub fn channel_maps(xmp: &str) -> Option<[ChannelMap; 3]> {
    let lin = &xmp[xmp.find("<crs:Linearization>")?..];
    let lin = &lin[..lin.find("</crs:Linearization>")?];
    let mut polys = Vec::new();
    for block in lin.split("<crs:MapPolynomial>").skip(1) {
        let block = &block[..block.find("</crs:MapPolynomial>")?];
        let c: Vec<f64> = block.split("<rdf:li>").skip(1).filter_map(|li| fraction(li.split("</rdf:li>").next()?)).collect();
        polys.push(<[f64; 4]>::try_from(c).ok()?);
    }
    let levels = &xmp[xmp.find("<crs:BlackLevels>")?..];
    let levels = &levels[..levels.find("</crs:BlackLevels>")?];
    let mut blacks = Vec::new();
    for item in levels.split("crs:Black=\"").skip(1) {
        let (b, w) = item.split('"').next()?.split_once(',')?;
        blacks.push((b.trim().parse::<f64>().ok()?, w.trim().parse::<f64>().ok()?));
    }
    if polys.len() != 3 || blacks.len() != 3 || blacks.iter().any(|(b, w)| w <= b) {
        return None;
    }
    Some([0, 1, 2].map(|i| ChannelMap { black: blacks[i].0, white: blacks[i].1, poly: polys[i] }))
}

/// The enhanced picture of the DNG at `path`; `None` for a DNG that holds
/// none (a file no raw converter has enhanced, or not in this way).
pub fn find(path: &Path) -> Result<Option<EnhancedPicture>> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let head = read_at(&mut file, 0, 8.min(size as usize))?;
    let little = match head.get(..4) {
        Some([b'I', b'I', 42, 0]) => true,
        Some([b'M', b'M', 0, 42]) => false,
        _ => return Ok(None),
    };
    let word = |b: &[u8]| if little { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) };
    // IFD0 may sit anywhere in the file: read it where the header points.
    let at = word(&head[4..8]) as u64;
    if at + 2 > size {
        return Ok(None);
    }
    let count = read_at(&mut file, at, 2)?;
    let n = if little { u16::from_le_bytes([count[0], count[1]]) } else { u16::from_be_bytes([count[0], count[1]]) } as usize;
    // The same parser, on a copy laid out as a TIFF whose IFD follows its header.
    let mut tiff = head.clone();
    tiff[4..8].copy_from_slice(&if little { 8u32.to_le_bytes() } else { 8u32.to_be_bytes() });
    tiff.extend_from_slice(&read_at(&mut file, at, 2 + 12 * n)?);
    let Some(ifd0) = first_ifd(&tiff) else { return Ok(None) };
    let (Some(block_at), Some(block_len)) = (ifd0.value(TAG_BLOCK_OFFSET), ifd0.value(TAG_BLOCK_LENGTH)) else { return Ok(None) };
    let (block_at, block_len) = (block_at as u64, block_len as u64);
    if block_len < HEADER as u64 + 16 || block_at + block_len > size {
        return Err(bad("the block the file points to is not inside it"));
    }
    // The block's header, then its own TIFF.
    let header = read_at(&mut file, block_at, HEADER)?;
    let h32 = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    let (width, height, planes, sample_type) = (h32(24), h32(28), h32(32), h32(36));
    if planes != 3 || sample_type != 3 || width == 0 || height == 0 {
        return Err(bad(format!("{width}x{height}, {planes} planes of sample type {sample_type}: not three 16-bit planes")));
    }
    let inner_at = block_at + HEADER as u64;
    let inner_head = read_at(&mut file, inner_at, 4096.min((block_len - HEADER as u64) as usize))?;
    let inner = first_ifd(&inner_head).ok_or_else(|| bad("no picture inside the block"))?;
    let need = |tag: u16| inner.value(tag).ok_or_else(|| bad(format!("the picture has no tag {tag}")));
    let (w, h, tile_w, tile_h) = (need(256)?, need(257)?, need(322)?, need(323)?);
    if (w, h) != (width, height) || need(259)? != JPEG_XL || need(277)? != 3 || tile_w == 0 || tile_h == 0 {
        return Err(bad(format!("a {w}x{h} picture of compression {} is not what the block's header states", need(259)?)));
    }
    let across = w.div_ceil(tile_w) as usize;
    let down = h.div_ceil(tile_h) as usize;
    let longs = |file: &mut File, tag: u16| -> Result<Vec<u64>> {
        let (kind, count, at) = inner.array(tag).ok_or_else(|| bad(format!("the picture has no tag {tag}")))?;
        if kind != 4 || count as usize != across * down {
            return Err(bad(format!("tag {tag} lists {count} tiles, {across}x{down} expected")));
        }
        let raw = read_at(file, inner_at + at as u64, 4 * count as usize)?;
        Ok(raw.chunks_exact(4).map(|b| inner.u32_of(b) as u64).collect())
    };
    let (offsets, lengths) = (longs(&mut file, 324)?, longs(&mut file, 325)?);
    let tiles: Vec<(u64, u64)> = offsets.iter().zip(&lengths).map(|(o, l)| (inner_at + o, *l)).collect();
    if tiles.iter().any(|(o, l)| o + l > block_at + block_len) {
        return Err(bad("a tile lies outside the block"));
    }
    // How the codes read: the file's XMP.
    let (_, xmp_len, xmp_at) = ifd0.array(TAG_XMP).ok_or_else(|| bad("the file has no XMP to say how the codes read"))?;
    let xmp = read_at(&mut file, xmp_at as u64, xmp_len as usize)?;
    let channels = channel_maps(&String::from_utf8_lossy(&xmp)).ok_or_else(|| bad("the XMP does not say how the codes read"))?;
    Ok(Some(EnhancedPicture { width, height, tile_w, tile_h, tiles, channels }))
}

/// The XMP packet of the DNG at `path`: what a raw converter wrote of its
/// settings for the frame, in the file itself or, when the file has none,
/// in a sidecar of the same name beside it. `None` when there is neither.
pub fn read_xmp(path: &Path) -> Result<Option<String>> {
    let embedded = (|| -> Result<Option<String>> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let head = read_at(&mut file, 0, 8.min(size as usize))?;
        let little = match head.get(..4) {
            Some([b'I', b'I', 42, 0]) => true,
            Some([b'M', b'M', 0, 42]) => false,
            _ => return Ok(None),
        };
        let at = if little { u32::from_le_bytes([head[4], head[5], head[6], head[7]]) } else { u32::from_be_bytes([head[4], head[5], head[6], head[7]]) } as u64;
        if at + 2 > size {
            return Ok(None);
        }
        let count = read_at(&mut file, at, 2)?;
        let n = if little { u16::from_le_bytes([count[0], count[1]]) } else { u16::from_be_bytes([count[0], count[1]]) } as usize;
        let mut tiff = head.clone();
        tiff[4..8].copy_from_slice(&if little { 8u32.to_le_bytes() } else { 8u32.to_be_bytes() });
        tiff.extend_from_slice(&read_at(&mut file, at, 2 + 12 * n)?);
        let Some(ifd0) = first_ifd(&tiff) else { return Ok(None) };
        let Some((_, len, xmp_at)) = ifd0.array(TAG_XMP) else { return Ok(None) };
        if len == 0 || len > 64 << 20 || xmp_at as u64 + len as u64 > size {
            return Ok(None);
        }
        Ok(Some(String::from_utf8_lossy(&read_at(&mut file, xmp_at as u64, len as usize)?).into_owned()))
    })()?;
    if embedded.is_some() {
        return Ok(embedded);
    }
    match std::fs::read(path.with_extension("xmp")) {
        Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
        Err(_) => Ok(None),
    }
}

/// Whether this build can decode an enhanced picture. Its tiles are JPEG XL,
/// read through the system's decoder on macOS; elsewhere there is none yet,
/// and a take's enhanced frames are left where they are: its frames come
/// from the capture ([`crate::x6_raw::enhanced_side`]). A file's settings
/// ([`read_xmp`]) are read everywhere.
pub fn decodable() -> bool {
    cfg!(target_os = "macos")
}

/// One JPEG XL codestream of 16-bit RGB samples: `(width, height, samples)`.
fn decode_tile(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>)> {
    #[cfg(target_os = "macos")]
    {
        crate::jxl_macos::decode_rgb16(bytes).map_err(Error::Ffmpeg)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = bytes;
        Err(bad("this build has no JPEG XL decoder on this system (macOS only for now)"))
    }
}

impl EnhancedPicture {
    /// Decode the picture of the file at `path` (the one [`find`] read)
    /// into `out`: `width × height` pixels of camera values as 16 bits (the
    /// sensor's white level = 65535), R, G, B and an opaque fourth sample.
    /// Tile rows decode on separate threads.
    pub fn decode_rgba16(&self, path: &Path, out: &mut [u16]) -> Result<()> {
        let (w, h, tw, th) = (self.width as usize, self.height as usize, self.tile_w as usize, self.tile_h as usize);
        if out.len() != w * h * 4 {
            return Err(bad("the buffer is not the picture's size"));
        }
        let across = w.div_ceil(tw);
        let tables: Vec<Vec<u16>> = self.channels.iter().map(ChannelMap::table).collect();
        let failed: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        let jobs = std::sync::Mutex::new(out.chunks_mut(th * w * 4).enumerate());
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, h.div_ceil(th));
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    let Ok(mut file) = File::open(path) else {
                        if let Ok(mut slot) = failed.lock() { slot.get_or_insert(bad("the file can no longer be opened")); }
                        return;
                    };
                    loop {
                        let job = jobs.lock().ok().and_then(|mut j| j.next());
                        let Some((row, band)) = job else { break };
                        let rows = band.len() / (w * 4);
                        let run = |file: &mut File, band: &mut [u16]| -> Result<()> {
                            for col in 0..across {
                                let (at, len) = self.tiles[row * across + col];
                                let bytes = read_at(file, at, len as usize)?;
                                let (dw, dh, samples) = decode_tile(&bytes)?;
                                let cols = tw.min(w - col * tw);
                                if dw < cols || dh < rows {
                                    return Err(bad(format!("tile {} decodes to {dw}x{dh}, at least {cols}x{rows} expected", row * across + col)));
                                }
                                for y in 0..rows {
                                    let src = &samples[y * dw * 3..(y * dw + cols) * 3];
                                    let dst = &mut band[(y * w + col * tw) * 4..(y * w + col * tw + cols) * 4];
                                    for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                                        d[0] = tables[0][s[0] as usize];
                                        d[1] = tables[1][s[1] as usize];
                                        d[2] = tables[2][s[2] as usize];
                                        d[3] = u16::MAX;
                                    }
                                }
                            }
                            Ok(())
                        };
                        if let Err(e) = run(&mut file, band) {
                            if let Ok(mut slot) = failed.lock() { slot.get_or_insert(e); }
                            break;
                        }
                    }
                });
            }
        });
        match failed.into_inner().ok().flatten() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XMP: &str = r#"<crs:ColorVariations><rdf:Seq><rdf:li rdf:parseType="Resource"><crs:Linearization><rdf:Seq>
        <rdf:li rdf:parseType="Resource"><crs:MapPolynomial><rdf:Seq><rdf:li>0/1</rdf:li><rdf:li>11772426/1073741824</rdf:li><rdf:li>0/1</rdf:li><rdf:li>176586395/1073741824</rdf:li></rdf:Seq></crs:MapPolynomial></rdf:li>
        <rdf:li rdf:parseType="Resource"><crs:MapPolynomial><rdf:Seq><rdf:li>0/1</rdf:li><rdf:li>15254426/1073741824</rdf:li><rdf:li>0/1</rdf:li><rdf:li>228816393/1073741824</rdf:li></rdf:Seq></crs:MapPolynomial></rdf:li>
        <rdf:li rdf:parseType="Resource"><crs:MapPolynomial><rdf:Seq><rdf:li>0/1</rdf:li><rdf:li>5253690/1073741824</rdf:li><rdf:li>0/1</rdf:li><rdf:li>78805346/1073741824</rdf:li></rdf:Seq></crs:MapPolynomial></rdf:li>
        </rdf:Seq></crs:Linearization><crs:BlackLevels><rdf:Seq><rdf:li crs:Black="1353,65535"/><rdf:li crs:Black="983,65535"/><rdf:li crs:Black="3058,65535"/></rdf:Seq></crs:BlackLevels>"#;

    #[test]
    fn the_codes_read_as_the_xmp_says() {
        let maps = channel_maps(XMP).expect("three channels");
        assert_eq!((maps[0].black, maps[0].white, maps[2].black), (1353.0, 65535.0, 3058.0));
        assert!((maps[0].poly[1] - 0.010_963_926).abs() < 1e-9 && (maps[1].poly[3] - 0.213_101_87).abs() < 1e-8 && maps[2].poly[0] == 0.0);
        // The sample's own pixels: codes and what they stand for (16-bit camera values).
        for (codes, want) in [([34919u16, 36914, 37334], [1917u16, 2927, 970]), ([47475, 49308, 50243], [4516, 6556, 2314]), ([7016, 6516, 9544], [71, 89, 39]), ([1261, 1164, 2625], [0, 3, 0])] {
            for c in 0..3 {
                assert_eq!(maps[c].table()[codes[c] as usize], want[c], "channel {c} code {}", codes[c]);
            }
        }
        // Black is zero, a code below it too; the top code is the channel's largest value.
        assert_eq!(maps[1].table()[983], 0);
        assert_eq!(maps[1].table()[0], 0);
        assert!((maps[1].camera(65535) - 0.227_31).abs() < 1e-4);
        // A packet that does not say it.
        assert!(channel_maps("<x:xmpmeta/>").is_none());
        assert!(channel_maps(&XMP.replace("crs:Black=\"983,65535\"", "")).is_none());
        assert_eq!(fraction(" 3/4 "), Some(0.75));
        assert_eq!(fraction("0.5"), Some(0.5));
        assert_eq!(fraction("1/0"), None);
    }

    /// A file laid out as Camera Raw lays one out, with made-up tiles.
    fn synthetic(dir: &Path, with_block: bool) -> std::path::PathBuf {
        let (w, h, tw, th) = (20u32, 14u32, 8u32, 8u32);
        let tiles: Vec<Vec<u8>> = (0..6u8).map(|i| vec![0xFF, 0x0A, i, i, i]).collect();
        // The block's own TIFF.
        let mut inner = b"II*\0\x08\0\0\0".to_vec();
        let entries: [(u16, u16, u32, u32); 9] = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 3, 0), (259, 3, 1, 52546), (277, 3, 1, 3), (322, 4, 1, tw), (323, 4, 1, th), (324, 4, 6, 0), (325, 4, 6, 0)];
        let tables_at = 8 + 2 + 12 * entries.len() as u32 + 4;
        let data_at = tables_at + 48;
        inner.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, kind, count, value) in entries {
            let value = match tag { 324 => tables_at, 325 => tables_at + 24, _ => value };
            inner.extend_from_slice(&tag.to_le_bytes());
            inner.extend_from_slice(&kind.to_le_bytes());
            inner.extend_from_slice(&count.to_le_bytes());
            inner.extend_from_slice(&if kind == 3 && count == 1 { (value as u16 as u32).to_le_bytes() } else { value.to_le_bytes() });
        }
        inner.extend_from_slice(&0u32.to_le_bytes());
        for i in 0..6u32 { inner.extend_from_slice(&(data_at + 5 * i).to_le_bytes()); }
        for _ in 0..6 { inner.extend_from_slice(&5u32.to_le_bytes()); }
        for t in &tiles { inner.extend_from_slice(t); }
        let mut block = Vec::new();
        for v in [5u32, 2] { block.extend_from_slice(&v.to_le_bytes()); }
        block.extend_from_slice(&[7u8; 16]);
        for v in [w, h, 3, 3, inner.len() as u32 + 16, 2, 1, 0, 0] { block.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(block.len(), HEADER);
        block.extend_from_slice(&inner);
        // The file: header, the block, the XMP, IFD0 at the end (as Camera Raw leaves it).
        let mut file = b"II*\0\0\0\0\0".to_vec();
        let block_at = file.len() as u32;
        file.extend_from_slice(&block);
        let xmp_at = file.len() as u32;
        file.extend_from_slice(XMP.as_bytes());
        let ifd_at = file.len() as u32;
        file[4..8].copy_from_slice(&ifd_at.to_le_bytes());
        let mut tags: Vec<(u16, u16, u32, u32)> = vec![(256, 4, 1, 10), (700, 1, XMP.len() as u32, xmp_at)];
        if with_block { tags.extend([(0xCD3D, 4, 1, block_at), (0xCD3E, 4, 1, block.len() as u32)]); }
        file.extend_from_slice(&(tags.len() as u16).to_le_bytes());
        for (tag, kind, count, value) in tags {
            file.extend_from_slice(&tag.to_le_bytes());
            file.extend_from_slice(&kind.to_le_bytes());
            file.extend_from_slice(&count.to_le_bytes());
            file.extend_from_slice(&value.to_le_bytes());
        }
        file.extend_from_slice(&0u32.to_le_bytes());
        let path = dir.join(if with_block { "enhanced.dng" } else { "plain.dng" });
        std::fs::write(&path, file).unwrap();
        path
    }

    #[test]
    fn a_files_picture_is_found_where_it_is() {
        let dir = std::env::temp_dir().join(format!("vr180-enhanced-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let picture = find(&synthetic(&dir, true)).unwrap().expect("a picture");
        assert_eq!((picture.width, picture.height, picture.tile_w, picture.tile_h), (20, 14, 8, 8));
        assert_eq!(picture.tiles.len(), 6);
        // The tiles are where the block's TIFF lists them, counted from the file's start.
        let bytes = std::fs::read(dir.join("enhanced.dng")).unwrap();
        for (i, (at, len)) in picture.tiles.iter().enumerate() {
            assert_eq!(&bytes[*at as usize..(*at + *len) as usize], &[0xFF, 0x0A, i as u8, i as u8, i as u8]);
        }
        assert_eq!(picture.channels[2].black, 3058.0);
        // A DNG nobody enhanced, and a file that is no TIFF.
        assert_eq!(find(&synthetic(&dir, false)).unwrap(), None);
        std::fs::write(dir.join("other.dng"), b"not a tiff at all").unwrap();
        assert_eq!(find(&dir.join("other.dng")).unwrap(), None);
        // Made-up tiles do not decode: the error says so instead of a picture of nothing.
        let mut out = vec![0u16; 20 * 14 * 4];
        assert!(picture.decode_rgba16(&dir.join("enhanced.dng"), &mut out).is_err());
        assert!(picture.decode_rgba16(&dir.join("enhanced.dng"), &mut [0u16; 8]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The sample frame Camera Raw 18.7 enhanced (skipped when the volume is
    /// not mounted or the file is no longer the enhanced one).
    #[test]
    fn the_sample_frame_decodes_to_its_camera_values() {
        let path = Path::new("/Volumes/Database/x6/1115 raw video/RAWVIDEO-20261116-011723_OriginalDNG/CAM0/000000364.dng");
        let Some(picture) = path.exists().then(|| find(path).unwrap()).flatten() else { return };
        assert_eq!((picture.width, picture.height, picture.tile_w, picture.tile_h, picture.tiles.len()), (7680, 7680, 768, 864, 90));
        if !cfg!(target_os = "macos") { return; }
        let mut out = vec![0u16; 7680 * 7680 * 4];
        let t = std::time::Instant::now();
        picture.decode_rgba16(path, &mut out).unwrap();
        eprintln!("decoded 7680x7680 in {:.2?}", t.elapsed());
        let px = |x: usize, y: usize| [out[(y * 7680 + x) * 4], out[(y * 7680 + x) * 4 + 1], out[(y * 7680 + x) * 4 + 2]];
        // What another decoder of the same file gave (16-bit camera values); a lossy
        // codestream decodes to within a few codes from one decoder to the next.
        for ((x, y), want) in [((2000usize, 1000usize), [1917u16, 2927, 970]), ((6000, 5000), [2414, 3423, 1225]), ((4100, 2500), [4516, 6556, 2314]), ((3840, 3840), [71, 89, 39]), ((768, 6912), [1, 0, 0])] {
            let got = px(x, y);
            assert!(got.iter().zip(&want).all(|(g, w)| (*g as i32 - *w as i32).abs() <= 6), "({x}, {y}): {got:?}, want {want:?}");
        }
        let mut mean = [0f64; 3];
        for p in out.chunks_exact(4) {
            for c in 0..3 { mean[c] += p[c] as f64; }
            assert_eq!(p[3], u16::MAX);
        }
        let mean = mean.map(|m| m / (7680.0 * 7680.0));
        assert!((mean[0] - 1603.139).abs() < 1.0 && (mean[1] - 2318.545).abs() < 1.0 && (mean[2] - 780.984).abs() < 1.0, "{mean:?}");
    }
}
