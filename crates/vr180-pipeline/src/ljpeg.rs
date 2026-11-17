//! Lossless JPEG (SOF3, predictor 1) tile encoder for the DNG writer.
//!
//! One tile → one complete JPEG stream (ITU T.81 process 14):
//!
//! * SOI, SOF3 with P = 16, Nf = 3 interleaved components (ids 0, 1, 2,
//!   H = V = 1, Tq = 0), one DHT per component (Tc = 0, Th = 0/1/2), SOS with
//!   Td = 0/1/2, Ss = 1 (predictor 1 = Ra), Se = 0, Ah/Al = 0, entropy-coded
//!   data with FF→FF 00 byte stuffing, the final byte padded with 1-bits, EOI.
//! * Prediction: the sample to the left; the first column of each row uses the
//!   first sample of the row above; the very first sample is predicted from
//!   2^(P−1) = 32768.  Differences are reduced modulo 2^16 to −32768..=32767,
//!   SSSS = bit length of |diff| (SSSS 16 carries no extra bits).
//! * Each component's Huffman table is built from that component's own SSSS
//!   histogram: a plain Huffman code-length pass with deterministic tie-breaks,
//!   the longest code lengthened by one so that no code is all ones (the JPEG
//!   rule), falling back to a fixed length set when a length would exceed 16;
//!   codes are assigned canonically (BITS/HUFFVAL).
//!
//! This is a byte-for-byte port of `ljpeg_encode` in the validated Python
//! reference (`lindng.py --ljpeg-ids 0 --ljpeg-per-comp-dht`), whose tiles are
//! identical to the ones Adobe DNG Converter writes and decode in macOS
//! ImageIO, exiftool and Adobe Camera Raw.  The fixture test below pins the
//! output against a tile produced by that reference.
//!
//! No `unsafe`, no platform-specific code, no allocations beyond the output
//! buffer (callers on the hot path use [`encode_tile_rgb16_into`] with reused
//! scratch buffers).

/// Fallback code lengths for SSSS 0..=16 when the Huffman pass would produce
/// a code longer than 16 bits (Kraft sum < 1, so no code is all ones).
const FIXED_LENGTHS: [u8; 17] = [2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16];

/// Number of interleaved components in a tile (R, G, B).
const NCOMP: usize = 3;

/// A canonical DC Huffman table for the 17 SSSS categories of one component.
#[derive(Clone, Debug)]
struct HuffTable {
    /// `BITS[l]` for l = 1..=16: number of codes of length `l` (index 0 unused).
    bits: [u8; 17],
    /// `HUFFVAL`: the symbols in canonical order.
    huffval: Vec<u8>,
    /// `(code, length)` per SSSS; length 0 means the symbol has no code.
    codes: [(u16, u8); 17],
}

/// Huffman code lengths for the SSSS symbols with non-zero frequency.
///
/// Returns one length per symbol (0 = absent).  Port of `huff_lengths` in the
/// reference: pairwise merging of the two lowest-weight nodes, ties broken by
/// creation order (leaves first, in symbol order, then merged nodes in the
/// order they were created), then the last canonical code is lengthened by
/// one so the tree is not full.  If any length exceeds 16 the fixed set is
/// used for all 17 symbols.
fn huff_lengths(freq: &[u64; 17]) -> [u8; 17] {
    let mut lengths = [0u8; 17];
    let syms: Vec<u8> = (0..17u8).filter(|&s| freq[s as usize] > 0).collect();
    if syms.len() <= 1 {
        let s = syms.first().copied().unwrap_or(0);
        lengths[s as usize] = 1;
        return lengths;
    }
    // (weight, sequence number, member symbols) — the sequence number makes the
    // ordering total, so the merge order is fully deterministic.
    let mut nodes: Vec<(u64, u32, Vec<u8>)> = syms
        .iter()
        .enumerate()
        .map(|(i, &s)| (freq[s as usize], i as u32, vec![s]))
        .collect();
    let mut seq = nodes.len() as u32;
    let mut depth = [0u32; 17];
    while nodes.len() > 1 {
        let a = take_min(&mut nodes);
        let b = take_min(&mut nodes);
        for &s in a.2.iter().chain(b.2.iter()) {
            depth[s as usize] += 1;
        }
        seq += 1;
        let mut members = a.2;
        members.extend_from_slice(&b.2);
        nodes.push((a.0 + b.0, seq, members));
    }
    // The symbol with the longest code (largest symbol value on ties) gets one
    // more bit: that is the last canonical code, which would otherwise be all
    // ones.
    let longest = syms
        .iter()
        .copied()
        .max_by_key(|&s| (depth[s as usize], s))
        .expect("at least two symbols");
    depth[longest as usize] += 1;
    if depth.iter().any(|&d| d > 16) {
        return FIXED_LENGTHS;
    }
    for &s in &syms {
        lengths[s as usize] = depth[s as usize] as u8;
    }
    lengths
}

/// Remove and return the node with the smallest `(weight, seq)` key.
fn take_min(nodes: &mut Vec<(u64, u32, Vec<u8>)>) -> (u64, u32, Vec<u8>) {
    let mut best = 0;
    for i in 1..nodes.len() {
        if (nodes[i].0, nodes[i].1) < (nodes[best].0, nodes[best].1) {
            best = i;
        }
    }
    nodes.swap_remove(best)
}

/// Canonical code assignment (ITU T.81 Annex C) from per-symbol lengths.
fn canonical(lengths: &[u8; 17]) -> HuffTable {
    let mut order: Vec<(u8, u8)> = (0..17u8)
        .filter(|&s| lengths[s as usize] > 0)
        .map(|s| (s, lengths[s as usize]))
        .collect();
    order.sort_by_key(|&(s, l)| (l, s));
    let mut bits = [0u8; 17];
    for &(_, l) in &order {
        bits[l as usize] += 1;
    }
    let huffval: Vec<u8> = order.iter().map(|&(s, _)| s).collect();
    let mut codes = [(0u16, 0u8); 17];
    let mut code: u32 = 0;
    let mut i = 0usize;
    for l in 1..=16usize {
        for _ in 0..bits[l] {
            codes[huffval[i] as usize] = (code as u16, l as u8);
            i += 1;
            code += 1;
        }
        code <<= 1;
    }
    HuffTable { bits, huffval, codes }
}

/// MSB-first bit packer with JPEG byte stuffing.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u32,
    n: u32,
}

impl<'a> BitWriter<'a> {
    #[inline]
    fn put(&mut self, value: u32, nbits: u32) {
        if nbits == 0 {
            return;
        }
        // nbits <= 16 and n < 8 on entry, so acc never exceeds 24 bits.
        self.acc = (self.acc << nbits) | (value & ((1u32 << nbits) - 1));
        self.n += nbits;
        while self.n >= 8 {
            let b = ((self.acc >> (self.n - 8)) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0); // byte stuffing
            }
            self.n -= 8;
        }
        self.acc &= (1u32 << self.n) - 1;
    }

    /// Pad the final partial byte with 1-bits.
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1u32 << pad) - 1, pad);
        }
    }
}

/// SSSS category of a difference: the bit length of |diff| (0..=16).
#[inline]
fn ssss_of(d: i16) -> u32 {
    let a = (d as i32).unsigned_abs();
    32 - a.leading_zeros()
}

/// Encode one tile of interleaved 16-bit RGB samples as a lossless JPEG
/// stream (see the module docs for the exact stream layout).
///
/// `tile.len()` must equal `width * height * 3`; `width` and `height` must be
/// in `1..=65535`.
///
/// # Panics
/// On a size mismatch or an out-of-range dimension.
pub fn encode_tile_rgb16(tile: &[u16], width: usize, height: usize) -> Vec<u8> {
    let mut diffs = Vec::with_capacity(tile.len());
    let mut out = Vec::new();
    encode_tile_rgb16_into(tile, width, height, &mut diffs, &mut out);
    out
}

/// Scratch-buffer variant of [`encode_tile_rgb16`] for callers encoding many
/// tiles: `diffs` is reused across calls, `out` is cleared and filled with the
/// complete JPEG stream.
pub fn encode_tile_rgb16_into(
    tile: &[u16],
    width: usize,
    height: usize,
    diffs: &mut Vec<i16>,
    out: &mut Vec<u8>,
) {
    assert!(
        (1..=65535).contains(&width) && (1..=65535).contains(&height),
        "ljpeg: tile dimensions {width}x{height} out of range"
    );
    assert_eq!(
        tile.len(),
        width * height * NCOMP,
        "ljpeg: tile has {} samples, expected {}x{}x3",
        tile.len(),
        width,
        height
    );
    let row_len = width * NCOMP;

    // ---- pass 1: prediction residuals + per-component SSSS histograms -----
    diffs.clear();
    diffs.reserve(tile.len());
    let mut freq = [[0u64; 17]; NCOMP];
    let mut prev_row: &[u16] = &[];
    for (y, row) in tile.chunks_exact(row_len).enumerate() {
        for (x, px) in row.chunks_exact(NCOMP).enumerate() {
            for (c, &s) in px.iter().enumerate() {
                let pred = if x > 0 {
                    row[(x - 1) * NCOMP + c]
                } else if y > 0 {
                    prev_row[c]
                } else {
                    1u16 << 15
                };
                let d = s.wrapping_sub(pred) as i16;
                freq[c][ssss_of(d) as usize] += 1;
                diffs.push(d);
            }
        }
        prev_row = row;
    }
    let tabs: Vec<HuffTable> = freq.iter().map(|f| canonical(&huff_lengths(f))).collect();

    // ---- headers ----------------------------------------------------------
    out.clear();
    out.reserve(tile.len() * 2 + 128);
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    // SOF3: Lf = 8 + 3*Nf, P = 16, Y, X, Nf, then (Ci, HiVi = 0x11, Tqi = 0).
    let lf = (8 + 3 * NCOMP) as u16;
    out.extend_from_slice(&[0xFF, 0xC3]);
    out.extend_from_slice(&lf.to_be_bytes());
    out.push(16);
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(NCOMP as u8);
    for c in 0..NCOMP {
        out.extend_from_slice(&[c as u8, 0x11, 0]);
    }
    // DHT, one per component: Tc = 0 (DC), Th = component index.
    for (t, tab) in tabs.iter().enumerate() {
        let lh = (2 + 17 + tab.huffval.len()) as u16;
        out.extend_from_slice(&[0xFF, 0xC4]);
        out.extend_from_slice(&lh.to_be_bytes());
        out.push(t as u8);
        out.extend_from_slice(&tab.bits[1..17]);
        out.extend_from_slice(&tab.huffval);
    }
    // SOS: Ns = 3, (Cs, Td<<4 | Ta = t<<4), Ss = 1 (predictor 1), Se = 0, AhAl = 0.
    let ls = (6 + 2 * NCOMP) as u16;
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&ls.to_be_bytes());
    out.push(NCOMP as u8);
    for c in 0..NCOMP {
        out.extend_from_slice(&[c as u8, (c as u8) << 4]);
    }
    out.extend_from_slice(&[1, 0, 0]);

    // ---- pass 2: entropy-coded data ---------------------------------------
    let codes: [[(u16, u8); 17]; NCOMP] = [tabs[0].codes, tabs[1].codes, tabs[2].codes];
    let mut bw = BitWriter { out, acc: 0, n: 0 };
    for px in diffs.chunks_exact(NCOMP) {
        for (c, &d) in px.iter().enumerate() {
            let ssss = ssss_of(d);
            let (code, len) = codes[c][ssss as usize];
            bw.put(code as u32, len as u32);
            if ssss > 0 && ssss < 16 {
                let v = if d > 0 { d as i32 } else { d as i32 - 1 };
                bw.put(v as u32, ssss);
            }
        }
    }
    bw.flush();
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
}

// ---------------------------------------------------------------------------
// Bayer mosaics
// ---------------------------------------------------------------------------

/// MSB-first bit packer for the mosaic encoder: a 64-bit accumulator emptied
/// four bytes at a time, with JPEG byte stuffing.
struct WideBits<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    n: u32,
}

impl WideBits<'_> {
    /// Append the low `nbits` bits of `value` (`nbits` ≤ 32, `value` holds
    /// nothing above them).
    #[inline(always)]
    fn put(&mut self, value: u32, nbits: u32) {
        self.acc = (self.acc << nbits) | value as u64;
        self.n += nbits;
        if self.n >= 32 {
            self.n -= 32;
            let word = (self.acc >> self.n) as u32;
            // No byte of the word is FF: the common case, written whole.
            let inv = !word;
            if inv.wrapping_sub(0x0101_0101) & !inv & 0x8080_8080 == 0 {
                self.out.extend_from_slice(&word.to_be_bytes());
            } else {
                for b in word.to_be_bytes() {
                    self.out.push(b);
                    if b == 0xFF {
                        self.out.push(0);
                    }
                }
            }
        }
    }

    /// Write what is left, the last byte padded with 1-bits.
    fn finish(mut self) {
        while self.n >= 8 {
            self.n -= 8;
            let b = (self.acc >> self.n) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0);
            }
        }
        if self.n > 0 {
            let pad = 8 - self.n;
            let b = (((self.acc << pad) as u8) | ((1u16 << pad) - 1) as u8) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0);
            }
        }
    }
}

/// Encode one tile of a Bayer mosaic as a lossless JPEG stream: `width`
/// sites a row (an even number), `height` rows, `precision` bits a sample
/// (2..=16).
///
/// The two sites of a row's 2-site group are the two components of a frame
/// `width / 2` wide, so every sample is predicted from the sample of its own
/// colour two sites to its left (the first group of a row from the first
/// group of the row above, the very first from 2^(precision−1)) — the
/// layout raw converters expect of a mosaic in a DNG. Otherwise the stream
/// is the one of [`encode_tile_rgb16_into`]: SOF3, a Huffman table per
/// component from its own statistics, one scan, predictor 1.
///
/// `out` is cleared and filled with the complete stream.
///
/// # Panics
/// On a size mismatch, an odd width or a dimension out of range.
pub fn encode_tile_cfa_into(tile: &[u16], width: usize, height: usize, precision: u8, out: &mut Vec<u8>) {
    assert!(
        width % 2 == 0 && (2..=131070).contains(&width) && (1..=65535).contains(&height),
        "ljpeg: mosaic tile dimensions {width}x{height} out of range"
    );
    assert!((2..=16).contains(&precision), "ljpeg: precision {precision} out of range");
    assert_eq!(tile.len(), width * height, "ljpeg: mosaic tile has {} samples, expected {}x{}", tile.len(), width, height);

    // The residual of every sample, row by row: `f(component, residual)`.
    #[inline(always)]
    fn residuals(tile: &[u16], width: usize, first: u16, mut f: impl FnMut(usize, i16)) {
        let mut above = [first, first];
        for row in tile.chunks_exact(width) {
            f(0, row[0].wrapping_sub(above[0]) as i16);
            f(1, row[1].wrapping_sub(above[1]) as i16);
            above = [row[0], row[1]];
            for (prev, cur) in row.chunks_exact(2).zip(row[2..].chunks_exact(2)) {
                f(0, cur[0].wrapping_sub(prev[0]) as i16);
                f(1, cur[1].wrapping_sub(prev[1]) as i16);
            }
        }
    }
    let first = 1u16 << (precision - 1);

    // ---- pass 1: the SSSS histogram of each component ----------------------
    let mut freq = [[0u64; 17]; 2];
    residuals(tile, width, first, |c, d| freq[c][ssss_of(d) as usize] += 1);
    let tabs = [canonical(&huff_lengths(&freq[0])), canonical(&huff_lengths(&freq[1]))];

    // ---- headers ----------------------------------------------------------
    out.clear();
    out.reserve(tile.len() * 2 + 128);
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    out.extend_from_slice(&[0xFF, 0xC3, 0, 14, precision]); // SOF3, Lf = 8 + 3·2
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&((width / 2) as u16).to_be_bytes());
    out.extend_from_slice(&[2, 0, 0x11, 0, 1, 0x11, 0]);
    for (t, tab) in tabs.iter().enumerate() {
        let lh = (2 + 17 + tab.huffval.len()) as u16;
        out.extend_from_slice(&[0xFF, 0xC4]);
        out.extend_from_slice(&lh.to_be_bytes());
        out.push(t as u8);
        out.extend_from_slice(&tab.bits[1..17]);
        out.extend_from_slice(&tab.huffval);
    }
    // SOS: Ns = 2, (Cs, Td<<4), Ss = 1 (predictor 1), Se = 0, AhAl = 0.
    out.extend_from_slice(&[0xFF, 0xDA, 0, 10, 2, 0, 0x00, 1, 0x10, 1, 0, 0]);

    // ---- pass 2: entropy-coded data ---------------------------------------
    let codes = [tabs[0].codes, tabs[1].codes];
    let mut bits = WideBits { out, acc: 0, n: 0 };
    residuals(tile, width, first, |c, d| {
        let ssss = ssss_of(d);
        let (code, len) = codes[c][ssss as usize];
        if ssss == 0 || ssss == 16 {
            bits.put(code as u32, len as u32);
        } else {
            let v = (if d > 0 { d as i32 } else { d as i32 - 1 }) as u32 & ((1u32 << ssss) - 1);
            bits.put(((code as u32) << ssss) | v, len as u32 + ssss);
        }
    });
    bits.finish();
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
}

// ---------------------------------------------------------------------------
// Test support: an independent decoder (port of ljpeg_check.py) and the
// reference synthetic image generator (port of lindng.py's synth_image), both
// shared with the DNG writer tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;

    /// Decoded lossless-JPEG stream: `(height, width, ncomp, samples)` with
    /// samples interleaved row-major.
    #[derive(Debug)]
    pub(crate) struct Decoded {
        pub height: usize,
        pub width: usize,
        pub ncomp: usize,
        pub samples: Vec<u16>,
    }

    fn be16(d: &[u8], pos: usize) -> usize {
        ((d[pos] as usize) << 8) | d[pos + 1] as usize
    }

    /// Decode one SOF3 stream exactly the way `ljpeg_check.py` does: walks the
    /// marker segments, builds the (length, code) → symbol maps, decodes with
    /// predictor 1 and checks that EOI directly follows the entropy data.
    pub(crate) fn ljpeg_decode(d: &[u8]) -> Decoded {
        assert_eq!(&d[..2], &[0xFF, 0xD8], "missing SOI");
        let mut pos = 2usize;
        let mut tables: HashMap<u8, HashMap<(u32, u32), u8>> = HashMap::new();
        let (mut p, mut y, mut x, mut nf) = (0u8, 0usize, 0usize, 0usize);
        let mut comps: Vec<(u8, u8, u8)> = Vec::new();
        // The loop yields the scan header: (component selectors, Ss, Se, AhAl).
        let (sel, ss, se, ahal): (Vec<(u8, u8)>, u8, u8, u8) = loop {
            assert_eq!(d[pos], 0xFF, "expected marker at {pos}");
            let marker = d[pos + 1];
            pos += 2;
            let len = be16(d, pos);
            match marker {
                0xC3 => {
                    p = d[pos + 2];
                    y = be16(d, pos + 3);
                    x = be16(d, pos + 5);
                    nf = d[pos + 7] as usize;
                    comps = (0..nf)
                        .map(|i| (d[pos + 8 + 3 * i], d[pos + 9 + 3 * i], d[pos + 10 + 3 * i]))
                        .collect();
                }
                0xC4 => {
                    let (mut q, end) = (pos + 2, pos + len);
                    while q < end {
                        let tcth = d[q];
                        let bits: Vec<usize> = d[q + 1..q + 17].iter().map(|&b| b as usize).collect();
                        let nsym: usize = bits.iter().sum();
                        let vals = &d[q + 17..q + 17 + nsym];
                        q += 17 + nsym;
                        let mut m = HashMap::new();
                        let (mut code, mut i) = (0u32, 0usize);
                        for l in 1..=16u32 {
                            for _ in 0..bits[(l - 1) as usize] {
                                m.insert((l, code), vals[i]);
                                i += 1;
                                code += 1;
                            }
                            code <<= 1;
                        }
                        tables.insert(tcth & 0x0F, m);
                    }
                }
                0xDA => {
                    let ns = d[pos + 2] as usize;
                    let sel = (0..ns).map(|i| (d[pos + 3 + 2 * i], d[pos + 4 + 2 * i] >> 4)).collect();
                    let hdr = (sel, d[pos + 3 + 2 * ns], d[pos + 4 + 2 * ns], d[pos + 5 + 2 * ns]);
                    pos += len;
                    break hdr;
                }
                m => panic!("unexpected marker {m:02x}"),
            }
            pos += len;
        };
        assert!(ss == 1 && se == 0 && ahal == 0 && sel.len() == nf, "unsupported scan");
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(c.0, sel[i].0, "component id order differs between SOF3 and SOS");
            assert_eq!(c.1, 0x11, "H/V must be 1/1");
        }

        struct Bits<'a> {
            d: &'a [u8],
            pos: usize,
            acc: u8,
            n: u32,
        }
        impl<'a> Bits<'a> {
            fn bit(&mut self) -> u32 {
                if self.n == 0 {
                    let b = self.d[self.pos];
                    self.pos += 1;
                    if b == 0xFF {
                        assert_eq!(self.d[self.pos], 0x00, "marker inside entropy data");
                        self.pos += 1;
                    }
                    self.acc = b;
                    self.n = 8;
                }
                self.n -= 1;
                ((self.acc >> self.n) & 1) as u32
            }
            fn bits(&mut self, n: u32) -> u32 {
                let mut v = 0u32;
                for _ in 0..n {
                    v = (v << 1) | self.bit();
                }
                v
            }
        }
        let mut br = Bits { d, pos, acc: 0, n: 0 };
        let mut out = vec![0u16; y * x * nf];
        for yy in 0..y {
            for xx in 0..x {
                for c in 0..nf {
                    let tbl = &tables[&sel[c].1];
                    let (mut code, mut l) = (0u32, 0u32);
                    let ssss = loop {
                        code = (code << 1) | br.bit();
                        l += 1;
                        if let Some(&s) = tbl.get(&(l, code)) {
                            break s as u32;
                        }
                        assert!(l <= 16, "bad code");
                    };
                    let diff: i32 = if ssss == 0 {
                        0
                    } else if ssss == 16 {
                        32768
                    } else {
                        let v = br.bits(ssss) as i32;
                        if v >= (1 << (ssss - 1)) {
                            v
                        } else {
                            v - (1 << ssss) + 1
                        }
                    };
                    let pred: i32 = if xx > 0 {
                        out[(yy * x + xx - 1) * nf + c] as i32
                    } else if yy > 0 {
                        out[((yy - 1) * x) * nf + c] as i32
                    } else {
                        1 << (p as i32 - 1)
                    };
                    out[(yy * x + xx) * nf + c] = ((pred + diff) & 0xFFFF) as u16;
                }
            }
        }
        assert_eq!(
            &d[br.pos..br.pos + 2],
            &[0xFF, 0xD9],
            "EOI not found right after entropy data"
        );
        assert_eq!(br.pos + 2, d.len(), "trailing bytes after EOI");
        Decoded { height: y, width: x, ncomp: nf, samples: out }
    }

    // Facts copied from the Insta360 X6 sample, exactly as lindng.py has them.
    pub(crate) const AS_SHOT_NEUTRAL: [f64; 3] = [0.4599609375, 1.0, 0.537109375];

    /// `2.0 ** (-BASELINE_EXPOSURE) * 65535.0` with BaselineExposure 2.6, as
    /// Python computes it (bit pattern captured from CPython so the generator
    /// is identical on every platform, independent of the local `pow`).
    const SYNTH_SCALE: f64 = f64::from_bits(0x40c51c9f3704ffb3);

    /// Exact port of `lindng.py::synth_image`: a horizontal grey ramp, neutral
    /// at the top, tinted towards the bottom, stored as camera values
    /// `scene * AsShotNeutral[c] * 2^-BaselineExposure * 65535` (Python
    /// `round` = ties-to-even).  Interleaved RGB, `w * h * 3` samples.
    pub(crate) fn synth_image(w: usize, h: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            let t = y as f64 / (h.max(2) - 1) as f64;
            for x in 0..w {
                let g = x as f64 / (w.max(2) - 1) as f64;
                let scene = [g * (1.0 - 0.6 * t), g, g * (1.0 - 0.3 * t)];
                for c in 0..3 {
                    let v = (scene[c] * AS_SHOT_NEUTRAL[c] * SYNTH_SCALE).round_ties_even();
                    out.push(v.clamp(0.0, 65535.0) as u16);
                }
            }
        }
        out
    }

    /// Cut a `tw x th` tile at (`x0`, `y0`) out of an interleaved RGB image
    /// (no padding; the caller keeps it in bounds).
    pub(crate) fn cut_tile(img: &[u16], w: usize, x0: usize, y0: usize, tw: usize, th: usize) -> Vec<u16> {
        let mut t = Vec::with_capacity(tw * th * 3);
        for y in y0..y0 + th {
            t.extend_from_slice(&img[(y * w + x0) * 3..(y * w + x0 + tw) * 3]);
        }
        t
    }

    /// Small deterministic PRNG (xorshift64*) for the round-trip tests.
    pub(crate) struct Rng(pub u64);
    impl Rng {
        pub(crate) fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    /// Tile 0 (128x128, top-left) of `lindng.py --size 256x128 --comp 7
    /// --tile 128x128 --ljpeg-ids 0 --ljpeg-per-comp-dht` (t_c7_D_all.dng).
    const FIXTURE_TILE0: &[u8] = include_bytes!("../tests/fixtures/ljpeg_t_c7_D_all_tile0.bin");

    #[test]
    fn fixture_tile0_is_byte_identical_to_python_reference() {
        let img = synth_image(256, 128);
        let tile = cut_tile(&img, 256, 0, 0, 128, 128);
        let enc = encode_tile_rgb16(&tile, 128, 128);
        assert_eq!(enc.len(), FIXTURE_TILE0.len(), "stream length differs");
        if enc != FIXTURE_TILE0 {
            let first = enc.iter().zip(FIXTURE_TILE0).position(|(a, b)| a != b).unwrap();
            panic!("stream differs from the Python reference at byte {first}");
        }
    }

    #[test]
    fn fixture_tile0_decodes_to_the_synthetic_tile() {
        let img = synth_image(256, 128);
        let tile = cut_tile(&img, 256, 0, 0, 128, 128);
        let dec = ljpeg_decode(FIXTURE_TILE0);
        assert_eq!((dec.height, dec.width, dec.ncomp), (128, 128, 3));
        assert_eq!(dec.samples, tile);
    }

    fn roundtrip(tile: &[u16], w: usize, h: usize) -> Vec<u8> {
        let enc = encode_tile_rgb16(tile, w, h);
        let dec = ljpeg_decode(&enc);
        assert_eq!((dec.height, dec.width, dec.ncomp), (h, w, 3));
        assert_eq!(dec.samples, tile, "round-trip mismatch for {w}x{h}");
        enc
    }

    #[test]
    fn roundtrip_random_tiles() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for &(w, h) in &[(1usize, 1usize), (1, 7), (7, 1), (3, 5), (16, 16), (37, 29), (64, 64), (256, 256)] {
            // Full-range noise (worst case: SSSS 15/16 everywhere).
            let noise: Vec<u16> = (0..w * h * 3).map(|_| rng.next_u32() as u16).collect();
            roundtrip(&noise, w, h);
            // Smooth-ish content with occasional jumps.
            let mut smooth = Vec::with_capacity(w * h * 3);
            let mut v = [20000i32, 30000, 40000];
            for _ in 0..w * h {
                for c in 0..3 {
                    let step = (rng.next_u32() % 41) as i32 - 20;
                    v[c] = if rng.next_u32() % 97 == 0 { (rng.next_u32() & 0xFFFF) as i32 } else { (v[c] + step).clamp(0, 65535) };
                    smooth.push(v[c] as u16);
                }
            }
            roundtrip(&smooth, w, h);
        }
    }

    #[test]
    fn roundtrip_edge_values_and_constant_tiles() {
        for &v in &[0u16, 1, 32767, 32768, 65534, 65535] {
            let tile = vec![v; 32 * 24 * 3];
            let enc = roundtrip(&tile, 32, 24);
            // A constant tile: first sample is one SSSS-16 or small-diff code,
            // the rest are SSSS-0 single-bit codes -> tiny stream.
            assert!(enc.len() < 32 * 24 * 3 / 8 + 128, "constant tile not compact: {} bytes", enc.len());
        }
        // Alternating extremes: every diff is ±65535 mod 2^16 = ±1 or 32768.
        let mut tile = Vec::new();
        for i in 0..16 * 16 * 3 {
            tile.push(if i % 2 == 0 { 0 } else { 65535 });
        }
        roundtrip(&tile, 16, 16);
        // Diffs of exactly -32768 / +32767 (SSSS 16 and 15 boundaries).
        let mut tile = Vec::new();
        for i in 0..8 * 8 * 3 {
            tile.push(match i % 4 { 0 => 0, 1 => 32768, 2 => 1, _ => 32768 });
        }
        roundtrip(&tile, 8, 8);
        // Per-component different statistics (one table each).
        let mut tile = Vec::new();
        for i in 0..40 * 40 {
            tile.push(100 + (i % 3) as u16);
            tile.push(((i * 7919) % 65536) as u16);
            tile.push(65535);
        }
        roundtrip(&tile, 40, 40);
    }

    #[test]
    fn stream_structure_matches_recipe() {
        let img = synth_image(64, 32);
        let enc = encode_tile_rgb16(&img, 64, 32);
        // SOI, SOF3 with P=16, Y=32, X=64, Nf=3, ids 0/1/2 with H=V=1.
        assert_eq!(&enc[..4], &[0xFF, 0xD8, 0xFF, 0xC3]);
        assert_eq!(&enc[4..6], &[0x00, 0x11]);
        assert_eq!(enc[6], 16);
        assert_eq!(&enc[7..9], &[0, 32]);
        assert_eq!(&enc[9..11], &[0, 64]);
        assert_eq!(enc[11], 3);
        assert_eq!(&enc[12..21], &[0, 0x11, 0, 1, 0x11, 0, 2, 0x11, 0]);
        // Three DHT segments (Th 0, 1, 2) then SOS with Td 0/1/2 and Ss=1.
        let mut pos = 21;
        for t in 0..3u8 {
            assert_eq!(&enc[pos..pos + 2], &[0xFF, 0xC4]);
            let len = ((enc[pos + 2] as usize) << 8) | enc[pos + 3] as usize;
            assert_eq!(enc[pos + 4], t);
            let nsym: usize = enc[pos + 5..pos + 21].iter().map(|&b| b as usize).sum();
            assert_eq!(len, 19 + nsym);
            pos += 2 + len;
        }
        assert_eq!(&enc[pos..pos + 2], &[0xFF, 0xDA]);
        assert_eq!(&enc[pos + 2..pos + 14], &[0x00, 0x0C, 3, 0, 0x00, 1, 0x10, 2, 0x20, 1, 0, 0]);
        assert_eq!(&enc[enc.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn huffman_no_code_is_all_ones() {
        // Any histogram: the canonical table must never contain an all-ones
        // code (reserved as a marker prefix in JPEG).
        let mut rng = Rng(7);
        for _ in 0..200 {
            let mut freq = [0u64; 17];
            for f in freq.iter_mut() {
                if rng.next_u32() % 3 != 0 {
                    *f = (rng.next_u32() % 1000) as u64;
                }
            }
            let tab = canonical(&huff_lengths(&freq));
            for &(code, len) in &tab.codes {
                if len > 0 {
                    assert_ne!(code as u32, (1u32 << len) - 1, "all-ones code of length {len}");
                }
            }
        }
        // Degenerate skew that overflows 16 bits -> fixed lengths.
        let mut freq = [0u64; 17];
        for (i, f) in freq.iter_mut().enumerate() {
            *f = 1u64 << (2 * i);
        }
        assert_eq!(huff_lengths(&freq), FIXED_LENGTHS);
        // Single symbol -> one code of length 1.
        let mut freq = [0u64; 17];
        freq[5] = 10;
        let l = huff_lengths(&freq);
        assert_eq!(l[5], 1);
        assert_eq!(l.iter().filter(|&&x| x > 0).count(), 1);
    }

    fn roundtrip_cfa(tile: &[u16], w: usize, h: usize, precision: u8) -> Vec<u8> {
        let mut enc = vec![9, 9];
        encode_tile_cfa_into(tile, w, h, precision, &mut enc);
        let dec = ljpeg_decode(&enc);
        assert_eq!((dec.height, dec.width, dec.ncomp), (h, w / 2, 2));
        assert_eq!(dec.samples, tile, "mosaic round-trip mismatch for {w}x{h} at {precision} bits");
        enc
    }

    #[test]
    fn a_mosaic_tile_comes_back_sample_for_sample() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for &(w, h) in &[(2usize, 1usize), (2, 9), (8, 1), (6, 5), (64, 16), (3840, 160)] {
            // Every 14-bit code, at random.
            let noise: Vec<u16> = (0..w * h).map(|_| (rng.next_u32() & 0x3FFF) as u16).collect();
            roundtrip_cfa(&noise, w, h, 14);
            // A scene: two colours of a row that drift on their own, and a
            // highlight now and then.
            let mut scene = Vec::with_capacity(w * h);
            let mut v = [900i32, 2400];
            for i in 0..w * h {
                let c = i % 2;
                let step = (rng.next_u32() % 61) as i32 - 30;
                v[c] = if rng.next_u32() % 211 == 0 { 15408 } else { (v[c] + step).clamp(0, 16383) };
                scene.push(v[c] as u16);
            }
            let enc = roundtrip_cfa(&scene, w, h, 14);
            if w * h >= 1024 {
                assert!(enc.len() < w * h * 2, "a scene is no larger packed than loose: {} bytes for {} samples", enc.len(), w * h);
            }
        }
        // The codes at the ends of the range and steps between them.
        for &v in &[0u16, 1, 64, 8191, 8192, 15408, 16383] {
            let enc = roundtrip_cfa(&vec![v; 32 * 8], 32, 8, 14);
            assert!(enc.len() < 32 * 8 / 8 + 96, "a flat tile is tiny: {} bytes", enc.len());
        }
        let stripes: Vec<u16> = (0..64 * 4).map(|i| if (i / 2) % 2 == 0 { 0 } else { 16383 }).collect();
        roundtrip_cfa(&stripes, 64, 4, 14);
        // Sixteen bits: residuals of every size, the largest one included.
        let wide: Vec<u16> = (0..128 * 6).map(|i| match i % 8 { 0 => 0, 1 => 65535, 2 => 32768, 3 => 1, 4 => 32768, 5 => 32769, _ => (rng.next_u32() & 0xFFFF) as u16 }).collect();
        roundtrip_cfa(&wide, 128, 6, 16);
    }

    #[test]
    fn a_mosaic_stream_is_two_components_half_as_wide() {
        let tile: Vec<u16> = (0..3840 * 4).map(|i| ((i * 37) % 9001) as u16).collect();
        let mut enc = Vec::new();
        encode_tile_cfa_into(&tile, 3840, 4, 14, &mut enc);
        // SOI, SOF3 with P = 14, Y = 4, X = 1920, Nf = 2, ids 0 / 1 with H = V = 1.
        assert_eq!(&enc[..4], &[0xFF, 0xD8, 0xFF, 0xC3]);
        assert_eq!(&enc[4..18], &[0x00, 0x0E, 14, 0, 4, 0x07, 0x80, 2, 0, 0x11, 0, 1, 0x11, 0]);
        let mut pos = 18;
        for t in 0..2u8 {
            assert_eq!(&enc[pos..pos + 2], &[0xFF, 0xC4]);
            let len = ((enc[pos + 2] as usize) << 8) | enc[pos + 3] as usize;
            assert_eq!(enc[pos + 4], t);
            pos += 2 + len;
        }
        assert_eq!(&enc[pos..pos + 12], &[0xFF, 0xDA, 0x00, 0x0A, 2, 0, 0x00, 1, 0x10, 1, 0, 0]);
        assert_eq!(&enc[enc.len() - 2..], &[0xFF, 0xD9]);
        // No marker inside the entropy data: every FF there is stuffed.
        let data = &enc[pos + 12..enc.len() - 2];
        assert!(data.windows(2).all(|p| p[0] != 0xFF || p[1] == 0x00), "an unstuffed FF in the scan");
    }

    #[test]
    fn scratch_variant_matches_simple_variant() {
        let img = synth_image(48, 20);
        let a = encode_tile_rgb16(&img, 48, 20);
        let mut diffs = Vec::new();
        let mut out = vec![1, 2, 3]; // must be cleared
        encode_tile_rgb16_into(&img, 48, 20, &mut diffs, &mut out);
        assert_eq!(a, out);
        encode_tile_rgb16_into(&img, 48, 20, &mut diffs, &mut out);
        assert_eq!(a, out);
    }
}
