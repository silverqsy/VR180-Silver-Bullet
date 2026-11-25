//! The exact coding of an archive copy's pictures: a frame's samples, every
//! one as it was, in fewer bits than the camera has the time to find, and
//! read back at close to the pace of the camera's own records.
//!
//! The camera packs a block of samples at the width of its widest one. Here
//! each sample is predicted from the ones beside and above it, and what is
//! left is coded with the odds it has where the picture is that restless:
//!
//! 1. **Numbers.** A sample is not coded as its value but as the place of
//!    that value among the values a sample of the take can have
//!    ([`Numbers`]): a take whose values are the levels of a table has
//!    neighbouring levels one apart, whatever the levels are.
//! 2. **Planes.** The four colour planes of the mosaic are coded apart, each
//!    a picture half the frame's width and height.
//! 3. **Prediction.** The mean of the sample to the left and of the middle
//!    between the one above and the one above to the right. A raw frame is
//!    noise first and picture second, and a mean carries less of its
//!    neighbours' noise into the prediction than picking one of them does.
//!    (Only one addition and one halving lie between a sample and the next:
//!    that chain is what a row is read at the pace of.)
//! 4. **Classes.** How far the samples of the two rows above are apart
//!    (seven differences around the sample) puts it in one of 26 classes; a
//!    class has its own table of how often each difference occurs, counted
//!    over the plane and stored with it. A difference is one of 29 tokens:
//!    the small ones each their own, the larger ones by magnitude with
//!    their lower bits written plain. Nothing of a class comes from the row
//!    being read, so a whole row's tokens are read before its samples are
//!    put together, eight coders side by side.
//! 5. **Coder.** A range coder of the asymmetric numeral kind, which reads
//!    back a token with a multiplication and two table look-ups. Eight of
//!    them take a row's tokens in turn, each with a stream of its own, so
//!    that none waits for another; the plain bits are one stream beside
//!    them. A plane is cut into stripes of 128 rows, each coded on its own,
//!    so that a frame is read on every core.
//!
//! A mosaic is written by [`encode_mosaic`] and read back by
//! [`decode_mosaics`].

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::{Error, Result};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Ffmpeg(format!("archive copy: {msg}"))
}

/// Colour planes of a mosaic.
pub const PLANES: usize = 4;
/// Classes by how far a sample's neighbours above are apart.
const ACTIVITIES: usize = 26;
/// The class of a stripe's first row, which has no row above it.
const FIRST_ROW: usize = ACTIVITIES;
const CLASSES: usize = ACTIVITIES + 1;
/// Tokens a difference is one of: the direct ones, and one for every
/// number of plain bits up to sixteen (a difference of sixteen-bit numbers
/// has a zigzag number of seventeen bits).
const TOKENS: usize = 29;
/// Differences that are a token each (their zigzag numbers below this).
const DIRECT: u32 = 12;
/// log2 of the total of a class's shares.
const SCALE: u32 = 12;
const TOTAL: u32 = 1 << SCALE;
/// A coder's state stays at or above this.
const LOW: u32 = 1 << 16;
/// Rows of a stripe (of a plane).
pub const STRIPE_ROWS: usize = 128;
/// Coders a stripe's tokens are dealt out to, by column.
const WAYS: usize = 8;
/// Samples a row is carried on past either end, for the neighbours of its
/// first and last samples.
const PAD: usize = 2;
/// The largest sum of differences a class is told apart by.
const ACTIVITY_MOST: usize = 4095;
/// Bytes of nothing after a stripe's plain bits.
const PLAIN_END: usize = 4;
/// Rows of a stripe one plane is read for before the next plane's turn: the
/// planes of a mosaic row lie in the same rows of memory.
const TURN_ROWS: usize = 32;

/// The class of a sum of differences (up to [`ACTIVITY_MOST`]): the sum
/// itself while it is small, two classes an octave from there.
#[inline(always)]
fn class_of(apart: u16) -> u8 {
    let e = 15 - (apart | 8).leading_zeros() as u16;
    let far = 2 * e + 2 + ((apart >> (e - 1)) & 1);
    (if apart < 8 { apart } else { far }) as u8
}

/// For each token: the plain bits that follow it, and above those (8 bits
/// up) the zigzag number they are added to. (The table is longer than the
/// tokens are, so that a token read from a damaged picture finds an entry.)
static TOKEN: [u32; 32] = {
    let mut table = [0u32; 32];
    let mut t = 0usize;
    while t < TOKENS {
        table[t] = if t < DIRECT as usize { (t as u32) << 8 } else { (((1 << (t - DIRECT as usize)) + DIRECT - 1) << 8) | (t as u32 - DIRECT) };
        t += 1;
    }
    table
};

/// Twice what a direct token leaves of a sample (the token is the zigzag
/// number of it).
static DIRECT_TWICE: [i8; 16] = {
    let mut table = [0i8; 16];
    let mut t = 0usize;
    while t < DIRECT as usize {
        let left = ((t >> 1) as i8) ^ -((t & 1) as i8);
        table[t] = 2 * left;
        t += 1;
    }
    table
};

/// The values a sample of a take can have, as runs of evenly spaced ones,
/// and with them the number of each value: its place among them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Numbers {
    /// `(first value, step, count)` of each run, rising.
    runs: Vec<(u16, u16, u32)>,
}

impl Numbers {
    /// The numbers of `values` (rising, each once).
    pub fn of(values: &[u16]) -> Self {
        let mut runs: Vec<(u16, u16, u32)> = Vec::new();
        let mut i = 0usize;
        while i < values.len() {
            // A run goes on while the step stays what its first step was.
            let step = values.get(i + 1).map_or(1, |next| next - values[i]);
            let mut n = 1usize;
            while i + n < values.len() && values[i + n] - values[i + n - 1] == step {
                n += 1;
            }
            runs.push((values[i], step.max(1), n as u32));
            i += n;
        }
        Self { runs }
    }

    /// Every value from 0 to `top`: a number is the value.
    pub fn plain(top: u16) -> Self {
        Self { runs: vec![(0, 1, top as u32 + 1)] }
    }

    pub fn runs(&self) -> &[(u16, u16, u32)] {
        &self.runs
    }

    /// From runs read out of a record: checked to be rising values that fit sixteen bits.
    pub fn from_runs(runs: Vec<(u16, u16, u32)>) -> Result<Self> {
        let mut next = 0u32;
        for &(first, step, count) in &runs {
            if count == 0 || step == 0 || (first as u32) < next {
                return Err(err("a record's list of values does not rise"));
            }
            let last = first as u64 + (count as u64 - 1) * step as u64;
            if last > u16::MAX as u64 {
                return Err(err("a record's list of values goes past sixteen bits"));
            }
            next = last as u32 + 1;
        }
        if runs.is_empty() {
            return Err(err("a record has no list of values"));
        }
        Ok(Self { runs })
    }

    /// How many values there are.
    pub fn len(&self) -> usize {
        self.runs.iter().map(|r| r.2 as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// A number is the value it stands for.
    pub fn is_plain(&self) -> bool {
        matches!(self.runs[..], [(0, 1, _)])
    }

    /// The value of every number.
    pub fn values(&self) -> Vec<u16> {
        let mut value = Vec::with_capacity(self.len());
        for &(first, step, count) in &self.runs {
            value.extend((0..count as usize).map(|k| (first as usize + k * step as usize) as u16));
        }
        value
    }

    /// The number of every value from 0 to the largest; `u32::MAX` for a
    /// value a sample cannot have.
    pub fn number_of(&self) -> Vec<u32> {
        let values = self.values();
        let mut number = vec![u32::MAX; values.last().map_or(0, |v| *v as usize + 1)];
        for (k, v) in values.iter().enumerate() {
            number[*v as usize] = k as u32;
        }
        number
    }
}

/// The token of a difference's zigzag number, how many plain bits follow
/// it, and those bits.
#[inline]
fn token_of(z: u32) -> (u8, u32, u32) {
    if z < DIRECT {
        (z as u8, 0, 0)
    } else {
        let v = z - DIRECT + 1;
        let bits = 31 - v.leading_zeros();
        ((DIRECT + bits) as u8, bits, v - (1 << bits))
    }
}

/// A row's own edge samples carried on past its ends.
#[inline]
fn pad(row: &mut [u16]) {
    let n = row.len();
    (row[0], row[1]) = (row[PAD], row[PAD]);
    (row[n - 2], row[n - 1]) = (row[n - 1 - PAD], row[n - 1 - PAD]);
}

/// What the rows above say of a row before any of it is known: the class of
/// each of its samples, and their part of each sample's prediction.
struct Foreseen {
    class: Vec<u8>,
    /// The middle between the sample above and the one above to the right,
    /// rounded up.
    mid: Vec<u16>,
    across: Vec<u16>,
    down: Vec<u16>,
    apart: Vec<u16>,
}

impl Foreseen {
    fn new(width: usize) -> Self {
        let padded = width + 2 * PAD;
        Self { class: vec![0; width], mid: vec![0; width], across: vec![0; padded], down: vec![0; padded], apart: vec![0; width] }
    }

    /// A stripe's first row: one class, and the sample to the left alone.
    fn first_row(&mut self) {
        self.class.fill(FIRST_ROW as u8);
    }

    /// The row under `up1`, which lies under `up2` (both carried on past
    /// their ends). The seven differences: the four between neighbours in
    /// the row above from two to the left to two to the right, and the
    /// three between the rows above, to the left, above and to the right.
    fn row_under(&mut self, up1: &[u16], up2: &[u16]) {
        let width = self.class.len();
        let padded = width + 2 * PAD;
        let (up1, up2) = (&up1[..padded], &up2[..padded]);
        for ((d, a), b) in self.across[1..padded].iter_mut().zip(&up1[1..]).zip(&up1[..padded - 1]) {
            *d = a.abs_diff(*b);
        }
        for ((d, a), b) in self.down[..padded].iter_mut().zip(up1).zip(up2) {
            *d = a.abs_diff(*b);
        }
        let (h, v) = (&self.across[..padded], &self.down[..padded]);
        let (h1, h2, h3, h4) = (&h[1..1 + width], &h[2..2 + width], &h[3..3 + width], &h[4..4 + width]);
        let (v1, v2, v3) = (&v[1..1 + width], &v[2..2 + width], &v[3..3 + width]);
        let apart = &mut self.apart[..width];
        for x in 0..width {
            apart[x] = h1[x].saturating_add(h2[x]).saturating_add(h3[x]).saturating_add(h4[x])
                .saturating_add(v1[x]).saturating_add(v2[x]).saturating_add(v3[x])
                .min(ACTIVITY_MOST as u16);
        }
        for (class, a) in self.class.iter_mut().zip(apart.iter()) {
            *class = class_of(*a);
        }
        let (n, ne) = (&up1[PAD..PAD + width], &up1[PAD + 1..PAD + 1 + width]);
        for ((mid, n), ne) in self.mid.iter_mut().zip(n).zip(ne) {
            *mid = ((*n as u32 + *ne as u32 + 1) >> 1) as u16;
        }
    }
}

/// A class's counts as shares of [`TOTAL`]: every token that occurs has at
/// least one, and they add up to the total.
fn shares(counts: &[u32; TOKENS]) -> [u16; TOKENS] {
    let all: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut out = [0u16; TOKENS];
    if all == 0 {
        return out;
    }
    let mut given = 0i64;
    for (share, &count) in out.iter_mut().zip(counts) {
        if count > 0 {
            *share = ((count as u64 * TOTAL as u64 + all / 2) / all).clamp(1, TOTAL as u64) as u16;
            given += *share as i64;
        }
    }
    // What rounding left over or took too much is settled on the largest shares, one at a time.
    let mut order: Vec<usize> = (0..TOKENS).filter(|&t| out[t] > 0).collect();
    order.sort_by_key(|&t| std::cmp::Reverse(out[t]));
    let mut k = 0usize;
    while given != TOTAL as i64 {
        let t = order[k % order.len()];
        if given > TOTAL as i64 {
            if out[t] > 1 {
                out[t] -= 1;
                given -= 1;
            }
        } else {
            out[t] += 1;
            given += 1;
        }
        k += 1;
    }
    out
}

/// Run `job` for every number below `jobs` on `threads` threads (0: one a
/// core); what they return, in order.
fn in_parallel<T: Send>(jobs: usize, threads: usize, job: impl Fn(usize) -> Result<T> + Sync) -> Result<Vec<T>> {
    let workers = workers_for(threads, jobs);
    if workers <= 1 {
        return (0..jobs).map(job).collect();
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let done: Mutex<Vec<Option<T>>> = Mutex::new((0..jobs).map(|_| None).collect());
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs || stop.load(Ordering::Relaxed) {
                    break;
                }
                match job(i) {
                    Ok(result) => {
                        if let Ok(mut done) = done.lock() { done[i] = Some(result); }
                    }
                    Err(e) => {
                        stop.store(true, Ordering::Relaxed);
                        if let Ok(mut first) = failed.lock() { first.get_or_insert(e); }
                        break;
                    }
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().ok().flatten() {
        return Err(e);
    }
    done.into_inner().map_err(|_| err("a worker stopped"))?.into_iter().map(|r| r.ok_or_else(|| err("a worker stopped"))).collect()
}

fn workers_for(threads: usize, jobs: usize) -> usize {
    match threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        n => n,
    }.clamp(1, jobs.max(1))
}

// ── Writing ────────────────────────────────────────────────────────────

/// A stripe's samples as tokens, before the odds they are coded with are
/// known.
struct Tokens {
    class: Vec<u8>,
    token: Vec<u8>,
    /// The plain bits, the first sample's first, lowest bit first.
    plain: Vec<u8>,
    counts: Vec<[u32; TOKENS]>,
}

/// The tokens of rows `rows` of colour plane `plane` of a mosaic `width`
/// wide; `number` is the number of every value.
fn tokens_of(mosaic: &[u16], width: usize, plane: usize, rows: Range<usize>, number: &[u32]) -> Result<Tokens> {
    let (py, px) = (plane / 2, plane % 2);
    let pw = width / 2;
    let samples = rows.len() * pw;
    let mut out = Tokens { class: Vec::with_capacity(samples), token: Vec::with_capacity(samples), plain: Vec::new(), counts: vec![[0; TOKENS]; CLASSES] };
    let mut kept: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0u16; pw + 2 * PAD]);
    let mut seen = Foreseen::new(pw);
    let (mut hold, mut held) = (0u64, 0u32);
    let mut unknown = 0u32;
    for (k, r) in rows.enumerate() {
        kept.rotate_right(1);
        let (row, above) = kept.split_at_mut(1);
        let row = &mut row[0];
        let line = &mosaic[(2 * r + py) * width..(2 * r + py + 1) * width];
        for (own, v) in row[PAD..PAD + pw].iter_mut().zip(line[px..].iter().step_by(2)) {
            let n = number.get(*v as usize).copied().unwrap_or(u32::MAX);
            unknown |= n;
            *own = n as u16;
        }
        pad(row);
        let first = k == 0;
        if first {
            seen.first_row();
        } else {
            seen.row_under(&above[0], if k == 1 { &above[0] } else { &above[1] });
        }
        let mut w = if first { 0 } else { above[0][PAD] as i32 };
        for x in 0..pw {
            let v = row[PAD + x] as i32;
            let predicted = if first { w } else { (w + seen.mid[x] as i32) >> 1 };
            let d = v - predicted;
            let (token, bits, extra) = token_of(((d << 1) ^ (d >> 31)) as u32);
            let class = seen.class[x];
            out.class.push(class);
            out.token.push(token);
            out.counts[class as usize][token as usize] += 1;
            hold |= (extra as u64) << held;
            held += bits;
            if held >= 32 {
                out.plain.extend_from_slice(&(hold as u32).to_le_bytes());
                hold >>= 32;
                held -= 32;
            }
            w = v;
        }
    }
    while held > 0 {
        out.plain.push(hold as u8);
        hold >>= 8;
        held = held.saturating_sub(8);
    }
    // The reader looks at four bytes at a time: four more, so that its last look is inside the stripe.
    out.plain.extend_from_slice(&[0; PLAIN_END]);
    if unknown > u16::MAX as u32 {
        return Err(err("a sample has a value that is not among the take's"));
    }
    Ok(out)
}

/// A stripe's tokens through the coders (`width` tokens a row).
///
/// A stripe: the length of each coder's stream (`u32` each), the streams
/// (a coder's last state as `u32`, then its 16-bit words in the order they
/// are read), then the plain bits and four bytes of nothing.
fn code_stripe(tokens: &Tokens, width: usize, share: &[[u16; TOKENS]], start: &[[u16; TOKENS]]) -> Vec<u8> {
    let mut words: [Vec<u16>; WAYS] = std::array::from_fn(|_| Vec::with_capacity(tokens.token.len() / (6 * WAYS)));
    let mut state = [LOW; WAYS];
    // The coders work backwards, so that the reader goes forwards.
    for r in (0..tokens.token.len() / width).rev() {
        let (class, token) = (&tokens.class[r * width..(r + 1) * width], &tokens.token[r * width..(r + 1) * width]);
        for x in (0..width).rev() {
            let way = x % WAYS;
            let (f, b) = (share[class[x] as usize][token[x] as usize] as u32, start[class[x] as usize][token[x] as usize] as u32);
            let mut s = state[way];
            if s as u64 >= ((LOW >> SCALE) as u64 * f as u64) << 16 {
                words[way].push(s as u16);
                s >>= 16;
            }
            state[way] = ((s / f) << SCALE) + (s % f) + b;
        }
    }
    let coded: usize = words.iter().map(|w| 4 + 2 * w.len()).sum();
    let mut out = Vec::with_capacity(4 * WAYS + coded + tokens.plain.len());
    for w in &words {
        out.extend_from_slice(&(4 + 2 * w.len() as u32).to_le_bytes());
    }
    for (w, s) in words.iter().zip(state) {
        out.extend_from_slice(&s.to_le_bytes());
        for word in w.iter().rev() {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    out.extend_from_slice(&tokens.plain);
    out
}

/// A plane's picture from its shares and its coded stripes.
///
/// The picture: the bytes 2, classes, tokens, scale, coders and three of 0,
/// the stripes as `u32`; for every class the tokens that occur in it (a bit
/// each of a `u32`) and the share of each of those (`u16`); the length of
/// every stripe (`u32` each); the stripes.
fn picture_of(share: &[[u16; TOKENS]], stripes: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![2u8, CLASSES as u8, TOKENS as u8, SCALE as u8, WAYS as u8, 0, 0, 0];
    out.extend_from_slice(&(stripes.len() as u32).to_le_bytes());
    for class in share {
        let occur = class.iter().enumerate().fold(0u32, |m, (t, s)| m | ((*s > 0) as u32) << t);
        out.extend_from_slice(&occur.to_le_bytes());
        for s in class.iter().filter(|s| **s > 0) {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    for stripe in stripes {
        out.extend_from_slice(&(stripe.len() as u32).to_le_bytes());
    }
    out.reserve(stripes.iter().map(Vec::len).sum());
    for stripe in stripes {
        out.extend_from_slice(stripe);
    }
    out
}

/// The four colour planes of a mosaic (`width × height` samples, both even,
/// row after row) as a picture each, on `threads` threads (0: one a core).
/// An error where a sample has a value that is none of `numbers`'.
pub fn encode_mosaic(mosaic: &[u16], width: usize, height: usize, numbers: &Numbers, threads: usize) -> Result<Vec<Vec<u8>>> {
    if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 || mosaic.len() != width * height {
        return Err(err(format!("a mosaic of {width} x {height} in {} samples is not coded", mosaic.len())));
    }
    let (pw, ph) = (width / 2, height / 2);
    let stripes = ph.div_ceil(STRIPE_ROWS);
    let number = numbers.number_of();
    let tokens = in_parallel(PLANES * stripes, threads, |k| {
        let s = k % stripes;
        tokens_of(mosaic, width, k / stripes, s * STRIPE_ROWS..((s + 1) * STRIPE_ROWS).min(ph), &number)
    })?;
    // A plane's odds: counted over all of it.
    let tables: Vec<(Vec<[u16; TOKENS]>, Vec<[u16; TOKENS]>)> = tokens.chunks(stripes).map(|plane| {
        let mut counts = vec![[0u32; TOKENS]; CLASSES];
        for stripe in plane {
            for (sum, c) in counts.iter_mut().zip(&stripe.counts) {
                for (a, b) in sum.iter_mut().zip(c) {
                    *a += b;
                }
            }
        }
        let share: Vec<[u16; TOKENS]> = counts.iter().map(shares).collect();
        let start = share.iter().map(|s| {
            let mut at = 0u16;
            std::array::from_fn(|t| { let b = at; at += s[t]; b })
        }).collect();
        (share, start)
    }).collect();
    let coded = in_parallel(PLANES * stripes, threads, |k| {
        let (share, start) = &tables[k / stripes];
        Ok(code_stripe(&tokens[k], pw, share, start))
    })?;
    Ok(coded.chunks(stripes).zip(&tables).map(|(plane, (share, _))| picture_of(share, plane)).collect())
}

// ── Reading ────────────────────────────────────────────────────────────

/// What a picture's head says, and where its stripes lie.
struct Plan<'a> {
    /// For every class, for every place in the total: the token there.
    /// (Room for 32 classes and 32 tokens, so that neither is checked for
    /// where it is looked up.)
    slot: Box<[[u8; TOTAL as usize]; 32]>,
    /// For every class and token: its share, and above that (16 bits up)
    /// the place its share starts at.
    share: Box<[[u32; 32]; 32]>,
    stripes: Vec<&'a [u8]>,
}

fn plan(picture: &[u8], height: usize) -> Result<Plan<'_>> {
    let short = || err("a picture ends early");
    let word = |at: usize| picture.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(short);
    if picture.len() < 12 || picture[..5] != [2, CLASSES as u8, TOKENS as u8, SCALE as u8, WAYS as u8] {
        return Err(err("a picture is not one this build reads"));
    }
    let stripes = word(8)? as usize;
    if stripes != height.div_ceil(STRIPE_ROWS) {
        return Err(err("a picture is not its frame's height"));
    }
    let table = || err("no room for a picture's tables");
    let mut slot: Box<[[u8; TOTAL as usize]; 32]> = vec![[0u8; TOTAL as usize]; 32].into_boxed_slice().try_into().map_err(|_| table())?;
    let mut share: Box<[[u32; 32]; 32]> = vec![[TOTAL; 32]; 32].into_boxed_slice().try_into().map_err(|_| table())?;
    let mut at = 12usize;
    for class in 0..CLASSES {
        let occur = word(at)?;
        at += 4;
        if occur >> TOKENS != 0 {
            return Err(err("a picture names a token there is none of"));
        }
        if occur == 0 {
            // A class no sample of the plane is in: any state reads as its first token.
            continue;
        }
        let mut place = 0usize;
        for token in (0..TOKENS).filter(|t| occur >> t & 1 == 1) {
            let s = picture.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or_else(short)?;
            at += 2;
            if s == 0 || place + s > TOTAL as usize {
                return Err(err("a picture's shares do not add up to the total"));
            }
            slot[class][place..place + s].fill(token as u8);
            share[class][token] = s as u32 | (place as u32) << 16;
            place += s;
        }
        if place != TOTAL as usize {
            return Err(err("a picture's shares do not add up to the total"));
        }
    }
    let lengths = at;
    at += 4 * stripes;
    let mut parts = Vec::with_capacity(stripes);
    for s in 0..stripes {
        let length = word(lengths + 4 * s)? as usize;
        let part = at.checked_add(length).and_then(|end| picture.get(at..end)).ok_or_else(short)?;
        parts.push(part);
        at += length;
    }
    if at != picture.len() {
        return Err(err("a picture has bytes left over"));
    }
    Ok(Plan { slot, share, stripes: parts })
}

/// The reader of one stripe of one plane, a row at a time.
struct Reader<'a> {
    stripe: &'a [u8],
    /// Where each coder's next word lies, where its stream ends, its state.
    at: [u32; WAYS],
    end: [u32; WAYS],
    state: [u32; WAYS],
    plain: &'a [u8],
    /// Plain bits read.
    bit: usize,
    /// The row being read and the two above it, carried on past their ends.
    kept: [Vec<u16>; 3],
    read: usize,
    seen: Foreseen,
    token: Vec<u8>,
}

/// The four bytes of `bytes` from `at`, with nothing for those past its end.
#[cold]
fn last_bytes(bytes: &[u8], at: usize) -> u32 {
    let mut four = [0u8; 4];
    for (k, b) in four.iter_mut().enumerate() {
        *b = bytes.get(at + k).copied().unwrap_or(0);
    }
    u32::from_le_bytes(four)
}

/// A row's tokens (one for each of `class`): each coder reads every eighth
/// one, from its own stream of `stripe`.
fn read_tokens(plan: &Plan, stripe: &[u8], states: &mut [u32; WAYS], ats: &mut [u32; WAYS], class: &[u8], token: &mut [u8]) {
    let (slot, share) = (&*plan.slot, &*plan.share);
    let (mut state, mut at) = (*states, *ats);
    let mut one = |way: usize, class: u8, token: &mut u8| {
        let s = state[way];
        let place = s & (TOTAL - 1);
        let class = (class & 31) as usize;
        let t = slot[class][place as usize];
        let e = share[class][(t & 31) as usize];
        *token = t;
        let s = (e & 0xffff).wrapping_mul(s >> SCALE).wrapping_add(place).wrapping_sub(e >> 16);
        // The coder's next word is taken up where its state has run low.
        let next = match stripe.get(at[way] as usize..at[way] as usize + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]) as u32,
            None => 0,
        };
        let low = s < LOW;
        state[way] = if low { (s << 16) | next } else { s };
        at[way] = at[way].wrapping_add(2 * low as u32);
    };
    let mut classes = class.chunks_exact(WAYS);
    let mut tokens = token.chunks_exact_mut(WAYS);
    for (class, token) in (&mut classes).zip(&mut tokens) {
        for way in 0..WAYS {
            one(way, class[way], &mut token[way]);
        }
    }
    for (way, (class, token)) in classes.remainder().iter().zip(tokens.into_remainder()).enumerate() {
        one(way, *class, token);
    }
    (*states, *ats) = (state, at);
}

impl<'a> Reader<'a> {
    fn new(stripe: &'a [u8], width: usize) -> Result<Self> {
        let short = || err("a stripe ends early");
        let head = 4 * WAYS;
        if stripe.len() < head {
            return Err(short());
        }
        if stripe.len() > u32::MAX as usize - 8 {
            return Err(err("a stripe is longer than one is written"));
        }
        let (mut at, mut end, mut state) = ([0u32; WAYS], [0u32; WAYS], [0u32; WAYS]);
        let mut from = head;
        for way in 0..WAYS {
            let length = u32::from_le_bytes([stripe[4 * way], stripe[4 * way + 1], stripe[4 * way + 2], stripe[4 * way + 3]]) as usize;
            let stream = from.checked_add(length).and_then(|to| stripe.get(from..to)).ok_or_else(short)?;
            if length < 4 || length % 2 != 0 {
                return Err(short());
            }
            state[way] = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]);
            (at[way], end[way]) = ((from + 4) as u32, (from + length) as u32);
            from += length;
        }
        Ok(Self {
            stripe, at, end, state, plain: &stripe[from..], bit: 0,
            kept: std::array::from_fn(|_| vec![0u16; width + 2 * PAD]), read: 0,
            seen: Foreseen::new(width), token: vec![0; width],
        })
    }

    /// Read the next row: its samples (numbers up to `top`) through `value`
    /// into every other sample of `line` from `px`.
    #[inline]
    fn row(&mut self, plan: &Plan, top: u32, line: &mut [u16], px: usize, value: impl Fn(u16) -> u16) -> Result<()> {
        self.kept.rotate_right(1);
        let (row, above) = self.kept.split_at_mut(1);
        let row = &mut row[0];
        let width = self.token.len();
        let first = self.read == 0;
        if first {
            self.seen.first_row();
        } else {
            self.seen.row_under(&above[0], if self.read == 1 { &above[0] } else { &above[1] });
        }

        // The row's tokens: each coder reads every eighth one.
        read_tokens(plan, self.stripe, &mut self.state, &mut self.at, &self.seen.class, &mut self.token);

        // The row's samples: what each token says is added to what its neighbours foretell.
        let plain = self.plain;
        let mut bit = self.bit;
        // What is left of a sample after its prediction, from its token and the plain bits behind it.
        let mut left = |token: u8| -> i32 {
            let e = TOKEN[(token & 31) as usize];
            let bits = e & 31;
            let byte = bit >> 3;
            let held = match plain.get(byte..byte + 4) {
                Some(b) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                None => last_bytes(plain, byte),
            };
            let z = (e >> 8) + ((held >> (bit & 7)) & !(u32::MAX << bits));
            bit += bits as usize;
            ((z >> 1) as i32) ^ -((z & 1) as i32)
        };
        let own = &mut row[PAD..PAD + width];
        let px = px & 1;
        // Every sample or-ed together: a number below nothing or past sixteen bits shows above them.
        let mut all = 0u32;
        if first {
            let mut w = 0i32;
            for ((token, own), out) in self.token.iter().zip(own.iter_mut()).zip(line.chunks_exact_mut(2)) {
                let v = w.wrapping_add(left(*token));
                all |= v as u32;
                *own = v as u16;
                out[px] = value(v as u16);
                w = v;
            }
        } else {
            // Twice what is left, added before the halving, is what is left added after it: all of a
            // sample that does not come from the one to its left is ready before that one is.
            let mut w = above[0][PAD] as i32;
            let mut tokens = self.token.chunks_exact(WAYS);
            let mut mids = self.seen.mid.chunks_exact(WAYS);
            let mut owns = own.chunks_exact_mut(WAYS);
            let mut outs = line.chunks_exact_mut(2 * WAYS);
            for (((token, mid), own), out) in (&mut tokens).zip(&mut mids).zip(&mut owns).zip(&mut outs) {
                if token.iter().all(|t| (*t as u32) < DIRECT) {
                    // None of the eight has plain bits: a token is what is left.
                    for k in 0..WAYS {
                        let v = w.wrapping_add(mid[k] as i32 + DIRECT_TWICE[(token[k] & 15) as usize] as i32) >> 1;
                        all |= v as u32;
                        own[k] = v as u16;
                        out[2 * k + px] = value(v as u16);
                        w = v;
                    }
                } else {
                    for k in 0..WAYS {
                        let v = w.wrapping_add(mid[k] as i32 + (left(token[k]) << 1)) >> 1;
                        all |= v as u32;
                        own[k] = v as u16;
                        out[2 * k + px] = value(v as u16);
                        w = v;
                    }
                }
            }
            let rest = tokens.remainder().iter().zip(mids.remainder()).zip(owns.into_remainder()).zip(outs.into_remainder().chunks_exact_mut(2));
            for (((token, mid), own), out) in rest {
                let v = w.wrapping_add(*mid as i32 + (left(*token) << 1)) >> 1;
                all |= v as u32;
                *own = v as u16;
                out[px] = value(v as u16);
                w = v;
            }
        }
        self.bit = bit;
        if all > u16::MAX as u32 || own.iter().copied().max().unwrap_or(0) as u32 > top {
            return Err(err("a stripe holds a number no sample has"));
        }
        pad(row);
        self.read += 1;
        Ok(())
    }

    /// Whether the stripe ended where its rows did.
    fn finish(&self) -> Result<()> {
        if self.at != self.end || self.state != [LOW; WAYS] || self.bit.div_ceil(8) + PLAIN_END != self.plain.len() {
            return Err(err("a stripe does not end where its rows do"));
        }
        Ok(())
    }
}

/// Stripe `s` of the two planes a mosaic's even rows are made of, or its
/// odd ones, into `rows`: those rows of the mosaic that the stripe holds.
fn read_stripe(plans: &[Plan], s: usize, top: u32, value: Option<&[u16; 1 << 16]>, rows: &mut [&mut [u16]]) -> Result<()> {
    let width = rows.first().map_or(0, |r| r.len());
    let mut readers: Vec<Reader> = plans.iter().map(|p| Reader::new(p.stripes[s], width / 2)).collect::<Result<_>>()?;
    let mut from = 0usize;
    while from < rows.len() {
        let to = (from + TURN_ROWS).min(rows.len());
        for (px, (reader, plan)) in readers.iter_mut().zip(plans).enumerate() {
            for line in rows[from..to].iter_mut() {
                match value {
                    Some(table) => reader.row(plan, top, line, px, |n| table[n as usize])?,
                    None => reader.row(plan, top, line, px, |n| n)?,
                }
            }
        }
        from = to;
    }
    readers.iter().try_for_each(Reader::finish)
}

/// Read `pictures`, four for each mosaic of `width × height`, back into
/// `out`: the mosaics one after the other, on `threads` threads (0: one a
/// core). `numbers` are the ones the pictures were written with.
pub fn decode_mosaics(pictures: &[&[u8]], width: usize, height: usize, numbers: &Numbers, out: &mut [u16], threads: usize) -> Result<()> {
    if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 || pictures.len() % PLANES != 0 || out.len() != pictures.len() / PLANES * width * height {
        return Err(err(format!("{} pictures of a mosaic of {width} x {height} do not fill {} samples", pictures.len(), out.len())));
    }
    if numbers.is_empty() {
        return Err(err("a record has no list of values"));
    }
    let plans: Vec<Plan> = pictures.iter().map(|p| plan(p, height / 2)).collect::<Result<_>>()?;
    let top = numbers.len() as u32 - 1;
    let value: Option<Box<[u16; 1 << 16]>> = if numbers.is_plain() {
        None
    } else {
        let mut table = numbers.values();
        table.resize(1 << 16, 0);
        table.into_boxed_slice().try_into().ok()
    };
    let value = value.as_deref();
    // A stripe of a mosaic is two jobs: its even rows, which two of the planes fill, and its odd ones.
    let mut jobs: Vec<(&[Plan], usize, Vec<&mut [u16]>)> = Vec::new();
    for (mosaic, plans) in out.chunks_mut(width * height).zip(plans.chunks(PLANES)) {
        for (s, rows) in mosaic.chunks_mut(2 * STRIPE_ROWS * width).enumerate() {
            let (mut even, mut odd) = (Vec::with_capacity(STRIPE_ROWS), Vec::with_capacity(STRIPE_ROWS));
            for (r, row) in rows.chunks_mut(width).enumerate() {
                if r % 2 == 0 { even.push(row) } else { odd.push(row) }
            }
            jobs.push((&plans[..PLANES / 2], s, even));
            jobs.push((&plans[PLANES / 2..], s, odd));
        }
    }
    let workers = workers_for(threads, jobs.len());
    if workers <= 1 {
        return jobs.into_iter().try_for_each(|(plans, s, mut rows)| read_stripe(plans, s, top, value, &mut rows));
    }
    let queue = Mutex::new(jobs.into_iter());
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let job = queue.lock().ok().and_then(|mut q| q.next());
                let Some((plans, s, mut rows)) = job else { break };
                if let Err(e) = read_stripe(plans, s, top, value, &mut rows) {
                    if let Ok(mut first) = failed.lock() { first.get_or_insert(e); }
                    break;
                }
            });
        }
    });
    match failed.into_inner().ok().flatten() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: u64) -> impl FnMut() -> u32 {
        let mut state = seed | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32
        }
    }

    /// Write a mosaic and read it back, alone on one thread and twice over
    /// on every core; the bytes it came to.
    fn there_and_back(mosaic: &[u16], width: usize, height: usize, numbers: &Numbers) -> usize {
        let pictures = encode_mosaic(mosaic, width, height, numbers, 0).unwrap();
        assert_eq!(pictures.len(), PLANES);
        assert_eq!(encode_mosaic(mosaic, width, height, numbers, 1).unwrap(), pictures, "one thread writes what many do");
        let refs: Vec<&[u8]> = pictures.iter().map(|p| &p[..]).collect();
        let mut one = vec![0xAAAAu16; mosaic.len()];
        decode_mosaics(&refs, width, height, numbers, &mut one, 1).unwrap();
        assert!(one == mosaic, "{width} x {height} does not come back");
        let twice: Vec<&[u8]> = refs.iter().chain(&refs).copied().collect();
        let mut two = vec![0x5555u16; 2 * mosaic.len()];
        decode_mosaics(&twice, width, height, numbers, &mut two, 0).unwrap();
        assert!(two[..mosaic.len()] == *mosaic && two[mosaic.len()..] == *mosaic);
        pictures.iter().map(Vec::len).sum()
    }

    /// Whatever a mosaic holds comes back as it was: nothing, noise of
    /// every size, a smooth picture with noise, the largest steps a sample
    /// can take; widths that are no whole number of coders, heights that
    /// are no whole stripes, the smallest mosaic there is.
    #[test]
    fn a_mosaic_comes_back_as_it_was() {
        let mut next = noise(0x9E3779B97F4A7C15);
        let all = Numbers::plain(u16::MAX);
        for (width, height) in [(2usize, 2usize), (4, 6), (14, 2), (16, 258), (130, 254), (666, 258), (1280, 800)] {
            let n = width * height;
            let head = PLANES * (12 + CLASSES * 6 + (4 + 4 * WAYS + 4 * WAYS) * (height / 2).div_ceil(STRIPE_ROWS));
            assert!(there_and_back(&vec![0u16; n], width, height, &all) <= head + n / 32, "{width} x {height}: nothing is not small");
            there_and_back(&vec![65535u16; n], width, height, &all);
            for bits in [1u32, 4, 10, 14, 16] {
                let mosaic: Vec<u16> = (0..n).map(|_| (next() & ((1 << bits) - 1)) as u16).collect();
                there_and_back(&mosaic, width, height, &all);
            }
            let smooth: Vec<u16> = (0..n).map(|i| (64 + (i % width) / 3 + (i / width) / 2 + (next() % 5) as usize) as u16).collect();
            let bytes = there_and_back(&smooth, width, height, &all);
            // A slope with noise of five values: under three bits a sample (a mosaic with heads to pay for needs some size).
            assert!(n < 100_000 || bytes * 8 < n * 3, "{bytes} bytes for {n} samples");
            let steps: Vec<u16> = (0..n).map(|i| if (i / 7) % 2 == 0 { 0 } else { 65535 }).collect();
            there_and_back(&steps, width, height, &all);
        }
    }

    /// A take whose values are the levels of a table: its samples are coded
    /// as their places in the table and come back as the levels; a sample
    /// with a value the table has not is refused.
    #[test]
    fn levels_of_a_table_are_coded_as_their_places() {
        let mut next = noise(99);
        let values: Vec<u16> = (0..85).chain((0..405).map(|i| 86 + 2 * i)).chain(896..1024).collect();
        let numbers = Numbers::of(&values);
        assert_eq!((numbers.values(), numbers.runs().len()), (values.clone(), 3));
        let (width, height) = (300usize, 268usize);
        let mosaic: Vec<u16> = (0..width * height).map(|i| {
            let level = (60 + (i % width) * 2 + (i / width) + (next() % 7) as usize).min(values.len() - 1);
            values[level]
        }).collect();
        let as_places = there_and_back(&mosaic, width, height, &numbers);
        // The same samples as plain values: the steps of two between the levels cost a bit a sample there.
        let as_values = there_and_back(&mosaic, width, height, &Numbers::plain(1023));
        assert!(as_places < as_values, "{as_places} against {as_values}");
        let mut odd = mosaic.clone();
        odd[width * 100 + 17] = 87;
        assert!(encode_mosaic(&odd, width, height, &numbers, 0).is_err());
        assert!(encode_mosaic(&odd, width, height, &Numbers::plain(1023), 0).is_ok());
        // Read with a shorter list of values, a number past its end is refused.
        let pictures = encode_mosaic(&mosaic, width, height, &numbers, 0).unwrap();
        let refs: Vec<&[u8]> = pictures.iter().map(|p| &p[..]).collect();
        let mut out = vec![0u16; mosaic.len()];
        assert!(decode_mosaics(&refs, width, height, &Numbers::of(&values[..200]), &mut out, 0).is_err());
    }

    /// A picture that was cut short, or changed, is refused or reads as
    /// something else; nothing is read past its end.
    #[test]
    fn a_damaged_picture_does_not_read_past_its_end() {
        let mut next = noise(7);
        let (width, height) = (200usize, 600usize);
        let numbers = Numbers::plain(4095);
        let mosaic: Vec<u16> = (0..width * height).map(|i| (500 + (i % width) + (next() % 90) as usize) as u16).collect();
        let pictures = encode_mosaic(&mosaic, width, height, &numbers, 0).unwrap();
        let refs: Vec<&[u8]> = pictures.iter().map(|p| &p[..]).collect();
        let mut out = vec![0u16; mosaic.len()];
        for cut in [0usize, 3, 11, 12, 60, 200, refs[2].len() / 2, refs[2].len() - 1] {
            let mut short = refs.clone();
            short[2] = &refs[2][..cut];
            assert!(decode_mosaics(&short, width, height, &numbers, &mut out, 0).is_err(), "cut at {cut}");
        }
        assert!(decode_mosaics(&refs, width, height + 2 * STRIPE_ROWS, &numbers, &mut vec![0u16; width * (height + 2 * STRIPE_ROWS)], 0).is_err());
        assert!(decode_mosaics(&refs[..3], width, height, &numbers, &mut out, 0).is_err());
        assert!(decode_mosaics(&refs, width, height, &numbers, &mut out[1..], 0).is_err());
        for k in 0..400 {
            let mut damaged = pictures[1].clone();
            let at = (next() as usize) % damaged.len();
            damaged[at] ^= 1 << (k % 8);
            let mut with = refs.clone();
            with[1] = &damaged;
            // Refused, or other samples: never a panic.
            let _ = decode_mosaics(&with, width, height, &numbers, &mut out, 1);
        }
    }

    /// The values a take can have as runs, and a value's number.
    #[test]
    fn values_have_numbers() {
        let plain = Numbers::plain(1023);
        assert_eq!((plain.runs(), plain.len(), plain.is_plain()), (&[(0u16, 1u16, 1024u32)][..], 1024, true));
        // A table of levels with every value from a threshold up.
        let values: Vec<u16> = (0..32).map(|i| i * 8).chain((0..24).map(|i| 256 + i * 32)).chain(13472..=13480).collect();
        let numbers = Numbers::of(&values);
        assert_eq!(numbers.len(), values.len());
        assert!(!numbers.is_plain());
        assert_eq!(numbers.values(), values);
        let number = numbers.number_of();
        for (k, v) in values.iter().enumerate() {
            assert_eq!(number[*v as usize], k as u32);
        }
        assert_eq!((number[1], number[9], number[257], number.len()), (u32::MAX, u32::MAX, u32::MAX, 13481));
        assert_eq!(Numbers::from_runs(numbers.runs().to_vec()).unwrap(), numbers);
        assert!(Numbers::from_runs(vec![(8, 8, 4), (16, 1, 2)]).is_err());
        assert!(Numbers::from_runs(vec![(65000, 100, 9)]).is_err());
        assert!(Numbers::from_runs(vec![(0, 1, u32::MAX)]).is_err());
        assert!(Numbers::from_runs(Vec::new()).is_err());
        assert_eq!(Numbers::plain(u16::MAX).len(), 65536);
    }

    #[test]
    fn shares_add_up_and_leave_no_token_out() {
        let mut counts = [0u32; TOKENS];
        assert_eq!(shares(&counts), [0u16; TOKENS]);
        counts[3] = 1;
        counts[5] = 1_000_000;
        counts[28] = 7;
        let s = shares(&counts);
        assert_eq!(s.iter().map(|&v| v as u32).sum::<u32>(), TOTAL);
        assert!(s[3] >= 1 && s[28] >= 1 && s[5] > 4000 && s[0] == 0);
        let even = [5u32; TOKENS];
        assert_eq!(shares(&even).iter().map(|&v| v as u32).sum::<u32>(), TOTAL);
    }

    /// The classes rise with how far the neighbours are apart, and the
    /// tokens' tables say what [`token_of`] does.
    #[test]
    fn classes_and_tokens_are_what_their_tables_say() {
        let classes: Vec<u8> = (0..=ACTIVITY_MOST as u16).map(class_of).collect();
        assert_eq!((classes[0], classes[7], classes[8], classes[11], classes[12], classes[16], classes[23], classes[24], classes[4095]), (0, 7, 8, 8, 9, 10, 10, 11, 25));
        assert!(classes.windows(2).all(|w| w[0] <= w[1] && w[1] - w[0] <= 1) && (classes[ACTIVITY_MOST] as usize) == ACTIVITIES - 1);
        for z in (0..5000u32).chain([65535, 131070, 131082]) {
            let (token, bits, extra) = token_of(z);
            let e = TOKEN[token as usize];
            assert!((token as usize) < TOKENS && bits == e & 31 && extra < (1 << bits));
            assert_eq!((e >> 8) + extra, z);
        }
    }
}
