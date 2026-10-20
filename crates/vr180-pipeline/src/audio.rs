//! Audio extraction + spatial-audio plumbing.
//!
//! Phase 0.8.6 adds the side of the pipeline that handles the GoPro
//! MAX 2's ambisonic 4-channel PCM track (stream index 5 in the `.360`
//! container; `pcm_s24le`, 48 kHz, AmbiX W,Y,Z,X channel layout).
//!
//! Pipeline:
//!   .360 → `extract_ambisonic_to_wav` (in-process libav remux)
//!        → temp.wav (4ch pcm_s24le)
//!        → `helpers::spawn_apac_encode` (Swift helper, Vision Pro APAC)
//!        → final.mov (video passthrough + APAC audio)
//!
//! In-process WAV remux avoids the Python app's subprocess hop to the
//! system ffmpeg binary; ffmpeg-next opens the `.360` once for video
//! decode and a second time here for audio extraction (the cost is
//! ~1ms of header parsing per 5+ GB file — negligible).

use crate::{Error, Result};
use ffmpeg_next as ffmpeg;
use std::path::Path;

/// Description of the source's ambisonic audio track.
#[derive(Debug, Clone, Copy)]
pub struct AmbisonicInfo {
    /// 0-based stream index in the source container.
    pub stream_index: usize,
    pub sample_rate: u32,
    pub channels: u32,
    /// AVCodecID enum value (e.g. `AV_CODEC_ID_PCM_S24LE = 65557`).
    /// Stored as `i32` to avoid leaking ffmpeg-next's `codec::Id` into
    /// public callers that don't depend on it.
    pub codec_id: i32,
}

/// Probe the source `.360` (or any container) for a 4-channel ambisonic
/// audio track. Returns `Ok(Some(_))` if one is found, `Ok(None)` if
/// the file has audio but none ambisonic, or `Err` on open failure.
///
/// GoPro MAX 2 puts the ambisonic track at stream index 5 with codec
/// `pcm_s24le`, 4 channels, "ambisonic 1" layout. We match on the
/// channel count + audio media type + raw PCM codec ID to be robust
/// against future firmware tweaks (e.g. if GoPro ever switches to
/// pcm_f32le, the match still hits as long as channels == 4 and it's
/// uncompressed PCM).
pub fn probe_ambisonic(input: &Path) -> Result<Option<AmbisonicInfo>> {
    crate::decode::init();
    let ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open {input:?}: {e}")))?;
    for stream in ictx.streams() {
        let params = stream.parameters();
        let medium = params.medium();
        if medium != ffmpeg::media::Type::Audio { continue; }
        // SAFETY: we hold the input context borrow via `stream`.
        let (channels, codec_id, sample_rate) = unsafe {
            let raw = params.as_ptr();
            (
                (*raw).ch_layout.nb_channels as u32,
                (*raw).codec_id as i32,
                (*raw).sample_rate as u32,
            )
        };
        if channels != 4 { continue; }
        // Accept any uncompressed PCM ambisonic — pcm_s24le is what
        // GoPro produces today; pcm_s16le / pcm_f32le would be valid
        // if the firmware ever switches.
        if !is_pcm_codec_id(codec_id) { continue; }
        return Ok(Some(AmbisonicInfo {
            stream_index: stream.index(),
            sample_rate,
            channels,
            codec_id,
        }));
    }
    Ok(None)
}

/// The Insta360 X6's "360° Audio" track: a 4-channel AAC with no channel
/// semantics in the file (AAC channel config 4, no `chan`/SA3D). Measured
/// against Insta360 Studio's SA3D-tagged export (2026-10-10): the decoded
/// channels ARE first-order AmbiX — `[W, Y, Z, X]` (ACN order, SN3D) — Studio
/// only rotates X/Y by its stabilisation yaw, scales, and tags SA3D.
/// (Older firmware / the "Auto" or "Stereo" audio modes write stereo AAC,
/// for which this returns `None`.)
pub fn probe_foa_aac(input: &Path) -> Result<Option<AmbisonicInfo>> {
    crate::decode::init();
    let ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open {input:?}: {e}")))?;
    for stream in ictx.streams() {
        let params = stream.parameters();
        if params.medium() != ffmpeg::media::Type::Audio { continue; }
        // SAFETY: we hold the input context borrow via `stream`.
        let (channels, codec_id, sample_rate) = unsafe {
            let raw = params.as_ptr();
            ((*raw).ch_layout.nb_channels as u32, (*raw).codec_id as i32, (*raw).sample_rate as u32)
        };
        if channels != 4 || codec_id != ffmpeg::ffi::AVCodecID::AV_CODEC_ID_AAC as i32 { continue; }
        return Ok(Some(AmbisonicInfo { stream_index: stream.index(), sample_rate, channels, codec_id }));
    }
    Ok(None)
}

/// Any 4-channel first-order ambisonic track: GoPro's PCM
/// (`probe_ambisonic`) or the X6's AAC (`probe_foa_aac`).
pub fn probe_spatial_4ch(input: &Path) -> Result<Option<AmbisonicInfo>> {
    match probe_ambisonic(input)? {
        Some(i) => Ok(Some(i)),
        None => probe_foa_aac(input),
    }
}

/// Front-facing stereo decode of first-order AmbiX: a pair of virtual
/// cardioids at ±45° — SN3D cardioid = ½·W + ½·(X·cos θ + Y·sin θ), +X the
/// camera's front (its front lens, the VR180 rig's forward), +Y left.
/// Rows `[L, R]`, columns `[W, Y, Z, X]` (decoded channel order).
/// A plain layout downmix of the X6 track instead cancels two anti-phase
/// channels and lands ~9 dB left-heavy (measured 2026-10-10).
pub const FOA_STEREO_MATRIX: [[f32; 4]; 2] = [
    [0.5,  0.353_553_4, 0.0, 0.353_553_4],
    [0.5, -0.353_553_4, 0.0, 0.353_553_4],
];

/// One sample of `[W, Y, Z, X]` → `[L, R]` through [`FOA_STEREO_MATRIX`].
#[inline]
pub fn foa_to_stereo_sample(wyzx: [f32; 4]) -> [f32; 2] {
    let m = &FOA_STEREO_MATRIX;
    [
        m[0][0] * wyzx[0] + m[0][1] * wyzx[1] + m[0][2] * wyzx[2] + m[0][3] * wyzx[3],
        m[1][0] * wyzx[0] + m[1][1] * wyzx[1] + m[1][2] * wyzx[2] + m[1][3] * wyzx[3],
    ]
}

/// Kandao QooCam 3 Ultra "4-channel" audio: the four microphone capsules,
/// recorded as a plain 4-channel AAC (no sound-field encoding; the
/// container carries no layout). QooCam Studio's "Audio SuperSteady 360°"
/// is Fraunhofer upHear — a parametric spatial process (direction per
/// time-frequency tile), which is why Studio's output is not a linear
/// function of the capsules above ~400 Hz and cannot be reproduced by any
/// matrix. The app encodes the array itself.
///
/// Capsule → face. The four capsules sit on the body's four faces (the user
/// has had the camera open: strictly front / back / left / right) and the
/// recordings say which channel is which, with no help from Studio:
/// * Front/back: the loudest coherent windows of the walking clip — the
///   operator, holding the camera at arm's length in front of them — reach
///   channel 2 first and channel 0 last in every one (relative arrival
///   +7.6 / −2.1 / −4.6 / −1.0 samples), and that pattern holds through 230°
///   of camera yaw while everything else sweeps: channel 2 faces the
///   operator (back), channel 0 the scene (front).
/// * Left/right: the camera's yaw was read off the image (phase correlation
///   of the frame centre; r = −1.00 with gyro y, positive gyro y = turning
///   right). A world-fixed sound must keep its world angle `θ_cam − yaw`
///   through the turns; the lag-fitted directions do so only with channel 1
///   on the LEFT and channel 3 on the RIGHT (world-angle spread 16° vs 48°
///   for the mirror image; both big turns agree). The operator's voice then
///   sits ~7° left of straight behind — a right-hand grip, the hand visible
///   at the frame's lower right.
///
/// So, 0° = front, +90° = left: **channel 0 front, 1 left, 2 back, 3
/// right**. (An earlier fit against Studio's ambisonic export read its
/// channels in the wrong order — the file is standard AmbiX `W Y Z X`, Z the
/// quiet one — and put the capsules on the corners. Studio's field is
/// view-locked and puts the operator at −15°, so its "front" is the
/// opposite face; it plays no part in the layout.) Every capsule is in
/// phase (correlation peak positive in 96–100 % of windows — no polarity
/// flip) and there is no path latency (pair offsets within ±1.6 samples).
/// The apparent acoustic spacing of an opposite pair, body diffraction
/// included, is ~12 samples ≈ 85 mm ([`KANDAO_CAPSULE_SPACING_M`]).
///
/// Channel timing. The second unit (firmware 1.1.6.6) records channel 0
/// 12–21 samples (0.25–0.45 ms) LATE in every one of its nine clips,
/// indoors and out, January to April — the lag of channel 0 against each of
/// the other three is the same in the diffuse windows, where acoustics
/// would give zero — and the amount differs between recordings; the first
/// unit (1.1.3.8) shows none. Left as is, the front capsule hears
/// everything after the back one, so the front/back dipole put a child
/// standing in front of that camera behind it. [`measure_quad_channel_delays`]
/// reads the offset off each file (the lag range of every capsule pair
/// over coherent windows is shifted by a latency and only widened by
/// acoustics, so the range midpoints give the per-channel delay; applied
/// when the four fit the six pairs within 3.5 samples and the largest is
/// ≥ 3) and [`QuadEncoder::with_calibration`] re-aligns the channels
/// before anything else.
///
/// Channel response. The capsules are not matched either. Quiet-window
/// band levels relative to the four-channel mean, the same in every clip
/// of a unit: unit 2 below ~500 Hz ch0 +5 dB, ch1 −8, ch2 −4, ch3 −5, and
/// from 600 Hz up ch0 +2, ch1 −3, ch2/ch3 0 (a low-cut on three of the
/// four, apparently); unit 1 a flatter ±3 dB spread (ch0 +2…+3, ch1/ch3
/// −2…−4). Unmatched, the front − back difference below 500 Hz is mostly
/// `(g0 − g2)·p` — pressure leaking into the dipole, larger than the
/// gradient itself — so the same probe measures each channel's level in
/// a low band (150–500 Hz) and a mid band (600–4000 Hz) over the quieter
/// half of the windows and the encoder balances each OPPOSITE PAIR
/// (ch0/ch2, ch1/ch3) about its geometric mean with a first-order low
/// shelf at 500 Hz: the pressure leak is gone, and no pair as a whole is
/// boosted or cut (matching all four to one mean instead scaled the X
/// and Y dipoles differently — ×0.84 vs ×1.5 in the mid band on unit 2 —
/// and pulled every direction ~20° sideways on both units' clips). ±12 dB
/// at most, nothing below 1 dB.
///
/// Encoding. A spaced pair's difference is the pressure GRADIENT: for a
/// plane wave it is `j·k·d·cos θ` times the pressure — 90° out of phase
/// with it and rising 6 dB/octave. Fed straight to an AmbiX decoder (whose
/// virtual microphones add W and X/Y in phase) that gives every direction
/// the same level and only a phase smear. So the difference is integrated
/// first: a leaky integrator (first-order low-pass at
/// [`KANDAO_GRAD_INTEGRATE_HZ`]: −90° and −6 dB/octave above it) scaled by
/// `c / (2π·f·d)` turns `front − back` into SN3D `X = p·cos θ` from ~250 Hz
/// up to where the pair aliases (~2 kHz; above that the dipoles fade,
/// which a first-order field takes better than lobes). Every channel, W
/// included, goes through the same two first-order high-passes at
/// [`KANDAO_GRAD_HP_HZ`] — a low-cut like most camera microphones'. The
/// dipoles get two more poles at [`KANDAO_GRAD_WIND_HZ`], because a
/// differential pair amplifies capsule mismatch and wind by `1/(k·d)` at
/// low frequency (measured on the park clip: without them a sub-300 Hz
/// gust peaked 10–18 dB above W and set the export's level; an 85 mm pair
/// resolves little below ~300 Hz anyway); W gets the matching first-order
/// all-pass `(β − z⁻¹)/(1 − βz⁻¹)`, which has exactly the phase of that
/// pole pair and leaves W's magnitude alone. So the dipoles stay
/// phase-matched to W except for the integrator's own 90° − atan(f/250 Hz)
/// (14° at 1 kHz, never inverted). `Z` = 0 (a planar array). Stereo is the
/// usual ±45° cardioid pair
/// of that field ([`FOA_STEREO_MATRIX`]). Everything carries
/// ×[`KANDAO_QUAD_GAIN`] (the capsules are recorded ~15 dB low) and the
/// exports then peak-normalise the window ([`kandao_window_gain`]).
/// [`QuadEncoder`] is the stateful implementation (filter state per
/// stream, sample rate given at open).
pub const KANDAO_QUAD_GAIN: f32 = 3.0;
/// Decoded channel of the capsule on each face.
pub const KANDAO_CAPSULE_FRONT: usize = 0;
pub const KANDAO_CAPSULE_LEFT: usize = 1;
pub const KANDAO_CAPSULE_BACK: usize = 2;
pub const KANDAO_CAPSULE_RIGHT: usize = 3;
/// Apparent acoustic spacing of an opposite capsule pair (m).
pub const KANDAO_CAPSULE_SPACING_M: f32 = 0.085;
/// Corner of the leaky integrator that turns a pair difference into a
/// velocity component (Hz).
pub const KANDAO_GRAD_INTEGRATE_HZ: f32 = 250.0;
/// Corner of the two first-order high-passes every channel goes through (Hz).
pub const KANDAO_GRAD_HP_HZ: f32 = 150.0;
/// Corner of the two extra poles on the dipoles against wind (Hz); W gets
/// the all-pass with the same phase.
pub const KANDAO_GRAD_WIND_HZ: f32 = 250.0;
const SPEED_OF_SOUND_M_S: f32 = 343.0;

/// Two cascaded first-order high-passes at `corner_hz`
/// (`β(1 − z⁻¹)/(1 − βz⁻¹)` each).
#[derive(Clone, Debug)]
struct Hp2 { beta: f32, s1: [f32; 2], s2: [f32; 2] }

impl Hp2 {
    fn new(rate: f32, corner_hz: f32) -> Self {
        Self { beta: 1.0 / (1.0 + std::f32::consts::TAU * corner_hz / rate), s1: [0.0; 2], s2: [0.0; 2] }
    }
    fn reset(&mut self) { self.s1 = [0.0; 2]; self.s2 = [0.0; 2]; }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        // y[n] = β·(y[n−1] + x[n] − x[n−1]); state [x[n−1], y[n−1]]
        let y1 = self.beta * (self.s1[1] + x - self.s1[0]);
        self.s1 = [x, y1];
        let y2 = self.beta * (self.s2[1] + y1 - self.s2[0]);
        self.s2 = [y1, y2];
        y2
    }
}

/// The first-order all-pass `(β − z⁻¹)/(1 − βz⁻¹)` sharing an [`Hp2`]'s
/// pole: unit magnitude, and exactly the phase of that [`Hp2`].
#[derive(Clone, Debug)]
struct Ap1 { beta: f32, s: [f32; 2] }

impl Ap1 {
    fn new(rate: f32, corner_hz: f32) -> Self {
        Self { beta: 1.0 / (1.0 + std::f32::consts::TAU * corner_hz / rate), s: [0.0; 2] }
    }
    fn reset(&mut self) { self.s = [0.0; 2]; }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        // y[n] = β·y[n−1] + β·x[n] − x[n−1]; state [x[n−1], y[n−1]]
        let y = self.beta * self.s[1] + self.beta * x - self.s[0];
        self.s = [x, y];
        y
    }
}

/// Gradient → velocity equaliser for one capsule pair (see the notes above
/// [`KANDAO_QUAD_GAIN`]): the shared high-passes, two more poles against
/// wind, a leaky integrator and the `c/(2π·f·d)` scale.
#[derive(Clone, Debug)]
struct GradEq { hp: Hp2, hp_wind: Hp2, alpha: f32, k: f32, lp: f32 }

impl GradEq {
    fn new(rate: f32) -> Self {
        let alpha = 1.0 - (-std::f32::consts::TAU * KANDAO_GRAD_INTEGRATE_HZ / rate).exp();
        let k = SPEED_OF_SOUND_M_S / (std::f32::consts::TAU * KANDAO_GRAD_INTEGRATE_HZ * KANDAO_CAPSULE_SPACING_M);
        Self { hp: Hp2::new(rate, KANDAO_GRAD_HP_HZ), hp_wind: Hp2::new(rate, KANDAO_GRAD_WIND_HZ), alpha, k, lp: 0.0 }
    }
    fn reset(&mut self) { self.hp.reset(); self.hp_wind.reset(); self.lp = 0.0; }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let h = self.hp_wind.run(self.hp.run(x));
        self.lp += self.alpha * (h - self.lp);
        self.k * self.lp
    }
}

/// An integer sample delay (a ring buffer; zero-length = pass-through).
#[derive(Clone, Debug, Default)]
struct Delay { buf: Vec<f32>, pos: usize }

impl Delay {
    fn new(samples: usize) -> Self { Self { buf: vec![0.0; samples], pos: 0 } }
    fn reset(&mut self) { self.buf.iter_mut().for_each(|v| *v = 0.0); self.pos = 0; }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        if self.buf.is_empty() { return x; }
        let y = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos += 1;
        if self.pos == self.buf.len() { self.pos = 0; }
        y
    }
}

/// Per-channel delays (samples) that re-align a QooCam capsule track — see
/// [`measure_quad_channel_delays`]; all zero when nothing is needed.
pub type QuadDelays = [usize; 4];

/// Per-channel level trims `[low band, mid band]` as linear gains (1.0 =
/// none) — see [`quad_trims_from_pcm`].
pub type QuadTrims = [[f32; 2]; 4];

/// Everything the encoder needs to undo a file's channel mismatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadCalibration { pub delays: QuadDelays, pub trims: QuadTrims }

impl Default for QuadCalibration {
    fn default() -> Self { Self { delays: [0; 4], trims: [[1.0; 2]; 4] } }
}

/// Corner between the two trim bands (Hz).
pub const KANDAO_TRIM_SHELF_HZ: f32 = 500.0;

/// A per-channel first-order low shelf: `g_mid·x + (g_low − g_mid)·LP(x)`.
#[derive(Clone, Debug)]
struct Shelf { alpha: f32, g_low: f32, g_mid: f32, lp: f32 }

impl Shelf {
    fn new(rate: f32, trims: [f32; 2]) -> Self {
        Self { alpha: 1.0 - (-std::f32::consts::TAU * KANDAO_TRIM_SHELF_HZ / rate).exp(), g_low: trims[0], g_mid: trims[1], lp: 0.0 }
    }
    fn reset(&mut self) { self.lp = 0.0; }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        self.lp += self.alpha * (x - self.lp);
        self.g_mid * x + (self.g_low - self.g_mid) * self.lp
    }
}

/// The QooCam capsule array → first-order AmbiX `[W, Y, Z, X]` (and the
/// stereo pair of it), carrying the filters' state — one per decoded
/// stream. `gain` multiplies the built-in ×[`KANDAO_QUAD_GAIN`].
#[derive(Clone, Debug)]
pub struct QuadEncoder { gain: f32, rate: f32, align: [Delay; 4], trim: [Shelf; 4], hp_w: Hp2, ap_w: Ap1, eq_x: GradEq, eq_y: GradEq }

impl QuadEncoder {
    pub fn new(sample_rate: u32, gain: f32) -> Self {
        let r = sample_rate.max(1) as f32;
        Self {
            gain: gain * KANDAO_QUAD_GAIN, rate: r, align: Default::default(),
            trim: std::array::from_fn(|_| Shelf::new(r, [1.0, 1.0])),
            hp_w: Hp2::new(r, KANDAO_GRAD_HP_HZ), ap_w: Ap1::new(r, KANDAO_GRAD_WIND_HZ),
            eq_x: GradEq::new(r), eq_y: GradEq::new(r),
        }
    }
    /// Delay each channel by `delays[k]` samples (at this encoder's rate)
    /// before encoding — the file's own channel misalignment undone.
    pub fn with_delays(mut self, delays: QuadDelays) -> Self {
        for (k, d) in delays.iter().enumerate() { self.align[k] = Delay::new(*d); }
        self
    }
    /// Trim each channel's low and mid band to the array mean.
    pub fn with_trims(mut self, trims: QuadTrims) -> Self {
        for (k, t) in trims.iter().enumerate() { self.trim[k] = Shelf::new(self.rate, *t); }
        self
    }
    /// Both of the above from one [`QuadCalibration`] (delays already at
    /// this encoder's rate).
    pub fn with_calibration(self, cal: QuadCalibration) -> Self { self.with_delays(cal.delays).with_trims(cal.trims) }
    /// Forget the filter history (after a seek).
    pub fn reset(&mut self) {
        self.align.iter_mut().for_each(Delay::reset);
        self.trim.iter_mut().for_each(Shelf::reset);
        self.hp_w.reset(); self.ap_w.reset(); self.eq_x.reset(); self.eq_y.reset();
    }
    /// One sample of the four capsules (decoded channel order) → `[W, Y, Z, X]`.
    #[inline]
    pub fn foa(&mut self, c: [f32; 4]) -> [f32; 4] {
        let c: [f32; 4] = std::array::from_fn(|k| self.trim[k].run(self.align[k].run(c[k])));
        let w = self.ap_w.run(self.hp_w.run(0.25 * (c[0] + c[1] + c[2] + c[3])));
        let x = self.eq_x.run(c[KANDAO_CAPSULE_FRONT] - c[KANDAO_CAPSULE_BACK]);
        let y = self.eq_y.run(c[KANDAO_CAPSULE_LEFT] - c[KANDAO_CAPSULE_RIGHT]);
        [self.gain * w, self.gain * y, 0.0, self.gain * x]
    }
    /// One sample → `[L, R]` (the ±45° cardioids of [`FOA_STEREO_MATRIX`]).
    #[inline]
    pub fn stereo(&mut self, c: [f32; 4]) -> [f32; 2] { foa_to_stereo_sample(self.foa(c)) }
}

/// Which stereo pair a 4-channel source gets in the player: the codec
/// alone cannot tell the X6's AmbiX from the QooCam's capsules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMix {
    /// First-order AmbiX through [`FOA_STEREO_MATRIX`].
    Foa,
    /// The QooCam's capsules through [`QuadEncoder`].
    Quad,
}

/// How a 4-channel track is mixed on the way out of [`transcode_audio_window`].
#[derive(Clone, Copy, Debug)]
pub enum Mix<'a> {
    /// A fixed matrix, one row per output channel (the X6's AmbiX → stereo).
    Matrix(&'a [[f32; 4]]),
    /// The QooCam capsules → AmbiX `[W, Y, Z, X]` through [`QuadEncoder`].
    QuadFoa { gain: f32, cal: QuadCalibration },
    /// The QooCam capsules → the stereo pair through [`QuadEncoder`].
    QuadStereo { gain: f32, cal: QuadCalibration },
}

impl Mix<'_> {
    fn out_channels(&self) -> usize {
        match self { Mix::Matrix(m) => m.len(), Mix::QuadFoa { .. } => 4, Mix::QuadStereo { .. } => 2 }
    }
}

/// A [`Mix`] in flight: the matrix, or the encoder with its state.
enum MixState<'a> { Matrix(&'a [[f32; 4]]), QuadFoa(QuadEncoder), QuadStereo(QuadEncoder) }

impl<'a> MixState<'a> {
    fn new(mix: Mix<'a>, rate: u32) -> Self {
        match mix {
            Mix::Matrix(m) => MixState::Matrix(m),
            Mix::QuadFoa { gain, cal } => MixState::QuadFoa(QuadEncoder::new(rate, gain).with_calibration(cal)),
            Mix::QuadStereo { gain, cal } => MixState::QuadStereo(QuadEncoder::new(rate, gain).with_calibration(cal)),
        }
    }
    #[inline]
    fn apply(&mut self, s: [f32; 4], out: &mut Vec<f32>) {
        match self {
            MixState::Matrix(m) => for row in m.iter() {
                out.push(row[0] * s[0] + row[1] * s[1] + row[2] * s[2] + row[3] * s[3]);
            },
            MixState::QuadFoa(e) => out.extend_from_slice(&e.foa(s)),
            MixState::QuadStereo(e) => out.extend_from_slice(&e.stereo(s)),
        }
    }
}

/// One sample through a `[L, R] × 4` mixing matrix.
#[inline]
pub fn mix_sample(m: &[[f32; 4]; 2], ch: [f32; 4]) -> [f32; 2] {
    [
        m[0][0] * ch[0] + m[0][1] * ch[1] + m[0][2] * ch[2] + m[0][3] * ch[3],
        m[1][0] * ch[0] + m[1][1] * ch[1] + m[1][2] * ch[2] + m[1][3] * ch[3],
    ]
}

/// Codec-ID set we recognize as "uncompressed PCM suitable for
/// ambisonic extraction". Extracted to its own function so the list of
/// fallback codecs is documented in one place.
fn is_pcm_codec_id(codec_id: i32) -> bool {
    use ffmpeg::ffi::AVCodecID::*;
    let id = unsafe { std::mem::transmute::<i32, ffmpeg::ffi::AVCodecID>(codec_id) };
    matches!(id,
          AV_CODEC_ID_PCM_S24LE
        | AV_CODEC_ID_PCM_S16LE
        | AV_CODEC_ID_PCM_S32LE
        | AV_CODEC_ID_PCM_F32LE
    )
}

/// Extract the ambisonic audio track from `input` into a fresh WAV
/// file at `output_wav`. Stream-copy remux (no decode / re-encode):
/// the raw PCM packets land in the WAV's `data` chunk byte-for-byte.
///
/// Caller should run `probe_ambisonic` first to confirm a track exists
/// AND to choose between APAC vs stereo-AAC paths. This function
/// errors if the source has no ambisonic stream.
pub fn extract_ambisonic_to_wav(input: &Path, output_wav: &Path) -> Result<AmbisonicInfo> {
    extract_ambisonic_to_wav_window(input, output_wav, 0.0, None)
}

/// Window-aware ambisonic extraction: only the `[offset_s, offset_s +
/// dur_s)` slice of the source's 4-ch track is written (rebased to start
/// at 0), so a TRIMMED export's APAC/ambisonic audio matches the
/// exported video. `offset_s = 0, dur_s = None` extracts the whole track.
pub fn extract_ambisonic_to_wav_window(
    input: &Path,
    output_wav: &Path,
    offset_s: f64,
    dur_s: Option<f64>,
) -> Result<AmbisonicInfo> {
    crate::decode::init();
    let info = probe_spatial_4ch(input)?
        .ok_or_else(|| Error::Ffmpeg(format!(
            "no ambisonic audio track in {input:?}"
        )))?;
    if !is_pcm_codec_id(info.codec_id) {
        // Compressed (the X6's AAC): decode to 24-bit PCM — the packet copy
        // below is PCM-only.
        let n = transcode_audio_window(
            input, info.stream_index, output_wav, offset_s, dur_s, None,
            AudioOut::Pcm24 { channels: info.channels as i32 },
        )?;
        tracing::info!(wav = %output_wav.display(), samples = n, channels = info.channels,
            sample_rate = info.sample_rate, "decoded 4-ch AAC ambisonic audio to WAV");
        return Ok(info);
    }

    let mut ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open input: {e}")))?;
    if offset_s > 0.001 {
        let ts = (offset_s * 1_000_000.0) as i64;
        let _ = ictx.seek(ts, ..ts);
    }
    let mut octx = ffmpeg::format::output(&output_wav)
        .map_err(|e| Error::Ffmpeg(format!("open output WAV {output_wav:?}: {e}")))?;

    // Build the output audio stream as a parameter copy of the input.
    let src_params = ictx.stream(info.stream_index)
        .ok_or_else(|| Error::Ffmpeg("ambisonic stream vanished".into()))?
        .parameters();
    let src_time_base = ictx.stream(info.stream_index).unwrap().time_base();

    let out_stream_index = {
        let mut out_stream = octx.add_stream(ffmpeg::codec::Id::None)
            .map_err(|e| Error::Ffmpeg(format!("add_stream: {e}")))?;
        out_stream.set_parameters(src_params);
        // The wav muxer expects an `audio` AVCodecParameters with the
        // PCM codec_id populated, which set_parameters() copied verbatim
        // from the input. Time base mirrors the source so PTS values
        // remain meaningful.
        out_stream.set_time_base(src_time_base);
        // Force `codec_tag = 0` so ffmpeg picks the right wav format
        // (the source MOV uses MOV-specific tags that would confuse
        // the wav muxer's "tag → format" round-trip).
        unsafe {
            let raw = out_stream.parameters().as_mut_ptr();
            (*raw).codec_tag = 0;
        }
        out_stream.index()
    };

    octx.write_header()
        .map_err(|e| Error::Ffmpeg(format!("WAV write_header: {e}")))?;

    // Window bounds in source seconds (None dur → to EOF).
    let tb_s = src_time_base.numerator() as f64 / src_time_base.denominator().max(1) as f64;
    let win_end = dur_s.map(|d| offset_s + d);
    let mut packet_count = 0u64;
    for (stream, mut packet) in ictx.packets() {
        if stream.index() != info.stream_index { continue; }
        // Window by the packet's source time. (Seek lands on/near the
        // keyframe before `offset`; skip the run-in, stop past the end.)
        let t = packet.pts().or(packet.dts()).unwrap_or(0) as f64 * tb_s;
        if t < offset_s - 0.05 { continue; }
        if let Some(end) = win_end { if t >= end { break; } }
        packet.set_stream(out_stream_index);
        // The wav muxer derives its own packet timing from sample count;
        // we just need PTS to monotonically increase. Rescale from
        // source time-base into output's (we set output to match input
        // above, so this is a noop in practice — but keeping it
        // explicit guards against future muxer surprise).
        packet.rescale_ts(
            src_time_base,
            octx.stream(out_stream_index).unwrap().time_base(),
        );
        packet.write_interleaved(&mut octx)
            .map_err(|e| Error::Ffmpeg(format!("write audio packet: {e}")))?;
        packet_count += 1;
    }
    octx.write_trailer()
        .map_err(|e| Error::Ffmpeg(format!("WAV write_trailer: {e}")))?;

    tracing::info!(
        wav = %output_wav.display(),
        packets = packet_count,
        channels = info.channels,
        sample_rate = info.sample_rate,
        "extracted ambisonic audio to WAV"
    );
    Ok(info)
}

/// An `N`-channel layout in ffmpeg's AMBISONIC order (`"ambisonic 1"` for
/// 4). The WAV muxer then writes channel mask 0, which AVFoundation reads as
/// "no channel layout" — exactly what GoPro's packet-copied track yields.
/// The positional default (`4.0` = L R C Cs) made the APAC helper's
/// AVAssetWriter "convert" quad → HOA and encode SILENCE (2026-10-10).
fn ambisonic_layout(channels: i32) -> ffmpeg::ChannelLayout {
    let order = ((channels as f64).sqrt() - 1.0).round().max(0.0) as i32; // (order+1)² channels
    let spec = std::ffi::CString::new(format!("ambisonic {order}")).expect("no NUL");
    // SAFETY: zero-initialised AVChannelLayout filled by ffmpeg from the spec string.
    let mut raw: ffmpeg::ffi::AVChannelLayout = unsafe { std::mem::zeroed() };
    let r = unsafe { ffmpeg::ffi::av_channel_layout_from_string(&mut raw, spec.as_ptr()) };
    if r < 0 || raw.nb_channels != channels {
        // Not a square count (or an old ffmpeg): fall back to an unspecified
        // order, which also writes no positional mask.
        unsafe {
            ffmpeg::ffi::av_channel_layout_uninit(&mut raw);
            raw = std::mem::zeroed();
            raw.order = ffmpeg::ffi::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC;
            raw.nb_channels = channels;
        }
    }
    ffmpeg::ChannelLayout::from(raw)
}

/// A stream's channel layout with an EMPTY mask (a WAV says "2 channels"
/// and nothing else) made swresample refuse every decoded frame ("Input
/// changed"): use the default layout for that channel count instead.
/// A decoder's layout with an EMPTY mask ("2 channels", as a WAV or the
/// PCM track of a Blackmagic RAW clip reports) replaced by the default
/// layout for that channel count — swresample rejects every frame that
/// carries the empty layout with "Input changed". Frames must be stamped
/// with the same layout before they go through a resampler.
pub fn normalized_layout(l: ffmpeg::ChannelLayout) -> ffmpeg::ChannelLayout {
    if l.is_empty() && l.channels() > 0 { ffmpeg::ChannelLayout::default(l.channels() as i32) } else { l }
}

/// Output of [`transcode_audio_window`].
enum AudioOut {
    /// Stereo AAC-LC (ffmpeg's native encoder) in an `.m4a`.
    AacStereo { bit_rate: usize },
    /// 4-channel AAC-LC (the layout tag is "4.0", like the camera files;
    /// SA3D is what gives it meaning) — a first-order ambisonic temp.
    Aac4 { bit_rate: usize },
    /// 24-bit PCM, `channels` wide, in a `.wav`.
    Pcm24 { channels: i32 },
}

/// Decode the `[offset_s, offset_s + dur_s)` window of audio stream
/// `stream_index`, optionally mix its 4 channels through `mix` (one row
/// per OUTPUT channel), and encode it to `out_path`. The output starts EXACTLY at
/// `offset_s` (sample-accurate: decoding begins a few packets earlier so
/// the codec's overlap has settled, then the head is dropped) and runs a
/// little past `offset_s + dur_s` (packet granularity) so a mux can cut at
/// `dur_s`. Returns the number of samples written.
fn transcode_audio_window(
    input: &Path,
    stream_index: usize,
    out_path: &Path,
    offset_s: f64,
    dur_s: Option<f64>,
    mix: Option<Mix<'_>>,
    out: AudioOut,
) -> Result<u64> {
    use ffmpeg::format::sample::{Sample, Type};
    use ffmpeg::ChannelLayout;
    crate::decode::init();

    let mut ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open input {input:?}: {e}")))?;
    if offset_s > 0.001 {
        let ts = (offset_s * 1_000_000.0) as i64;
        let _ = ictx.seek(ts, ..ts);
    }
    let (src_params, src_tb) = {
        let st = ictx.stream(stream_index)
            .ok_or_else(|| Error::Ffmpeg(format!("audio stream {stream_index} vanished")))?;
        (st.parameters(), st.time_base())
    };
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(src_params)
        .map_err(|e| Error::Ffmpeg(format!("decoder params: {e}")))?
        .decoder().audio()
        .map_err(|e| Error::Ffmpeg(format!("audio decoder: {e}")))?;
    let rate = decoder.rate();
    let src_layout = normalized_layout(decoder.channel_layout());
    let in_ch = src_layout.channels().max(1) as usize;
    if mix.is_some() && in_ch != 4 {
        return Err(Error::Ffmpeg(format!("a 4-channel mix needs 4 input channels, stream has {in_ch}")));
    }
    // Decoded frames → packed f32 at the source rate (format change only).
    let mut to_f32 = ffmpeg::software::resampling::Context::get(
        decoder.format(), src_layout, rate,
        Sample::F32(Type::Packed), src_layout, rate,
    ).map_err(|e| Error::Ffmpeg(format!("resampler: {e}")))?;

    // The muxer is named EXPLICITLY rather than guessed from the extension:
    // `.m4a` resolves to the `ipod` muxer, which the FFmpeg build bundled
    // with the Windows app (the gpl-lite dist) does not include, so the
    // extension-guessed open failed there with "Unable to choose an output
    // format" and every X6 360°-Audio export fell back to muxing the raw
    // 4-channel track. `mp4` carries an AAC-only stream fine and is present
    // in every build that can write our video; `wav` likewise.
    let (codec_id, enc_fmt, enc_layout, out_ch, bit_rate, muxer) = match out {
        AudioOut::AacStereo { bit_rate } =>
            (ffmpeg::codec::Id::AAC, Sample::F32(Type::Planar), ChannelLayout::STEREO, 2usize, bit_rate, "mp4"),
        AudioOut::Aac4 { bit_rate } =>
            (ffmpeg::codec::Id::AAC, Sample::F32(Type::Planar), ChannelLayout::_4POINT0, 4usize, bit_rate, "mp4"),
        AudioOut::Pcm24 { channels } =>
            (ffmpeg::codec::Id::PCM_S24LE, Sample::I32(Type::Packed), ambisonic_layout(channels), channels as usize, 0, "wav"),
    };
    if mix.is_none() && out_ch != in_ch {
        return Err(Error::Ffmpeg(format!("channel count {in_ch} → {out_ch} without a mix")));
    }
    if let Some(m) = mix {
        if m.out_channels() != out_ch {
            return Err(Error::Ffmpeg(format!("mix has {} rows for {out_ch} output channels", m.out_channels())));
        }
    }
    let mut mix_state = mix.map(|m| MixState::new(m, rate));
    let codec = ffmpeg::encoder::find(codec_id)
        .ok_or_else(|| Error::Ffmpeg(format!("no encoder for {codec_id:?}")))?;
    let mut octx = ffmpeg::format::output_as(&out_path, muxer)
        .map_err(|e| Error::Ffmpeg(format!("open output {out_path:?} as {muxer}: {e}")))?;
    let global_header = octx.format().flags().contains(ffmpeg::format::Flags::GLOBAL_HEADER);
    let (ost_idx, mut encoder) = {
        let mut ost = octx.add_stream(codec)
            .map_err(|e| Error::Ffmpeg(format!("add stream: {e}")))?;
        let mut enc = ffmpeg::codec::context::Context::from_parameters(ost.parameters())
            .map_err(|e| Error::Ffmpeg(format!("encoder params: {e}")))?
            .encoder().audio()
            .map_err(|e| Error::Ffmpeg(format!("audio encoder: {e}")))?;
        enc.set_rate(rate as i32);
        enc.set_format(enc_fmt);
        enc.set_channel_layout(enc_layout);
        enc.set_time_base((1, rate as i32));
        if bit_rate > 0 { enc.set_bit_rate(bit_rate); }
        if global_header { enc.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER); }
        let encoder = enc.open_as(codec)
            .map_err(|e| Error::Ffmpeg(format!("open encoder {codec_id:?}: {e}")))?;
        ost.set_parameters(&encoder);
        (ost.index(), encoder)
    };
    octx.write_header().map_err(|e| Error::Ffmpeg(format!("write_header: {e}")))?;
    let ost_tb = octx.stream(ost_idx).unwrap().time_base();
    let enc_tb = ffmpeg::Rational(1, rate as i32);
    let frame_size = match encoder.frame_size() { 0 => 4096, n => n as usize };

    // Interleaved f32 output samples waiting to be framed.
    let mut pending: Vec<f32> = Vec::with_capacity(frame_size * out_ch * 4);
    let mut written: u64 = 0;
    let mut packet = ffmpeg::Packet::empty();
    let mut drain = |encoder: &mut ffmpeg::encoder::Audio, octx: &mut ffmpeg::format::context::Output| -> Result<()> {
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(ost_idx);
            packet.rescale_ts(enc_tb, ost_tb);
            packet.write_interleaved(octx)
                .map_err(|e| Error::Ffmpeg(format!("write audio packet: {e}")))?;
        }
        Ok(())
    };
    let mut emit = |pending: &mut Vec<f32>, flush: bool, encoder: &mut ffmpeg::encoder::Audio,
                    octx: &mut ffmpeg::format::context::Output, written: &mut u64| -> Result<()> {
        while pending.len() >= frame_size * out_ch || (flush && !pending.is_empty()) {
            let n = (pending.len() / out_ch).min(frame_size);
            let mut frame = ffmpeg::frame::Audio::new(enc_fmt, frame_size, enc_layout);
            frame.set_rate(rate);
            frame.set_pts(Some(*written as i64));
            // Plane access through the raw pointers: ffmpeg sets only
            // `linesize[0]` for planar audio, so `data_mut(c > 0)` is empty.
            match enc_fmt {
                Sample::F32(Type::Planar) => {
                    for c in 0..out_ch {
                        // SAFETY: `Audio::new` allocated `frame_size` samples per plane.
                        let plane = unsafe {
                            std::slice::from_raw_parts_mut((*frame.as_mut_ptr()).data[c] as *mut f32, frame_size)
                        };
                        for i in 0..frame_size {
                            plane[i] = if i < n { pending[i * out_ch + c] } else { 0.0 };
                        }
                    }
                }
                _ => {
                    // SAFETY: packed — one plane of `frame_size × out_ch` samples.
                    let plane = unsafe {
                        std::slice::from_raw_parts_mut((*frame.as_mut_ptr()).data[0] as *mut i32, frame_size * out_ch)
                    };
                    for i in 0..frame_size * out_ch {
                        plane[i] = if i < n * out_ch {
                            (pending[i].clamp(-1.0, 1.0) as f64 * 2_147_483_647.0) as i32
                        } else { 0 };
                    }
                }
            }
            // PCM takes any frame length; AAC wants full frames — a padded
            // last frame is at most 1023 samples (21 ms) of silence.
            if frame_size != 4096 || n == frame_size { frame.set_samples(frame_size); } else { frame.set_samples(n); }
            encoder.send_frame(&frame).map_err(|e| Error::Ffmpeg(format!("send_frame: {e}")))?;
            drain(encoder, octx)?;
            *written += n as u64;
            pending.drain(..n * out_ch);
        }
        Ok(())
    };

    let tb_s = src_tb.numerator() as f64 / src_tb.denominator().max(1) as f64;
    // Warm-up / tail margins in source seconds: a few packets before the
    // window (the AAC frame after a mid-stream start lacks its overlap; it
    // falls inside the dropped head) and a few after it.
    const EDGE_S: f64 = 0.1;
    let win_end = dur_s.map(|d| offset_s + d + EDGE_S);
    let mut decoded = ffmpeg::frame::Audio::empty();
    let mut f32_frame = ffmpeg::frame::Audio::empty();
    // Samples still to drop so the output starts exactly at `offset_s`
    // (set from the first kept packet's timestamp).
    let mut drop_left: Option<usize> = None;
    let mut push = |f: &ffmpeg::frame::Audio, pending: &mut Vec<f32>, drop_left: &mut Option<usize>| {
        let n = f.samples();
        let bytes = &f.data(0)[..n * in_ch * 4];
        let mut src: &[f32] = bytemuck::cast_slice(bytes);
        if let Some(d) = drop_left {
            let take = (*d).min(n);
            src = &src[take * in_ch..];
            *d -= take;
        }
        match mix_state.as_mut() {
            Some(m) => for s in src.chunks_exact(4) { m.apply([s[0], s[1], s[2], s[3]], pending); },
            None => pending.extend_from_slice(src),
        }
    };
    for (stream, pkt) in ictx.packets() {
        if stream.index() != stream_index { continue; }
        let t = pkt.pts().or(pkt.dts()).unwrap_or(0) as f64 * tb_s;
        if t < offset_s - EDGE_S { continue; }
        if let Some(end) = win_end { if t >= end { break; } }
        if drop_left.is_none() {
            drop_left = Some(((offset_s - t) * rate as f64).round().max(0.0) as usize);
        }
        if decoder.send_packet(&pkt).is_err() { continue; }
        while decoder.receive_frame(&mut decoded).is_ok() {
            if { decoded.set_channel_layout(src_layout); to_f32.run(&decoded, &mut f32_frame) }.is_ok() && f32_frame.samples() > 0 {
                push(&f32_frame, &mut pending, &mut drop_left);
            }
        }
        emit(&mut pending, false, &mut encoder, &mut octx, &mut written)?;
    }
    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        if { decoded.set_channel_layout(src_layout); to_f32.run(&decoded, &mut f32_frame) }.is_ok() && f32_frame.samples() > 0 {
            push(&f32_frame, &mut pending, &mut drop_left);
        }
    }
    if to_f32.flush(&mut f32_frame).is_ok() && f32_frame.samples() > 0 {
        push(&f32_frame, &mut pending, &mut drop_left);
    }
    emit(&mut pending, true, &mut encoder, &mut octx, &mut written)?;
    encoder.send_eof().map_err(|e| Error::Ffmpeg(format!("send_eof: {e}")))?;
    drain(&mut encoder, &mut octx)?;
    octx.write_trailer().map_err(|e| Error::Ffmpeg(format!("write_trailer: {e}")))?;
    Ok(written)
}

/// Encode the window `[offset_s, offset_s + dur_s)` of a plain stereo (or
/// mono) track — a WAV the BRAW helper extracted, say — to a stereo AAC
/// `.m4a` starting at its own 0, like [`foa_to_stereo_aac`]. Returns the
/// samples written.
pub fn stereo_to_aac(input: &Path, output_m4a: &Path, offset_s: f64, dur_s: Option<f64>, bit_rate: usize) -> Result<u64> {
    let (stream_index, channels, _rate) = probe_any_audio(input)?
        .ok_or_else(|| Error::Ffmpeg(format!("no audio track in {input:?}")))?;
    if channels != 2 {
        return Err(Error::Ffmpeg(format!("stereo_to_aac: {channels}-channel track in {input:?}")));
    }
    let t0 = std::time::Instant::now();
    let n = transcode_audio_window(
        input, stream_index, output_m4a, offset_s, dur_s,
        None, AudioOut::AacStereo { bit_rate },
    )?;
    tracing::info!(m4a = %output_m4a.display(), samples = n, took_ms = t0.elapsed().as_millis() as u64,
        "audio: encoded a stereo window to AAC");
    Ok(n)
}

/// Decode the X6's 4-channel 360° Audio (first-order AmbiX) to a
/// front-facing stereo AAC `.m4a` — see [`FOA_STEREO_MATRIX`]. Only the
/// `[offset_s, offset_s + dur_s)` window (the whole track when `dur_s` is
/// `None`); the file starts exactly at `offset_s`, so a mux plays it from
/// its own 0. Returns the number of samples written.
pub fn foa_to_stereo_aac(input: &Path, output_m4a: &Path, offset_s: f64, dur_s: Option<f64>, bit_rate: usize) -> Result<u64> {
    let info = probe_foa_aac(input)?
        .ok_or_else(|| Error::Ffmpeg(format!("no 4-channel AAC (360° Audio) track in {input:?}")))?;
    let t0 = std::time::Instant::now();
    let n = transcode_audio_window(
        input, info.stream_index, output_m4a, offset_s, dur_s,
        Some(Mix::Matrix(&FOA_STEREO_MATRIX[..])), AudioOut::AacStereo { bit_rate },
    )?;
    tracing::info!(m4a = %output_m4a.display(), samples = n, sample_rate = info.sample_rate,
        window_start_s = offset_s, window_s = dur_s.unwrap_or(f64::NAN), took_ms = t0.elapsed().as_millis() as u64,
        "X6 360° Audio: decoded first-order AmbiX to a front-facing stereo AAC");
    Ok(n)
}

/// Peak the export window will reach, normalised so a QooCam export is
/// neither inaudible nor clipped: the capsules are recorded 15–20 dB below
/// a normal level and vary by that much between clips, and the app has no
/// limiter. The window is decoded once through the AmbiX matrix (W carries
/// the pressure; the stereo pair is derived from the same field, so one
/// gain serves every format) and the encoder is scaled so that peak lands at
/// [`KANDAO_PEAK_TARGET`], within `0.5×…16×` of the encoder's built-in ×3
/// (the park clip needs ~12× to reach Studio's own output level).
pub const KANDAO_PEAK_TARGET: f32 = 0.7;

/// What a Kandao export window needs before encoding: the channel
/// calibration read off the file ([`measure_quad_calibration`]) and the
/// gain on top of the encoder's ×3 (see [`KANDAO_PEAK_TARGET`]; 1.0 when the
/// window is silent or cannot be read).
pub fn kandao_window_prep(input: &Path, stream_index: usize, offset_s: f64, dur_s: Option<f64>) -> (f32, QuadCalibration) {
    let cal = measure_quad_calibration(input, stream_index, offset_s, dur_s);
    let gain = match mixed_window_peak(input, stream_index, offset_s, dur_s, Mix::QuadFoa { gain: 1.0, cal }) {
        Ok(peak) if peak > 1e-5 => (KANDAO_PEAK_TARGET / peak).clamp(0.5, 16.0),
        Ok(_) => 1.0,
        Err(e) => { tracing::warn!("kandao audio: peak pre-pass failed ({e}) — unity gain"); 1.0 }
    };
    (gain, cal)
}

/// [`measure_quad_channel_delays`] and [`quad_trims_from_pcm`] from one
/// decode of the first [`QUAD_DELAY_PROBE_S`] of the TRACK — not of the
/// export window: the latency and the capsule mismatch are properties of
/// the recording, a 10 s window gave the delay fit too few coherent windows
/// to pass its gate (64, residual 8.4 on the kids clip, where 24 s gives
/// 187 / 2.0), and reading the same stretch for every window of a file and
/// for the player keeps them all on one calibration. `offset_s` / `dur_s`
/// are kept for the call shape; they do not move the probe.
pub fn measure_quad_calibration(input: &Path, stream_index: usize, offset_s: f64, dur_s: Option<f64>) -> QuadCalibration {
    let _ = (offset_s, dur_s);
    let probe = Some(QUAD_DELAY_PROBE_S);
    let mut pcm: Vec<f32> = Vec::new();
    let rate = match decode_quad_window(input, stream_index, 0.0, probe, &mut |chunk| pcm.extend_from_slice(chunk)) {
        Ok(r) => r as f32,
        Err(e) => { tracing::warn!("kandao audio: calibration probe failed ({e}) — none applied"); return QuadCalibration::default(); }
    };
    QuadCalibration { delays: quad_channel_delays_from_pcm(&pcm, rate), trims: quad_trims_from_pcm(&pcm, rate) }
}

/// Per-channel level trims of a QooCam capsule track (see the
/// channel-response notes above [`KANDAO_QUAD_GAIN`]): each channel's
/// energy in a low band (150–500 Hz) and a mid band (600–4000 Hz) per
/// 2048-sample window; over the quieter half of the windows (by mid-band
/// total — the ambient, not the talker in front of one capsule) the
/// median of the channel's level relative to the four-channel mean; each
/// opposite pair is then balanced about its own geometric mean (half the
/// pair's difference up on one, down on the other), clamped to ±12 dB,
/// unity when every pair is within 1 dB.
pub fn quad_trims_from_pcm(pcm: &[f32], rate: f32) -> QuadTrims {
    const WIN: usize = 2048;
    let n = pcm.len() / 4;
    if n < WIN * 8 { return [[1.0; 2]; 4]; }
    let band = |lo: f32, hi: f32| -> Vec<Vec<f32>> {
        let bh = 1.0 / (1.0 + std::f32::consts::TAU * lo / rate);
        let al = 1.0 - (-std::f32::consts::TAU * hi / rate).exp();
        (0..4).map(|k| {
            let mut v: Vec<f32> = pcm.iter().skip(k).step_by(4).copied().collect();
            for _ in 0..2 {
                let (mut xp, mut yp) = (0.0f32, 0.0f32);
                for x in v.iter_mut() { let y = bh * (yp + *x - xp); xp = *x; yp = y; *x = y; }
                let mut lp = 0.0f32;
                for x in v.iter_mut() { lp += al * (*x - lp); *x = lp; }
            }
            v
        }).collect()
    };
    let bands = [band(150.0, 500.0), band(600.0, 4000.0)];
    let nw = n / WIN;
    // per window: [band][channel] energy
    let mut e = vec![[[0f32; 4]; 2]; nw];
    for (w, ew) in e.iter_mut().enumerate() {
        for b in 0..2 { for k in 0..4 {
            ew[b][k] = bands[b][k][w * WIN..(w + 1) * WIN].iter().map(|v| v * v).sum::<f32>() / WIN as f32;
        } }
    }
    let mut total: Vec<f32> = e.iter().map(|ew| ew[1].iter().sum::<f32>()).collect();
    total.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = total[total.len() / 2];
    let mut level = [[0f32; 2]; 4];   // dB relative to the four-channel mean, per band
    for b in 0..2 {
        for k in 0..4 {
            let mut rel: Vec<f32> = e.iter()
                .filter(|ew| ew[1].iter().sum::<f32>() <= median)
                .filter_map(|ew| {
                    let mean = ew[b].iter().sum::<f32>() / 4.0;
                    (mean > 1e-12 && ew[b][k] > 1e-12).then(|| 10.0 * (ew[b][k] / mean).log10())
                }).collect();
            if rel.len() < 20 { return [[1.0; 2]; 4]; }
            rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
            level[k][b] = rel[rel.len() / 2];
        }
    }
    let mut trims = [[1.0f32; 2]; 4];
    let mut largest_db = 0f32;
    for (i, j) in [(KANDAO_CAPSULE_FRONT, KANDAO_CAPSULE_BACK), (KANDAO_CAPSULE_LEFT, KANDAO_CAPSULE_RIGHT)] {
        for b in 0..2 {
            let half = (0.5 * (level[i][b] - level[j][b])).clamp(-12.0, 12.0);
            largest_db = largest_db.max(half.abs() * 2.0);
            trims[i][b] = 10f32.powf(-half / 20.0);
            trims[j][b] = 10f32.powf(half / 20.0);
        }
    }
    if largest_db < 1.0 { return [[1.0; 2]; 4]; }
    tracing::info!(?level, ?trims, "kandao audio: balancing the capsule pairs (low / mid band)");
    trims
}

/// Seconds of a track the channel-delay measurement looks at.
const QUAD_DELAY_PROBE_S: f64 = 24.0;

/// Per-channel recording delay of a QooCam capsule track, measured from
/// the file (see the channel-timing notes above [`KANDAO_QUAD_GAIN`]):
/// the first [`QUAD_DELAY_PROBE_S`] of the window `[offset_s, …)`, 300–4000
/// Hz, 2048-sample windows; in every window where all six capsule pairs
/// correlate above 0.5 within ±40 samples, the pair lags are kept; the
/// 3rd–97th percentile midpoint of each pair's lags is its latency
/// difference, the four per-channel latencies follow by least squares
/// (complete graph: `t_k = ¼·Σ_j lag(k, j)`), and the delay to apply is
/// `max(t) − t_k`. All zero (nothing applied) unless at least 40 windows
/// qualify, the fit residual is ≤ 3.5 samples and the largest delay is
/// ≥ 3 — a unit that is simply fine measures 0–2.
pub fn measure_quad_channel_delays(input: &Path, stream_index: usize, offset_s: f64, dur_s: Option<f64>) -> QuadDelays {
    let _ = (offset_s, dur_s);   // see `measure_quad_calibration`: the probe is the track's first stretch
    let probe = Some(QUAD_DELAY_PROBE_S);
    let mut pcm: Vec<f32> = Vec::new();
    let rate = match decode_quad_window(input, stream_index, 0.0, probe, &mut |chunk| pcm.extend_from_slice(chunk)) {
        Ok(r) => r as f32,
        Err(e) => { tracing::warn!("kandao audio: channel-delay probe failed ({e}) — no re-alignment"); return [0; 4]; }
    };
    quad_channel_delays_from_pcm(&pcm, rate)
}

/// [`measure_quad_channel_delays`] on decoded packed 4-channel f32 samples.
pub fn quad_channel_delays_from_pcm(pcm: &[f32], rate: f32) -> QuadDelays {
    const WIN: usize = 2048;
    const MAX_LAG: i32 = 40;
    let n = pcm.len() / 4;
    if n < WIN * 4 { return [0; 4]; }
    // Band-pass 300–4000 Hz: two first-order high-passes and two low-passes.
    let mut ch: Vec<Vec<f32>> = (0..4).map(|k| pcm.iter().skip(k).step_by(4).copied().collect()).collect();
    let bh = 1.0 / (1.0 + std::f32::consts::TAU * 300.0 / rate);
    let al = 1.0 - (-std::f32::consts::TAU * 4000.0 / rate).exp();
    for v in ch.iter_mut() {
        for _ in 0..2 {
            let (mut xp, mut yp) = (0.0f32, 0.0f32);
            for x in v.iter_mut() { let y = bh * (yp + *x - xp); xp = *x; yp = y; *x = y; }
            let mut lp = 0.0f32;
            for x in v.iter_mut() { lp += al * (*x - lp); *x = lp; }
        }
    }
    let hann: Vec<f32> = (0..WIN).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / WIN as f32).cos()).collect();
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut lags: [Vec<i32>; 6] = Default::default();
    let mut w = [[0f32; WIN]; 4];
    let mut start = 0;
    while start + WIN <= n {
        for k in 0..4 { for i in 0..WIN { w[k][i] = ch[k][start + i] * hann[i]; } }
        let energy: [f32; 4] = std::array::from_fn(|k| w[k].iter().map(|v| v * v).sum::<f32>());
        let mut cur = [0i32; 6];
        let mut ok = energy.iter().all(|e| *e > 1e-12);
        if ok {
            for (p, (i, j)) in PAIRS.iter().enumerate() {
                let (mut best, mut best_lag) = (f32::MIN, 0i32);
                for lag in -MAX_LAG..=MAX_LAG {
                    // c[lag] = Σ_t w_i[t]·w_j[t − lag]: if channel i is a
                    // delayed copy of j (i LATER by τ), the peak sits at lag = +τ.
                    let mut acc = 0f32;
                    let (lo, hi) = (lag.max(0) as usize, (WIN as i32 + lag.min(0)) as usize);
                    for t in lo..hi { acc += w[*i][t] * w[*j][(t as i32 - lag) as usize]; }
                    if acc > best { best = acc; best_lag = lag; }
                }
                if best / (energy[*i] * energy[*j]).sqrt() < 0.5 { ok = false; break; }
                cur[p] = best_lag;
            }
        }
        if ok { for p in 0..6 { lags[p].push(cur[p]); } }
        start += WIN;
    }
    let count = lags[0].len();
    if count < 40 {
        tracing::info!(windows = count, "kandao audio: too few coherent windows to measure channel delays — none applied");
        return [0; 4];
    }
    let mut mid = [0f32; 6];
    for p in 0..6 {
        let mut v = lags[p].clone();
        v.sort_unstable();
        let pct = |q: f32| v[((v.len() - 1) as f32 * q).round() as usize] as f32;
        mid[p] = 0.5 * (pct(0.03) + pct(0.97));
    }
    // t_k = ¼ Σ_j lag(k, j), lag(k, j) = −lag(j, k)
    let mut t = [0f32; 4];
    for (p, (i, j)) in PAIRS.iter().enumerate() { t[*i] += 0.25 * mid[p]; t[*j] -= 0.25 * mid[p]; }
    let residual = PAIRS.iter().enumerate().map(|(p, (i, j))| ((t[*i] - t[*j]) - mid[p]).abs()).fold(0f32, f32::max);
    let t_max = t.iter().cloned().fold(f32::MIN, f32::max);
    let delays: QuadDelays = std::array::from_fn(|k| (t_max - t[k]).round().max(0.0) as usize);
    let largest = *delays.iter().max().unwrap();
    if residual > 3.5 || largest < 3 || largest > MAX_LAG as usize {
        tracing::info!(windows = count, ?delays, residual, "kandao audio: channel delays not applied (inconsistent or negligible)");
        return [0; 4];
    }
    tracing::info!(windows = count, ?delays, residual, "kandao audio: re-aligning capsule channels");
    delays
}

/// Decode the `[offset_s, offset_s + dur_s)` window of 4-channel audio
/// stream `stream_index` as packed f32 and hand each decoded chunk to
/// `f`. Returns the sample rate.
fn decode_quad_window(input: &Path, stream_index: usize, offset_s: f64, dur_s: Option<f64>, f: &mut dyn FnMut(&[f32])) -> Result<u32> {
    use ffmpeg::format::sample::{Sample, Type};
    crate::decode::init();
    let mut ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open input {input:?}: {e}")))?;
    if offset_s > 0.001 {
        let ts = (offset_s * 1_000_000.0) as i64;
        let _ = ictx.seek(ts, ..ts);
    }
    let (src_params, src_tb) = {
        let st = ictx.stream(stream_index)
            .ok_or_else(|| Error::Ffmpeg(format!("audio stream {stream_index} vanished")))?;
        (st.parameters(), st.time_base())
    };
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(src_params)
        .map_err(|e| Error::Ffmpeg(format!("decoder params: {e}")))?
        .decoder().audio()
        .map_err(|e| Error::Ffmpeg(format!("audio decoder: {e}")))?;
    let rate = decoder.rate();
    let src_layout = normalized_layout(decoder.channel_layout());
    let in_ch = src_layout.channels().max(1) as usize;
    if in_ch != 4 {
        return Err(Error::Ffmpeg(format!("the 4-channel pre-pass needs 4 input channels, stream has {in_ch}")));
    }
    let mut to_f32 = ffmpeg::software::resampling::Context::get(
        decoder.format(), src_layout, rate,
        Sample::F32(Type::Packed), src_layout, rate,
    ).map_err(|e| Error::Ffmpeg(format!("resampler: {e}")))?;
    let tb_s = src_tb.numerator() as f64 / src_tb.denominator().max(1) as f64;
    let win_end = dur_s.map(|d| offset_s + d);
    let mut decoded = ffmpeg::frame::Audio::empty();
    let mut f32_frame = ffmpeg::frame::Audio::empty();
    for (stream, pkt) in ictx.packets() {
        if stream.index() != stream_index { continue; }
        let t = pkt.pts().or(pkt.dts()).unwrap_or(0) as f64 * tb_s;
        if t < offset_s { continue; }
        if let Some(end) = win_end { if t >= end { break; } }
        if decoder.send_packet(&pkt).is_err() { continue; }
        while decoder.receive_frame(&mut decoded).is_ok() {
            if { decoded.set_channel_layout(src_layout); to_f32.run(&decoded, &mut f32_frame) }.is_ok() && f32_frame.samples() > 0 {
                let n = f32_frame.samples();
                f(bytemuck::cast_slice(&f32_frame.data(0)[..n * in_ch * 4]));
            }
        }
    }
    Ok(rate)
}

/// Largest |sample| of the window `[offset_s, offset_s + dur_s)` of stream
/// `stream_index` after the mix (same window rule as
/// [`transcode_audio_window`], decode only).
fn mixed_window_peak(input: &Path, stream_index: usize, offset_s: f64, dur_s: Option<f64>, mix: Mix<'_>) -> Result<f32> {
    let mut peak = 0f32;
    let mut state: Option<MixState> = None;
    let mut tmp: Vec<f32> = Vec::with_capacity(4);
    // The encoder needs the rate, known only once the stream is open: build it on the first chunk.
    let mut pending: Vec<f32> = Vec::new();
    let rate = decode_quad_window(input, stream_index, offset_s, dur_s, &mut |chunk| pending.extend_from_slice(chunk))?;
    let st = state.get_or_insert_with(|| MixState::new(mix, rate));
    for s4 in pending.chunks_exact(4) {
        tmp.clear();
        st.apply([s4[0], s4[1], s4[2], s4[3]], &mut tmp);
        for v in &tmp { let v = v.abs(); if v > peak { peak = v; } }
    }
    Ok(peak)
}

/// Encode the QooCam's four capsules to a first-order AmbiX 4-channel AAC
/// `.m4a` through [`QuadEncoder`] (peak-normalised, see
/// [`kandao_window_gain`]) — the export window, starting at its own 0 like
/// [`quad_to_stereo_aac`]. The mux then tags it SA3D.
pub fn quad_to_foa_aac(input: &Path, output_m4a: &Path, offset_s: f64, dur_s: Option<f64>, bit_rate: usize) -> Result<u64> {
    let info = probe_foa_aac(input)?
        .ok_or_else(|| Error::Ffmpeg(format!("no 4-channel AAC track in {input:?}")))?;
    let t0 = std::time::Instant::now();
    let (g, cal) = kandao_window_prep(input, info.stream_index, offset_s, dur_s);
    let n = transcode_audio_window(
        input, info.stream_index, output_m4a, offset_s, dur_s,
        Some(Mix::QuadFoa { gain: g, cal }), AudioOut::Aac4 { bit_rate },
    )?;
    tracing::info!(m4a = %output_m4a.display(), samples = n, gain = g, ?cal, took_ms = t0.elapsed().as_millis() as u64,
        "QooCam 4-channel audio: encoded first-order AmbiX AAC");
    Ok(n)
}

/// The same encode as [`quad_to_foa_aac`] to a 24-bit 4-channel WAV with
/// the ambisonic layout (what the APAC helper consumes).
pub fn quad_to_foa_wav(input: &Path, output_wav: &Path, offset_s: f64, dur_s: Option<f64>) -> Result<u64> {
    let info = probe_foa_aac(input)?
        .ok_or_else(|| Error::Ffmpeg(format!("no 4-channel AAC track in {input:?}")))?;
    let (g, cal) = kandao_window_prep(input, info.stream_index, offset_s, dur_s);
    transcode_audio_window(
        input, info.stream_index, output_wav, offset_s, dur_s,
        Some(Mix::QuadFoa { gain: g, cal }), AudioOut::Pcm24 { channels: 4 },
    )
}

/// Decode the Kandao QooCam's 4-channel microphone track to a stereo AAC
/// `.m4a` through [`QuadEncoder::stereo`] — the window
/// `[offset_s, offset_s + dur_s)` (whole track when `None`), starting at
/// its own 0 like [`foa_to_stereo_aac`]. Returns the samples written.
pub fn quad_to_stereo_aac(input: &Path, output_m4a: &Path, offset_s: f64, dur_s: Option<f64>, bit_rate: usize) -> Result<u64> {
    let info = probe_foa_aac(input)?
        .ok_or_else(|| Error::Ffmpeg(format!("no 4-channel AAC track in {input:?}")))?;
    let t0 = std::time::Instant::now();
    // The same gain the ambisonic export of this window would get, so the
    // formats agree in level.
    let (g, cal) = kandao_window_prep(input, info.stream_index, offset_s, dur_s);
    let n = transcode_audio_window(
        input, info.stream_index, output_m4a, offset_s, dur_s,
        Some(Mix::QuadStereo { gain: g, cal }), AudioOut::AacStereo { bit_rate },
    )?;
    tracing::info!(m4a = %output_m4a.display(), samples = n, sample_rate = info.sample_rate, gain = g,
        window_start_s = offset_s, window_s = dur_s.unwrap_or(f64::NAN), took_ms = t0.elapsed().as_millis() as u64,
        "QooCam 4-channel audio: decoded to stereo AAC");
    Ok(n)
}

/// Pick the most useful audio stream in `path` — first non-attached-pic
/// audio track, regardless of channel count or codec. Returns the
/// stream index, channel count, and sample rate. `None` if the file
/// has no audio.
pub fn probe_any_audio(input: &Path) -> Result<Option<(usize, u32, u32)>> {
    crate::decode::init();
    // concat-aware so a multi-segment `.ffconcat` playlist probes the
    // first audio of the chain (preview player passes the playlist).
    let ictx = open_input_concat_aware(input)
        .map_err(|e| Error::Ffmpeg(format!("open {input:?}: {e}")))?;
    for stream in ictx.streams() {
        let params = stream.parameters();
        if params.medium() != ffmpeg::media::Type::Audio { continue; }
        let (channels, sample_rate) = unsafe {
            let raw = params.as_ptr();
            (
                (*raw).ch_layout.nb_channels as u32,
                (*raw).sample_rate as u32,
            )
        };
        return Ok(Some((stream.index(), channels, sample_rate)));
    }
    Ok(None)
}

/// Mux a video-only file with the audio track from a separate source.
/// Both video and audio are copied with no re-encode (stream copy), so
/// the operation is fast and lossless.
///
/// Use this as a post-export step: write the video-only output via
/// `H265Encoder`, then call this to fold the source clip's audio (e.g.
/// the stereo AAC track on DJI OSV) onto the final container.
///
/// Returns the number of audio packets copied. If `audio_src` has no
/// audio track the function returns `Ok(0)` and the output still
/// contains the video — caller can decide whether to keep using it.
/// Open an input, transparently handling an ffconcat playlist (the
/// multi-segment `.360` audio source). The concat demuxer needs
/// `safe=0` to accept the absolute segment paths; a normal file opens
/// plainly. Detected by the `.ffconcat` extension.
pub fn open_input_concat_aware(
    path: &Path,
) -> std::result::Result<ffmpeg::format::context::Input, ffmpeg::Error> {
    if path.extension().and_then(|e| e.to_str()) == Some("ffconcat") {
        let mut d = ffmpeg::Dictionary::new();
        d.set("safe", "0");
        ffmpeg::format::input_with_dictionary(&path, d)
    } else {
        ffmpeg::format::input(&path)
    }
}

/// Write an ffconcat playlist for a multi-segment chain to a temp file
/// so the segments present as ONE continuous audio timeline via the
/// concat demuxer. Returns `None` for a single segment (open it
/// directly). `tag` disambiguates concurrent playlists (export vs
/// preview). Caller deletes the returned temp when done.
pub fn build_segment_playlist(segments: &[std::path::PathBuf], tag: &str) -> Option<std::path::PathBuf> {
    if segments.len() <= 1 { return None; }
    let mut pl = String::from("ffconcat version 1.0\n");
    for s in segments {
        // ffconcat single-quote escaping: ' → '\''
        let esc = s.display().to_string().replace('\'', "'\\''");
        pl.push_str(&format!("file '{esc}'\n"));
    }
    let path = std::env::temp_dir().join(format!("vr180_segs_{tag}.ffconcat"));
    std::fs::write(&path, pl).ok()?;
    Some(path)
}

pub fn mux_video_with_passthrough_audio(
    audio_src: &Path,
    video_src: &Path,
    final_out: &Path,
    audio_offset_s: f64,
    max_duration_s: Option<f64>,
) -> Result<u64> {
    mux_video_with_audio_stream(audio_src, video_src, final_out, audio_offset_s, max_duration_s, None)
}

/// Like [`mux_video_with_passthrough_audio`] but lets the caller pick a
/// SPECIFIC audio stream index (e.g. the 4-channel ambisonic track of a
/// `.360`, which isn't the first audio). `None` = first audio stream.
pub fn mux_video_with_audio_stream(
    audio_src: &Path,
    video_src: &Path,
    final_out: &Path,
    audio_offset_s: f64,
    max_duration_s: Option<f64>,
    audio_stream_idx: Option<usize>,
) -> Result<u64> {
    // `audio_offset_s` / `max_duration_s` implement TRIMMED exports: the
    // video temp starts at source-time `trim_in` but its own timeline at 0,
    // so the source audio must be cut at `audio_offset_s` (= trim_in),
    // rebased to 0, and stopped after the video's duration. Without this a
    // trimmed export carried the FULL source audio at original timestamps —
    // wrong length and out of sync by trim_in seconds.
    crate::decode::init();

    let mut a_in = open_input_concat_aware(audio_src)
        .map_err(|e| Error::Ffmpeg(format!("open audio src {audio_src:?}: {e}")))?;
    let mut v_in = ffmpeg::format::input(&video_src)
        .map_err(|e| Error::Ffmpeg(format!("open video src {video_src:?}: {e}")))?;

    // Locate the audio stream: the caller-specified index (verified to
    // be audio), else the first audio stream.
    let a_idx = match audio_stream_idx {
        Some(i) if a_in.stream(i)
            .map(|s| s.parameters().medium() == ffmpeg::media::Type::Audio)
            .unwrap_or(false) => Some(i),
        _ => a_in.streams()
            .filter(|s| s.parameters().medium() == ffmpeg::media::Type::Audio)
            .map(|s| s.index())
            .next(),
    };
    let a_idx = match a_idx {
        Some(i) => i,
        None => {
            tracing::info!(
                src = %audio_src.display(),
                "no audio track in source — output will be video-only"
            );
            return Ok(0);
        }
    };

    // Locate the first video stream in the video source.
    let v_idx = v_in.streams()
        .filter(|s| s.parameters().medium() == ffmpeg::media::Type::Video)
        .map(|s| s.index())
        .next()
        .ok_or_else(|| Error::Ffmpeg(format!("no video stream in {video_src:?}")))?;

    let mut octx = ffmpeg::format::output(&final_out)
        .map_err(|e| Error::Ffmpeg(format!("open output {final_out:?}: {e}")))?;

    // Build output streams (params copy) for both video and audio.
    // We DO NOT clear `codec_tag` for the video stream — the encoder
    // wrote `hvc1` (QuickTime / Vision Pro require it for HEVC) and
    // zeroing it makes the muxer fall back to `hev1`, which those
    // players refuse to decode (the file looks "audio only" in them).
    // For audio we still clear so the muxer picks the right MOV/MP4
    // codec_tag for AAC.
    let v_src_params = v_in.stream(v_idx).unwrap().parameters();
    let v_src_tb = v_in.stream(v_idx).unwrap().time_base();
    let out_v_idx = {
        let mut s = octx.add_stream(ffmpeg::codec::Id::None)
            .map_err(|e| Error::Ffmpeg(format!("add video stream: {e}")))?;
        s.set_parameters(v_src_params);
        s.set_time_base(v_src_tb);
        s.index()
    };

    let a_src_params = a_in.stream(a_idx).unwrap().parameters();
    let a_src_tb = a_in.stream(a_idx).unwrap().time_base();
    let out_a_idx = {
        let mut s = octx.add_stream(ffmpeg::codec::Id::None)
            .map_err(|e| Error::Ffmpeg(format!("add audio stream: {e}")))?;
        s.set_parameters(a_src_params);
        s.set_time_base(a_src_tb);
        unsafe {
            let raw = s.parameters().as_mut_ptr();
            (*raw).codec_tag = 0;
            // GoPro's 4-ch ambisonic PCM carries an "ambisonic 1" channel
            // layout the MP4/MOV muxer rejects ("unsupported channel
            // layout" → write_trailer fails). Replace it with the plain
            // N-channel native layout so the container accepts it; the
            // SA3D atom (injected after) carries the ambisonic semantics.
            let nch = (*raw).ch_layout.nb_channels;
            if nch >= 3 {
                ffmpeg::ffi::av_channel_layout_uninit(&mut (*raw).ch_layout);
                ffmpeg::ffi::av_channel_layout_default(&mut (*raw).ch_layout, nch);
            }
        }
        s.index()
    };

    octx.write_header()
        .map_err(|e| Error::Ffmpeg(format!("write_header: {e}")))?;

    // Interleave video and audio packets by ascending PTS so the MP4
    // muxer doesn't have to buffer the whole stream of one type before
    // it sees the other. We use peekable iterators and pick whichever
    // packet is older next.
    let v_out_tb = octx.stream(out_v_idx).unwrap().time_base();
    let a_out_tb = octx.stream(out_a_idx).unwrap().time_base();
    let mut v_iter = v_in.packets().filter(|(s, _)| s.index() == v_idx).peekable();
    let mut a_iter = a_in.packets().filter(|(s, _)| s.index() == a_idx).peekable();
    let mut v_packets = 0u64;
    let mut a_packets = 0u64;

    // Compute seconds for a packet's dts/pts in its source time base.
    fn pkt_seconds(p: &ffmpeg::Packet, tb: ffmpeg::Rational) -> f64 {
        let t = p.dts().or_else(|| p.pts()).unwrap_or(0);
        t as f64 * tb.numerator() as f64 / tb.denominator() as f64
    }

    // Audio cut window in source-audio time, and the pts rebase (ticks)
    // that moves the first kept packet to ≈0 on the output timeline.
    let a_tb_per_s = a_src_tb.denominator() as f64 / a_src_tb.numerator() as f64;
    let a_off_ticks = (audio_offset_s * a_tb_per_s).round() as i64;
    let mut a_done = false;

    loop {
        // Skip leading audio entirely before trim_in (packet granularity,
        // ~21 ms for AAC — inaudible cut slop).
        while let Some((_, p)) = a_iter.peek() {
            if pkt_seconds(p, a_src_tb) < audio_offset_s - 1e-6 {
                a_iter.next();
            } else {
                break;
            }
        }
        // Stop audio once the video's duration is covered.
        if !a_done {
            if let (Some(max_d), Some((_, p))) = (max_duration_s, a_iter.peek()) {
                if pkt_seconds(p, a_src_tb) - audio_offset_s >= max_d {
                    a_done = true;
                }
            }
        }
        let v_t = v_iter.peek().map(|(_, p)| pkt_seconds(p, v_src_tb));
        let a_t = if a_done { None } else {
            // Compare on the OUTPUT timeline (audio rebased by trim_in).
            a_iter.peek().map(|(_, p)| pkt_seconds(p, a_src_tb) - audio_offset_s)
        };
        let take_video = match (v_t, a_t) {
            (Some(vt), Some(at)) => vt <= at,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_video {
            let (_, mut packet) = v_iter.next().unwrap();
            packet.set_stream(out_v_idx);
            packet.set_position(-1);
            packet.rescale_ts(v_src_tb, v_out_tb);
            packet.write_interleaved(&mut octx)
                .map_err(|e| Error::Ffmpeg(format!("write video packet: {e}")))?;
            v_packets += 1;
        } else {
            let (_, mut packet) = a_iter.next().unwrap();
            // Rebase to the output timeline (clamp the first packet's
            // sub-packet remainder to 0 rather than going negative).
            if let Some(pts) = packet.pts() {
                packet.set_pts(Some((pts - a_off_ticks).max(0)));
            }
            if let Some(dts) = packet.dts() {
                packet.set_dts(Some((dts - a_off_ticks).max(0)));
            }
            packet.set_stream(out_a_idx);
            packet.set_position(-1);
            packet.rescale_ts(a_src_tb, a_out_tb);
            packet.write_interleaved(&mut octx)
                .map_err(|e| Error::Ffmpeg(format!("write audio packet: {e}")))?;
            a_packets += 1;
        }
    }

    octx.write_trailer()
        .map_err(|e| Error::Ffmpeg(format!("write_trailer: {e}")))?;

    tracing::info!(
        final = %final_out.display(),
        video_packets = v_packets,
        audio_packets = a_packets,
        "mux complete — video + audio passthrough"
    );
    Ok(a_packets)
}

#[cfg(test)]
mod foa_tests {
    use super::*;

    #[test]
    fn frontal_source_is_centred_and_left_source_favours_left() {
        // SN3D plane wave from the front: W = 1, X = 1, Y = Z = 0.
        let [l, r] = foa_to_stereo_sample([1.0, 0.0, 0.0, 1.0]);
        assert!((l - r).abs() < 1e-6 && (l - 0.853_553).abs() < 1e-4, "{l} {r}");
        // From the left (+Y): the left cardioid is on axis, the right one at 90°.
        let [l, r] = foa_to_stereo_sample([1.0, 1.0, 0.0, 0.0]);
        assert!(l > r && (l - 0.853_553).abs() < 1e-4 && (r - 0.146_447).abs() < 1e-4, "{l} {r}");
        // From straight above: both cardioids see only W.
        let [l, r] = foa_to_stereo_sample([1.0, 0.0, 1.0, 0.0]);
        assert!((l - 0.5).abs() < 1e-6 && (r - 0.5).abs() < 1e-6);
        // From behind: the pair is 135° off axis — attenuated, still centred.
        let [l, r] = foa_to_stereo_sample([1.0, 0.0, 0.0, -1.0]);
        assert!((l - r).abs() < 1e-6 && l < 0.2, "{l} {r}");
    }

    /// A 1 kHz plane wave crossing the capsule cross from azimuth `az`
    /// (0 = front, +90 = left) through [`QuadEncoder`] (built-in ×3 taken
    /// out): per-channel rms, the in-phase correlation of X and Y with W,
    /// and the stereo pair's rms — over the second half of 0.5 s.
    fn quad_plane_wave(az_deg: f32) -> ([f32; 4], [f32; 2], [f32; 2]) {
        quad_plane_wave_misaligned(az_deg, 0, [0; 4])
    }

    /// [`quad_plane_wave`] with channel 0 recorded `late0` samples late and
    /// the encoder told to re-align by `delays`.
    fn quad_plane_wave_misaligned(az_deg: f32, late0: usize, delays: QuadDelays) -> ([f32; 4], [f32; 2], [f32; 2]) {
        let rate = 48_000u32;
        let d = KANDAO_CAPSULE_SPACING_M / 2.0;
        let pos = |ch: usize| -> (f32, f32) {
            if ch == KANDAO_CAPSULE_FRONT { (d, 0.0) } else if ch == KANDAO_CAPSULE_LEFT { (0.0, d) }
            else if ch == KANDAO_CAPSULE_BACK { (-d, 0.0) } else { (0.0, -d) }
        };
        let (ux, uy) = (az_deg.to_radians().cos(), az_deg.to_radians().sin());
        let mut enc = QuadEncoder::new(rate, 1.0 / KANDAO_QUAD_GAIN).with_delays(delays);
        let (mut acc, mut lr, mut wx, mut wy, mut n) = ([0f32; 4], [0f32; 2], 0f32, 0f32, 0usize);
        for i in 0..rate as usize / 2 {
            let t = i as f32 / rate as f32;
            let mut c = [0f32; 4];
            for (ch, v) in c.iter_mut().enumerate() {
                let (px, py) = pos(ch);
                let tc = if ch == 0 { t - late0 as f32 / rate as f32 } else { t };
                // the capsule along the arrival direction hears the wave first
                *v = (std::f32::consts::TAU * 1000.0 * (tc + (px * ux + py * uy) / SPEED_OF_SOUND_M_S)).sin();
            }
            let o = enc.foa(c);
            let s = foa_to_stereo_sample(o);
            if i >= rate as usize / 4 {
                for k in 0..4 { acc[k] += o[k] * o[k]; }
                lr[0] += s[0] * s[0]; lr[1] += s[1] * s[1];
                wx += o[0] * o[3]; wy += o[0] * o[1]; n += 1;
            }
        }
        let rms = |v: f32| (v / n as f32).sqrt();
        let r = [rms(acc[0]), rms(acc[1]), rms(acc[2]), rms(acc[3])];
        let corr = |p: f32, a: f32, b: f32| if a * b > 0.0 { p / n as f32 / (a * b) } else { 0.0 };
        (r, [corr(wx, r[0], r[3]), corr(wy, r[0], r[1])], [rms(lr[0]), rms(lr[1])])
    }

    #[test]
    fn kandao_quad_encoder_places_a_plane_wave() {
        // Front: X at the SN3D level and in phase with W, Y silent, stereo centred.
        let ([w, y, z, x], [cx, _], [l, r]) = quad_plane_wave(0.0);
        assert!(z == 0.0 && y < 0.02 * w, "front: w {w} y {y}");
        assert!((x / w - 1.0).abs() < 0.25 && cx > 0.95, "front: x/w {} corr {cx}", x / w);
        assert!((l - r).abs() < 0.02 * l, "front stereo {l} {r}");
        // Back: X in anti-phase, still centred.
        let (_, [cx, _], [l, r]) = quad_plane_wave(180.0);
        assert!(cx < -0.95 && (l - r).abs() < 0.02 * l, "back: corr {cx} stereo {l} {r}");
        // Left: Y in phase with W at the SN3D level, X silent, L well above R;
        // the right mirrors it.
        let ([w, y, _, x], [_, cy], [l, r]) = quad_plane_wave(90.0);
        assert!(x < 0.02 * w && (y / w - 1.0).abs() < 0.25 && cy > 0.95, "left: x {x} y/w {} corr {cy}", y / w);
        assert!(l > 2.0 * r, "left stereo {l} {r}");
        let (_, _, [l2, r2]) = quad_plane_wave(-90.0);
        assert!((l2 - r).abs() < 0.02 * r && (r2 - l).abs() < 0.02 * l, "right stereo {l2} {r2} vs {l} {r}");
        // Channel 0 recorded 14 samples late (the second unit): the front
        // wave's X goes out of phase with W; re-aligning the other three by
        // 14 restores it.
        let (_, [cx_bad, _], _) = quad_plane_wave_misaligned(0.0, 14, [0; 4]);
        let (_, [cx_fixed, _], _) = quad_plane_wave_misaligned(0.0, 14, [0, 14, 14, 14]);
        assert!(cx_bad < 0.5 && cx_fixed > 0.95, "misaligned corr {cx_bad}, re-aligned {cx_fixed}");
        // The measurement itself: broadband noise with channel 0 recorded
        // 14 samples late (plus a little independent noise per channel)
        // measures as "delay the other three by 14"; an aligned track → none.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0 };
        let len = 48_000 * 3;
        let src: Vec<f32> = (0..len + 14).map(|_| rnd()).collect();
        let mut late = Vec::with_capacity(len * 4);
        let mut aligned = Vec::with_capacity(len * 4);
        for i in 0..len {
            late.extend_from_slice(&[src[i] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd()]);
            aligned.extend_from_slice(&[src[i + 14] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd(), src[i + 14] + 0.1 * rnd()]);
        }
        let d = quad_channel_delays_from_pcm(&late, 48_000.0);
        assert!(d[0] == 0 && (1..4).all(|k| (d[k] as i32 - 14).abs() <= 1), "measured delays {d:?}");
        assert_eq!(quad_channel_delays_from_pcm(&aligned, 48_000.0), [0; 4]);
        // Level trims: channel 1 recorded 8 dB low below 500 Hz and 3 dB low
        // above (the second unit's pattern) on a two-band noise field;
        // the trims bring it back to the mean within 1 dB, and a matched
        // track gets none.
        let mut lo = 0f32; let mut hi_prev = 0f32;
        let (a_lo, a_hi) = (1.0 - (-std::f32::consts::TAU * 300.0 / 48_000.0).exp(), 1.0 / (1.0 + std::f32::consts::TAU * 1000.0 / 48_000.0));
        let mut mismatched = Vec::with_capacity(len * 4);
        let mut matched = Vec::with_capacity(len * 4);
        let (g_lo, g_hi) = (10f32.powf(-8.0 / 20.0), 10f32.powf(-3.0 / 20.0));
        let mut xp = 0f32;
        for _ in 0..len {
            let x = rnd();
            lo += a_lo * (x - lo);                         // low-band noise (< ~300 Hz)
            let hi = a_hi * (hi_prev + x - xp); xp = x; hi_prev = hi;   // high-band noise (> ~1 kHz)
            let base = lo + 0.3 * hi;
            matched.extend_from_slice(&[base, base, base, base]);
            mismatched.extend_from_slice(&[base, g_lo * lo + g_hi * 0.3 * hi, base, base]);
        }
        assert_eq!(quad_trims_from_pcm(&matched, 48_000.0), [[1.0; 2]; 4]);
        let t = quad_trims_from_pcm(&mismatched, 48_000.0);
        let db = |g: f32| 20.0 * g.log10();
        // expected: the 1/3 pair balanced about its mean — channel 1 up by
        // half the gap (4 dB / 1.5 dB), channel 3 down by the same; the 0/2
        // pair untouched.
        assert!((db(t[1][0]) - 4.0).abs() < 1.0 && (db(t[3][0]) + 4.0).abs() < 1.0, "low trims ch1 {} ch3 {} dB", db(t[1][0]), db(t[3][0]));
        assert!((db(t[1][1]) - 1.5).abs() < 1.0 && (db(t[3][1]) + 1.5).abs() < 1.0, "mid trims ch1 {} ch3 {} dB", db(t[1][1]), db(t[3][1]));
        assert!(db(t[0][0]).abs() < 0.5 && db(t[2][1]).abs() < 0.5, "pair 0/2 should stay at unity: {:?} {:?}", t[0], t[2]);
        // A static capsule offset (mismatch, wind) does not reach the dipoles.
        let mut enc = QuadEncoder::new(48_000, 1.0);
        let mut last = [0f32; 4];
        for _ in 0..48_000 { last = enc.foa([1.0, 0.0, 0.0, 0.0]); }
        assert!(last[0].abs() < 1e-3 && last[3].abs() < 1e-3 && last[1].abs() < 1e-3, "dc leak {last:?}");
    }
}
