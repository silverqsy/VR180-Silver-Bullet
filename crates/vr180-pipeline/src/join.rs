//! Several exported files of one shape as one file.
//!
//! The clips of a list exported as ONE file are each exported the usual
//! way, to a file of its own, and put together here: their video streams
//! one after the other, packet for packet. Nothing is encoded again, so a
//! cut costs no quality, and it falls on a keyframe (each part's first).
//! Their sound is decoded, each part's cut to the length of its own picture
//! to the sample (a part's track runs up to a codec frame past its last
//! picture, which would otherwise add up from cut to cut), faded over a few
//! milliseconds at its two ends, and encoded once for the whole length: no
//! codec delay sits at a cut and the sound stays on the picture however
//! many parts there are. The joined track is stereo at 48 kHz whatever
//! the parts hold, and a part without sound is silent for its length; or,
//! where every part holds four-channel first-order ambisonics, that sound
//! field as one track ([`JoinSound`]).
//!
//! The pictures have to be of one shape: the same codec, frame size, frame
//! rate and encoder parameters (they are, when one export run wrote them
//! with one set of options). A part that differs is named in the error.
//!
//! The projection metadata of the parts' own files is not carried by a
//! packet copy: the caller states it in the joined file
//! ([`crate::fisheye_export::state_export_metadata`]).
//!
//! Lens frames are the exception: each states its clip's own lens, and a
//! file described that way has one lens. Such parts become an Apple
//! Immersive Video file instead ([`join_immersive`]), which holds a camera
//! for every part and says, frame by frame, which one a frame is shown
//! through. The picture and the sound are put together here as for any
//! other join, and a helper writes them again, packet for packet, with that
//! description. macOS 26 or later.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use crate::fisheye_export::ExportProgress;
use crate::{Error, Result};

/// Bit rate of the joined file's stereo track.
const AUDIO_BIT_RATE: usize = 256_000;
/// And of its four-channel one: the same for each channel.
const AMBISONIC_BIT_RATE: usize = 512_000;
/// Bit rate the Apple Positional Audio encoder is asked for, as a single
/// export asks.
#[cfg(target_os = "macos")]
const APAC_BIT_RATE: u32 = 256_000;

/// The sound of a joined file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JoinSound {
    /// One stereo track, from parts with any sound or none.
    #[default]
    Stereo,
    /// The parts' four-channel first-order ambisonics (AmbiX, as an export
    /// with that track writes it) as one four-channel track, tagged as
    /// such. Every part has to hold a four-channel track.
    Ambisonic,
    /// The same sound field as Apple Positional Audio. macOS; where it
    /// cannot be encoded the file gets the ambisonic track.
    Apac,
}

impl JoinSound {
    /// Channels of the sound as it is put together.
    fn channels(self) -> usize {
        match self {
            JoinSound::Stereo => 2,
            JoinSound::Ambisonic | JoinSound::Apac => 4,
        }
    }
}
/// Its sample rate: every camera here records sound at it.
const SAMPLE_RATE: u32 = 48_000;
/// The fade at each end of a part's sound, seconds: long enough that a cut
/// between two unrelated waveforms does not click, too short to hear.
const EDGE_FADE_S: f64 = 0.005;

fn err(msg: impl Into<String>) -> Error {
    Error::Ffmpeg(format!("join: {}", msg.into()))
}

fn name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

/// What a part's video stream has to share with the first part's.
#[derive(Debug, Clone, PartialEq)]
struct VideoShape {
    codec: ffmpeg::codec::Id,
    width: u32,
    height: u32,
    /// Frames a second, as the container states it.
    fps: f64,
    /// The encoder's parameter sets (none for an intra codec).
    extradata: Vec<u8>,
}

/// One part, looked at before anything is written.
#[derive(Debug, Clone, PartialEq)]
struct PartInfo {
    shape: VideoShape,
    /// The clock its frames are stamped on.
    time_base: (i32, i32),
    video_index: usize,
    frames: u64,
    has_sound: bool,
    /// One of its tracks holds four channels.
    four_channel_sound: bool,
}

fn describe(part: &Path) -> Result<PartInfo> {
    let ictx = ffmpeg::format::input(&part).map_err(|e| err(format!("{}: {e}", name(part))))?;
    let st = ictx.streams().find(|s| s.parameters().medium() == ffmpeg::media::Type::Video)
        .ok_or_else(|| err(format!("{}: no video stream", name(part))))?;
    let params = st.parameters();
    let (width, height, extradata) = unsafe {
        let raw = params.as_ptr();
        let extra = if (*raw).extradata.is_null() || (*raw).extradata_size <= 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts((*raw).extradata, (*raw).extradata_size as usize).to_vec()
        };
        ((*raw).width as u32, (*raw).height as u32, extra)
    };
    let (tb, rate) = (st.time_base(), st.rate());
    if tb.numerator() <= 0 || tb.denominator() <= 0 {
        return Err(err(format!("{}: no frame clock", name(part))));
    }
    Ok(PartInfo {
        shape: VideoShape {
            codec: params.id(), width, height,
            fps: rate.numerator() as f64 / rate.denominator().max(1) as f64, extradata,
        },
        time_base: (tb.numerator(), tb.denominator()),
        video_index: st.index(),
        frames: st.frames().max(0) as u64,
        has_sound: ictx.streams().any(|s| s.parameters().medium() == ffmpeg::media::Type::Audio),
        four_channel_sound: ictx.streams().any(|s| is_audio_of(&s, 4)),
    })
}

/// An audio stream; with `channels` above zero, one of that many channels.
fn is_audio_of(stream: &ffmpeg::format::stream::Stream, channels: i32) -> bool {
    let params = stream.parameters();
    params.medium() == ffmpeg::media::Type::Audio
        && (channels == 0 || unsafe { (*params.as_ptr()).ch_layout.nb_channels } == channels)
}

/// Why `parts` cannot be joined with the sound `sound`, as far as their
/// headers tell (`None`: they can). A caller may ask before it writes
/// anything.
pub fn join_obstacle(parts: &[PathBuf], sound: JoinSound) -> Result<Option<String>> {
    crate::decode::init();
    let infos: Vec<PartInfo> = parts.iter().map(|p| describe(p)).collect::<Result<_>>()?;
    Ok(obstacle(parts, &infos, sound))
}

fn obstacle(parts: &[PathBuf], infos: &[PartInfo], sound: JoinSound) -> Option<String> {
    let Some((first, rest)) = infos.split_first() else { return Some("no clips".into()) };
    if sound != JoinSound::Stereo {
        if let Some((path, _)) = parts.iter().zip(infos).find(|(_, i)| !i.four_channel_sound) {
            return Some(format!("{} has no four-channel sound", name(path)));
        }
    }
    let a = &first.shape;
    for (path, part) in parts[1..].iter().zip(rest) {
        let b = &part.shape;
        if (a.codec, a.width, a.height) != (b.codec, b.width, b.height) {
            return Some(format!("{} is {:?} {} × {}, {} is {:?} {} × {}",
                name(&parts[0]), a.codec, a.width, a.height, name(path), b.codec, b.width, b.height));
        }
        if (a.fps - b.fps).abs() > FPS_TOLERANCE {
            return Some(format!("{} is {:.2} fps, {} is {:.2} fps", name(&parts[0]), a.fps, name(path), b.fps));
        }
        if a.extradata != b.extradata {
            return Some(format!("{} and {} were encoded with different parameters", name(&parts[0]), name(path)));
        }
    }
    None
}

/// What an export will write, as far as the parts of one joined file have
/// to agree: known before anything is exported, so that a run of clips
/// that cannot become one file is refused at its start and not after its
/// last clip. (The files themselves are looked at again before the join.)
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedPart {
    /// The frame written: both eyes side by side, or the one eye.
    pub frame: (u32, u32),
    pub fps: f32,
    encoder: crate::encode::EncoderBackend,
    /// Bits a sample of an H.265 stream; a ProRes one's profile.
    depth_or_profile: i32,
    described: Described,
}

/// What a file says of its picture to a player. A joined file says it
/// once, for all its parts.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Described {
    /// Nothing: a flat picture.
    Plain,
    HalfEquirect { youtube: bool, apple: bool, one_eye: bool },
    /// A fisheye disc with its lens stated: the angle at its edge, degrees.
    Disc { half_fov_deg: f64, stereo: bool },
    /// A lens frame, two eyes: every part states its own lens, and the
    /// joined file a lens for every part ([`join_immersive`]).
    LensFrame,
}

/// An export that cannot be a part of a joined file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unjoinable {
    /// A spatial video: one stream of two views.
    Spatial,
    /// A lens frame, where no file can be written that holds a lens for
    /// every part ([`immersive_available`]), or one that holds a single
    /// view.
    LensFrame,
}

/// Whether lens frames can be joined on this system: as an Apple Immersive
/// Video file, which macOS 26 or later writes ([`join_immersive`]).
pub fn immersive_available() -> bool {
    crate::helpers::aivu_wrap_available()
}

impl PlannedPart {
    /// The joined file of such parts is an Apple Immersive Video file
    /// ([`join_immersive`] writes it, not [`join_exports`]).
    pub fn immersive(&self) -> bool {
        self.described == Described::LensFrame
    }
}

/// What keeps two planned parts out of one file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mismatch {
    Frame((u32, u32), (u32, u32)),
    FrameRate(f32, f32),
    /// Another codec, bit depth or profile.
    Encoding,
    /// Another projection, or another description of it.
    View,
}

/// Whether the parts of a joined file read their takes' enhanced frames:
/// all of them do, or none does. `found`: for each part, whether its export
/// would read some by itself
/// ([`FisheyeExportConfig::reads_enhanced_frames`](crate::fisheye_export::FisheyeExportConfig::reads_enhanced_frames);
/// `false` for a part that is no raw take). A file made of parts from
/// enhanced frames and parts from the capture would change its look from one
/// clip to the next, and at a size picked for enhanced frames its frame as
/// well; so one part without them leaves them out of every part
/// ([`FisheyeExportConfig::without_enhanced_frames`](crate::fisheye_export::FisheyeExportConfig::without_enhanced_frames)).
pub fn parts_read_enhanced(found: &[bool]) -> bool {
    !found.is_empty() && found.iter().all(|has| *has)
}

/// What the caller knows of a part's source beyond its configuration, from
/// its own probe and, for a raw take's enhanced frames, from a lookup or a
/// decision of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartSource {
    /// The source holds one eye.
    pub one_eye: bool,
    /// One eye of the source's frame, pixels (the side of a `.360` clip's
    /// cross): what a lens frame is sized against.
    pub eye: (u32, u32),
    /// The export reads its take's enhanced frames, and is written at the
    /// size picked for them.
    pub enhanced: bool,
}

/// The part an export of `cfg` would be: the frame it writes
/// ([`crate::fisheye_export::written_eye_dims`]: a size picked for enhanced
/// frames where it reads those, a lens frame's margin or the size it is
/// held to), its rate, its encoding and what it says of its picture.
pub fn planned_part(cfg: &crate::fisheye_export::FisheyeExportConfig, source: PartSource) -> std::result::Result<PlannedPart, Unjoinable> {
    planned_part_where(cfg, source, immersive_available())
}

/// [`planned_part`] on a system that can (`immersive`) or cannot join lens
/// frames.
pub fn planned_part_where(cfg: &crate::fisheye_export::FisheyeExportConfig, source: PartSource, immersive: bool) -> std::result::Result<PlannedPart, Unjoinable> {
    use crate::fisheye_export::{FisheyeExportProjection as P, LensFrameOut, EAC_LENS_FRAME_HALF_FOV_DEG, FISHEYE_OUT_FULL_FOV_DEG};
    let PartSource { one_eye, eye, enhanced } = source;
    if cfg.spatial.is_some() {
        return Err(Unjoinable::Spatial);
    }
    let eac = cfg.source_kind.is_eac();
    let one_eye = one_eye && !eac;
    // As the export decides ([`crate::fisheye_export::state_export_metadata`]).
    let described = match cfg.projection {
        P::Reframe { .. } => Described::Plain,
        // An Apple Immersive Video camera is a pair of eyes.
        P::Fisheye if cfg.lens_frame != LensFrameOut::Off && immersive && !one_eye => Described::LensFrame,
        P::Fisheye if cfg.lens_frame != LensFrameOut::Off => return Err(Unjoinable::LensFrame),
        P::Fisheye if cfg.inject_apmp => Described::Disc {
            half_fov_deg: if eac { EAC_LENS_FRAME_HALF_FOV_DEG } else { FISHEYE_OUT_FULL_FOV_DEG as f64 * 0.5 },
            stereo: !one_eye,
        },
        P::Fisheye => Described::Plain,
        P::HalfEquirect => Described::HalfEquirect {
            youtube: cfg.inject_youtube_vr180 && !cfg.inject_apmp, apple: cfg.inject_apmp, one_eye,
        },
    };
    let (eye_w, eye_h) = crate::fisheye_export::written_eye_dims(cfg, eye, enhanced);
    Ok(PlannedPart {
        frame: (if one_eye { eye_w } else { eye_w * 2 }, eye_h),
        fps: cfg.fps,
        encoder: cfg.encoder,
        depth_or_profile: if cfg.encoder.is_prores() { cfg.prores_profile } else { cfg.bit_depth as i32 },
        described,
    })
}

/// What keeps `b` out of a file that starts with `a` (`None`: nothing).
pub fn mismatch(a: &PlannedPart, b: &PlannedPart) -> Option<Mismatch> {
    if a.frame != b.frame {
        Some(Mismatch::Frame(a.frame, b.frame))
    } else if (a.fps as f64 - b.fps as f64).abs() > FPS_TOLERANCE {
        Some(Mismatch::FrameRate(a.fps, b.fps))
    } else if (a.encoder, a.depth_or_profile) != (b.encoder, b.depth_or_profile) {
        Some(Mismatch::Encoding)
    } else if a.described != b.described {
        Some(Mismatch::View)
    } else {
        None
    }
}

/// Frame rates this close are one rate (a clock that states 29.969 beside
/// one that states 29.970): the frames keep the length their own part gave
/// them.
pub const FPS_TOLERANCE: f64 = 0.01;

/// The video streams of `parts` one after the other in `out`, packets
/// copied. Returns each part's length as written, seconds, and the number
/// of its packets.
fn concat_video(
    parts: &[PathBuf],
    infos: &[PartInfo],
    out: &Path,
    progress: &mut impl FnMut(ExportProgress),
    cancel: &AtomicBool,
) -> Result<Vec<(f64, u64)>> {
    let total_frames: u64 = infos.iter().map(|i| i.frames).sum();
    let started = std::time::Instant::now();
    let mut octx = ffmpeg::format::output(&out).map_err(|e| err(format!("open {}: {e}", name(out))))?;
    let mut out_tb = ffmpeg::Rational::new(1, 1);
    // Where the part being copied starts, on the joined file's clock.
    let mut offset: i64 = 0;
    let mut written: u64 = 0;
    let mut lengths = Vec::with_capacity(parts.len());
    for (k, (part, info)) in parts.iter().zip(infos).enumerate() {
        let mut ictx = ffmpeg::format::input(&part).map_err(|e| err(format!("{}: {e}", name(part))))?;
        let src_tb = ffmpeg::Rational::new(info.time_base.0, info.time_base.1);
        if k == 0 {
            let params = ictx.stream(info.video_index).ok_or_else(|| err("the video stream vanished"))?.parameters();
            let mut s = octx.add_stream(ffmpeg::codec::Id::None).map_err(|e| err(format!("add video stream: {e}")))?;
            // The stream as the encoder left it: its `hvc1` tag above all,
            // without which Apple's players do not take an HEVC track.
            s.set_parameters(params);
            s.set_time_base(src_tb);
            drop(s);
            octx.write_header().map_err(|e| err(format!("write header: {e}")))?;
            out_tb = octx.stream(0).ok_or_else(|| err("no output stream"))?.time_base();
        }
        // The part's length on the joined file's clock: where its last
        // frame ends. And the packets it gives.
        let mut end: i64 = 0;
        let before = written;
        for (stream, mut packet) in ictx.packets() {
            if stream.index() != info.video_index {
                continue;
            }
            if cancel.load(Ordering::SeqCst) {
                drop(octx);
                let _ = std::fs::remove_file(out);
                return Err(err("cancelled"));
            }
            packet.rescale_ts(src_tb, out_tb);
            let duration = packet.duration().max(0);
            if let Some(pts) = packet.pts() {
                end = end.max(pts + duration);
                packet.set_pts(Some(pts + offset));
            }
            if let Some(dts) = packet.dts() {
                packet.set_dts(Some(dts + offset));
            }
            packet.set_stream(0);
            packet.set_position(-1);
            packet.write_interleaved(&mut octx).map_err(|e| err(format!("write a packet of {}: {e}", name(part))))?;
            written += 1;
            if written % 30 == 0 {
                progress(ExportProgress {
                    frame_idx: written, total_frames: total_frames.max(written),
                    fps_avg: written as f32 / started.elapsed().as_secs_f32().max(1e-3),
                });
            }
        }
        if end <= 0 {
            return Err(err(format!("{} holds no frames", name(part))));
        }
        offset += end;
        lengths.push((end as f64 * out_tb.numerator() as f64 / out_tb.denominator().max(1) as f64, written - before));
    }
    octx.write_trailer().map_err(|e| err(format!("write trailer: {e}")))?;
    progress(ExportProgress {
        frame_idx: written, total_frames: total_frames.max(written),
        fps_avg: written as f32 / started.elapsed().as_secs_f32().max(1e-3),
    });
    Ok(lengths)
}

/// One part's sound on its way into the joined track: `want` frames of
/// `channels` samples, faded in over the first `fade` and out over the last
/// ones it holds, with silence after a track that ends before its picture
/// does. The last `fade` frames are held back until the end is known.
struct PartSound<'a, W: std::io::Write> {
    out: &'a mut W,
    channels: usize,
    want: usize,
    fade: usize,
    /// Frames taken so far (written or held).
    taken: usize,
    held: Vec<f32>,
}

impl<'a, W: std::io::Write> PartSound<'a, W> {
    fn new(out: &'a mut W, channels: usize, want: usize) -> Self {
        let fade = ((EDGE_FADE_S * SAMPLE_RATE as f64) as usize).min(want / 2);
        Self { out, channels, want, fade, taken: 0, held: Vec::new() }
    }

    /// Interleaved samples, in order. What lies past the picture is
    /// dropped.
    fn push(&mut self, samples: &[f32]) -> Result<()> {
        let ch = self.channels;
        let room = self.want - self.taken;
        let frames = (samples.len() / ch).min(room);
        for (i, frame) in samples[..frames * ch].chunks_exact(ch).enumerate() {
            let at = self.taken + i;
            let gain = if at < self.fade { at as f32 / self.fade as f32 } else { 1.0 };
            self.held.extend(frame.iter().map(|v| v * gain));
        }
        self.taken += frames;
        let keep = self.fade * ch;
        if self.held.len() > keep {
            let ready = self.held.len() - keep;
            self.out.write_all(bytemuck::cast_slice(&self.held[..ready]))?;
            self.held.drain(..ready);
        }
        Ok(())
    }

    /// The end of the part's sound: the fade out, then silence up to the
    /// picture's length.
    fn finish(mut self) -> Result<()> {
        let ch = self.channels;
        let n = self.held.len() / ch;
        for (i, frame) in self.held.chunks_exact_mut(ch).enumerate() {
            let gain = (n - i) as f32 / (n + 1) as f32;
            frame.iter_mut().for_each(|v| *v *= gain);
        }
        self.out.write_all(bytemuck::cast_slice(&self.held))?;
        let silence = vec![0f32; ch * 4096];
        let mut left = self.want - self.taken;
        while left > 0 {
            let n = left.min(4096);
            self.out.write_all(bytemuck::cast_slice(&silence[..n * ch]))?;
            left -= n;
        }
        Ok(())
    }
}

/// A part's sound at [`SAMPLE_RATE`] into `sound` (nothing when it has no
/// track): mixed to stereo whatever it holds, or, for a `sound` of four
/// channels, its four-channel track as it is, channel for channel. The
/// codec's priming is not in it: the demuxer marks it and the decoder leaves
/// it out, so the first sample is the picture's first moment.
fn decode_into<W: std::io::Write>(part: &Path, sound: &mut PartSound<W>) -> Result<()> {
    use ffmpeg::format::sample::{Sample, Type};
    let ch = sound.channels;
    let mut ictx = ffmpeg::format::input(&part).map_err(|e| err(format!("{}: {e}", name(part))))?;
    let wanted = if ch == 2 { 0 } else { ch as i32 };
    let Some((index, params)) = ictx.streams().find(|s| is_audio_of(s, wanted))
        .map(|s| (s.index(), s.parameters())) else { return Ok(()) };
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(params)
        .and_then(|c| c.decoder().audio())
        .map_err(|e| err(format!("{}: sound decoder: {e}", name(part))))?;
    let layout = crate::audio::normalized_layout(decoder.channel_layout());
    // Stereo is a mix; a sound field keeps its channels as they are.
    let out_layout = if ch == 2 { ffmpeg::ChannelLayout::STEREO } else { layout };
    let mut to_stereo = ffmpeg::software::resampling::Context::get(
        decoder.format(), layout, decoder.rate(),
        Sample::F32(Type::Packed), out_layout, SAMPLE_RATE,
    ).map_err(|e| err(format!("{}: sound format: {e}", name(part))))?;
    let (mut decoded, mut converted) = (ffmpeg::frame::Audio::empty(), ffmpeg::frame::Audio::empty());
    let mut failed: Option<Error> = None;
    let mut take = |decoder: &mut ffmpeg::decoder::Audio, sound: &mut PartSound<W>| {
        while decoder.receive_frame(&mut decoded).is_ok() {
            decoded.set_channel_layout(layout);
            if to_stereo.run(&decoded, &mut converted).is_ok() && converted.samples() > 0 && failed.is_none() {
                let samples: &[f32] = bytemuck::cast_slice(&converted.data(0)[..converted.samples() * ch * 4]);
                failed = sound.push(samples).err();
            }
        }
    };
    for (stream, packet) in ictx.packets() {
        if stream.index() == index && decoder.send_packet(&packet).is_ok() {
            take(&mut decoder, sound);
        }
    }
    let _ = decoder.send_eof();
    take(&mut decoder, sound);
    // What the rate converter still holds.
    while failed.is_none() && to_stereo.flush(&mut converted).is_ok() && converted.samples() > 0 {
        let samples: &[f32] = bytemuck::cast_slice(&converted.data(0)[..converted.samples() * ch * 4]);
        failed = sound.push(samples).err();
    }
    failed.map_or(Ok(()), Err)
}

/// The sound of the joined file as a 32-bit float WAV of `channels`
/// channels: each part's, cut or filled with silence to its picture's
/// length (`lengths`, seconds; the cuts are placed by the running total, so
/// nothing adds up) and faded at both ends.
fn concat_sound(parts: &[PathBuf], lengths: &[f64], channels: usize, wav: &Path) -> Result<()> {
    use std::io::Write;
    let total: u64 = (lengths.iter().sum::<f64>() * SAMPLE_RATE as f64).round() as u64;
    let frame_bytes = (channels * 4) as u32;
    let data_bytes = total.checked_mul(frame_bytes as u64).and_then(|b| u32::try_from(b).ok()).filter(|b| *b < u32::MAX - 36)
        .ok_or_else(|| err("the joined sound is too long for one file"))?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(wav)?);
    // RIFF / WAVE, format 3 (IEEE float), 32 bits.
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_bytes).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?;
    f.write_all(&(channels as u16).to_le_bytes())?;
    f.write_all(&SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&(SAMPLE_RATE * frame_bytes).to_le_bytes())?;
    f.write_all(&(frame_bytes as u16).to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    let (mut at_s, mut at_frames) = (0.0f64, 0u64);
    for (part, length) in parts.iter().zip(lengths) {
        at_s += length;
        let end = ((at_s * SAMPLE_RATE as f64).round() as u64).min(total);
        let mut sound = PartSound::new(&mut f, channels, (end - at_frames) as usize);
        at_frames = end;
        decode_into(part, &mut sound)?;
        sound.finish()?;
    }
    f.flush()?;
    Ok(())
}

/// The four-channel sound `wav` as the joined file's ambisonic track: one
/// four-channel AAC beside the picture `video`, tagged as first-order AmbiX.
fn mux_ambisonic(wav: &Path, video: &Path, output: &Path, seconds: f64) -> Result<()> {
    let sound_file = output.with_extension("joining.sound.mp4");
    let made = crate::audio::four_channel_to_aac(wav, &sound_file, AMBISONIC_BIT_RATE)
        .and_then(|_| crate::audio::mux_video_with_audio_stream(&sound_file, video, output, 0.0, Some(seconds), None));
    let _ = std::fs::remove_file(&sound_file);
    made?;
    // As a single export: without the tag the track still plays, only not
    // as a sound field.
    if let Err(e) = crate::spherical_inject::inject_sa3d_ambix(output, 4) {
        tracing::warn!("join: the ambisonic tag could not be written (the track is there): {e}");
    }
    Ok(())
}

/// The same sound as Apple Positional Audio: the encoder takes a 24-bit WAV
/// that states no layout and writes the picture and the track into `output`.
#[cfg(target_os = "macos")]
fn mux_apac(wav: &Path, video: &Path, output: &Path) -> Result<()> {
    let plain = output.with_extension("joining.ambi.wav");
    let made = crate::audio::four_channel_to_wav(wav, &plain)
        .and_then(|_| crate::helpers::spawn_apac_encode(&plain, Some(video), output, APAC_BIT_RATE));
    let _ = std::fs::remove_file(&plain);
    made
}

/// Join `parts` (exported files of one shape, in order) into `output`, with
/// the sound `sound`. `progress` counts the video frames copied; `cancel`
/// stops the copy and leaves no file.
pub fn join_exports(
    parts: &[PathBuf],
    output: &Path,
    sound: JoinSound,
    progress: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    join_parts(parts, output, sound, progress, cancel).map(|_| ())
}

/// Join `parts`, Lens Space files that each state their own lens, into the
/// Apple Immersive Video file `output` (`.aivu`): the parts' pictures one
/// after the other and their sound, as [`join_exports`] puts them together,
/// written again packet for packet by the `aivu_wrap` helper with a camera
/// for every part (a mesh of its picture, built from the lens the part's
/// own file states) and, for every frame, the camera it is shown through.
/// Nothing is encoded again, and the picture stays side by side. Apple's
/// validator reads the result; a file it refuses is an error and is not
/// left behind.
///
/// A sound field is written as Apple Positional Audio, the form the players
/// of such a file read ([`JoinSound::Ambisonic`] is taken as
/// [`JoinSound::Apac`]).
///
/// macOS 26 or later ([`immersive_available`]); an error elsewhere.
pub fn join_immersive(
    parts: &[PathBuf],
    output: &Path,
    sound: JoinSound,
    mut progress: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let started = std::time::Instant::now();
        let sound = if sound == JoinSound::Ambisonic { JoinSound::Apac } else { sound };
        // The plain joined file first, beside the output, in the parts' own
        // container; it goes once it has been written again.
        let ext = parts.first().and_then(|p| p.extension()).and_then(|e| e.to_str()).unwrap_or("mp4").to_string();
        let joined = output.with_extension(format!("joined.{ext}"));
        // Two passes over the frames: the bar runs once over both.
        let frames = join_parts(parts, &joined, sound, |p| progress(ExportProgress { total_frames: p.total_frames * 2, ..p }), cancel.clone())?;
        let total: u64 = frames.iter().sum();
        let listed: Vec<(PathBuf, u64)> = parts.iter().cloned().zip(frames).collect();
        let wrapped = crate::helpers::spawn_aivu_wrap(&joined, output, &listed, |done, _| progress(ExportProgress {
            frame_idx: total + done, total_frames: total * 2,
            fps_avg: (total + done) as f32 / started.elapsed().as_secs_f32().max(1e-3),
        }), &cancel);
        let _ = std::fs::remove_file(&joined);
        if let Err(e) = wrapped {
            let _ = std::fs::remove_file(output);
            return Err(match e {
                Error::Helper { stderr, .. } => err(stderr.trim().to_string()),
                other => other,
            });
        }
        tracing::info!("join: {} clips as Apple Immersive Video, a lens for each → {} in {:.1?}", parts.len(), output.display(), started.elapsed());
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (parts, output, sound, &mut progress, cancel);
        Err(err("an Apple Immersive Video file is written on macOS only"))
    }
}

/// [`join_exports`]; returns the number of video frames each part gave.
fn join_parts(
    parts: &[PathBuf],
    output: &Path,
    sound: JoinSound,
    mut progress: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<Vec<u64>> {
    crate::decode::init();
    let infos: Vec<PartInfo> = parts.iter().map(|p| describe(p)).collect::<Result<_>>()?;
    if let Some(why) = obstacle(parts, &infos, sound) {
        return Err(err(why));
    }
    let started = std::time::Instant::now();
    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("mp4").to_string();
    let with_sound = infos.iter().any(|i| i.has_sound);
    // The picture alone first (the whole file when there is no sound).
    let video = if with_sound { output.with_extension(format!("joining.{ext}")) } else { output.to_path_buf() };
    let copied = concat_video(parts, &infos, &video, &mut progress, &cancel)?;
    let (lengths, frames): (Vec<f64>, Vec<u64>) = copied.into_iter().unzip();
    let seconds: f64 = lengths.iter().sum();
    let mut written = "no sound";
    if with_sound {
        // One stretch of sound for the whole length, encoded once (in an
        // `mp4`, which carries an AAC track alone in every build).
        let wav = output.with_extension("joining.sound.wav");
        let made = concat_sound(parts, &lengths, sound.channels(), &wav).and_then(|()| match sound {
            JoinSound::Stereo => {
                written = "stereo sound encoded once";
                let sound_file = output.with_extension("joining.sound.mp4");
                let made = crate::audio::stereo_to_aac(&wav, &sound_file, 0.0, None, AUDIO_BIT_RATE)
                    .and_then(|_| crate::audio::mux_video_with_passthrough_audio(&sound_file, &video, output, 0.0, Some(seconds)));
                let _ = std::fs::remove_file(&sound_file);
                made.map(|_| ())
            }
            JoinSound::Ambisonic => {
                written = "four-channel ambisonic sound encoded once";
                mux_ambisonic(&wav, &video, output, seconds)
            }
            JoinSound::Apac => {
                #[cfg(target_os = "macos")]
                {
                    written = "Apple Positional Audio encoded once";
                    mux_apac(&wav, &video, output).or_else(|e| {
                        tracing::warn!("join: Apple Positional Audio could not be encoded ({e}): the file gets the ambisonic track");
                        written = "four-channel ambisonic sound encoded once";
                        let _ = std::fs::remove_file(output);
                        mux_ambisonic(&wav, &video, output, seconds)
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    written = "four-channel ambisonic sound encoded once";
                    mux_ambisonic(&wav, &video, output, seconds)
                }
            }
        });
        let _ = std::fs::remove_file(&wav);
        let _ = std::fs::remove_file(&video);
        if let Err(e) = made {
            let _ = std::fs::remove_file(output);
            return Err(e);
        }
    }
    tracing::info!(
        "join: {} clips, {:.2} s, {written} → {} in {:.1?}",
        parts.len(), seconds, output.display(), started.elapsed());
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{EncoderBackend, H265Encoder};

    const FPS: f32 = 25.0;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vr180_join_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A 16-bit WAV with one channel per entry of `hz`, each a tone of that
    /// pitch at half scale.
    fn write_tones(path: &Path, hz: &[f32], seconds: f64) {
        let n = (seconds * SAMPLE_RATE as f64).round() as u32;
        let frame = 2 * hz.len() as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + n * frame).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&(hz.len() as u16).to_le_bytes());
        wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE * frame).to_le_bytes());
        wav.extend_from_slice(&(frame as u16).to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(n * frame).to_le_bytes());
        for i in 0..n {
            for pitch in hz {
                let v = ((std::f32::consts::TAU * pitch * i as f32 / SAMPLE_RATE as f32).sin() * 16384.0) as i16;
                wav.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, wav).unwrap();
    }

    /// A clip of `frames` flat grey frames `side` pixels a side, with a
    /// stereo tone of `(hz, seconds)` under it or no sound at all.
    fn part(dir: &Path, name: &str, side: u32, frames: usize, tone: Option<(f32, f64)>) -> PathBuf {
        match tone {
            Some((hz, seconds)) => part_with(dir, name, side, frames, &[hz, hz], seconds),
            None => part_with(dir, name, side, frames, &[], 0.0),
        }
    }

    /// The same with one tone per channel of its sound (`hz`; none: no
    /// sound).
    fn part_with(dir: &Path, name: &str, side: u32, frames: usize, hz: &[f32], seconds: f64) -> PathBuf {
        let video = dir.join(format!("{name}.picture.mov"));
        let mut encoder = H265Encoder::create_with_bit_depth(&video, side, side, FPS, 0, EncoderBackend::ProResKs, 10, 0).unwrap();
        let rgb = vec![96u8; (side * side * 3) as usize];
        for _ in 0..frames {
            encoder.encode_frame(&rgb).unwrap();
        }
        encoder.finish().unwrap();
        let out = dir.join(format!("{name}.mov"));
        if hz.is_empty() {
            std::fs::rename(&video, &out).unwrap();
        } else {
            let wav = dir.join(format!("{name}.wav"));
            write_tones(&wav, hz, seconds);
            crate::audio::mux_video_with_passthrough_audio(&wav, &video, &out, 0.0, None).unwrap();
            std::fs::remove_file(&wav).unwrap();
            std::fs::remove_file(&video).unwrap();
        }
        out
    }

    /// Channel `which` of a file's sound of `channels` channels, as it
    /// decodes.
    fn channel(file: &Path, seconds: f64, channels: usize, which: usize) -> Vec<f32> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut sound = PartSound::new(&mut bytes, channels, (seconds * SAMPLE_RATE as f64).round() as usize);
        decode_into(file, &mut sound).unwrap();
        sound.finish().unwrap();
        bytes.chunks_exact(4 * channels).map(|f| f32::from_le_bytes([f[4 * which], f[4 * which + 1], f[4 * which + 2], f[4 * which + 3]])).collect()
    }

    /// The joined file's sound, as it decodes: left channel.
    fn left_channel(file: &Path, seconds: f64) -> Vec<f32> {
        channel(file, seconds, 2, 0)
    }

    /// How much of `hz` the stretch `[from, to)` seconds holds (the
    /// amplitude of that tone, by correlation).
    fn tone_level(samples: &[f32], from: f64, to: f64, hz: f32) -> f32 {
        let (a, b) = ((from * SAMPLE_RATE as f64) as usize, (to * SAMPLE_RATE as f64) as usize);
        let (mut re, mut im) = (0f32, 0f32);
        for (i, v) in samples[a..b].iter().enumerate() {
            let phase = std::f32::consts::TAU * hz * i as f32 / SAMPLE_RATE as f32;
            re += v * phase.cos();
            im += v * phase.sin();
        }
        2.0 * (re * re + im * im).sqrt() / (b - a) as f32
    }

    #[test]
    fn each_part_keeps_its_sound_under_its_own_picture() {
        crate::decode::init();
        let dir = scratch("sound");
        // Three parts of 5 frames (0.2 s): a tone that outlasts its picture
        // by half its length, no sound at all, a tone that ends early.
        let parts = [
            part(&dir, "a", 64, 5, Some((440.0, 0.30))),
            part(&dir, "b", 64, 5, None),
            part(&dir, "c", 64, 5, Some((1760.0, 0.10))),
        ];
        assert_eq!(join_obstacle(&parts, JoinSound::Stereo).unwrap(), None);
        let out = dir.join("joined.mov");
        let mut last = None;
        join_exports(&parts, &out, JoinSound::Stereo, |p| last = Some(p), Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(last.map(|p| (p.frame_idx, p.total_frames)), Some((15, 15)));

        let joined = describe(&out).unwrap();
        assert_eq!(joined.frames, 15);
        assert!(joined.has_sound);
        // Nothing of the making is left beside it.
        let left: Vec<String> = std::fs::read_dir(&dir).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("joining")).collect();
        assert!(left.is_empty(), "{left:?}");

        let sound = left_channel(&out, 0.6);
        let level = |from, to, hz| tone_level(&sound, from, to, hz);
        // The first tone under the first picture, and gone where the second
        // part starts (a codec frame is 21 ms: look past it).
        assert!(level(0.02, 0.18, 440.0) > 0.4, "{}", level(0.02, 0.18, 440.0));
        assert!(level(0.24, 0.38, 440.0) < 0.01, "{}", level(0.24, 0.38, 440.0));
        // The third part's tone starts with ITS picture, at 0.4 s: a first
        // tone left at its full length would have pushed it to 0.5 s.
        assert!(level(0.42, 0.48, 1760.0) > 0.4, "{}", level(0.42, 0.48, 1760.0));
        assert!(level(0.24, 0.38, 1760.0) < 0.01, "{}", level(0.24, 0.38, 1760.0));
        assert!(level(0.54, 0.60, 1760.0) < 0.01, "{}", level(0.54, 0.60, 1760.0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_sound_field_stays_one_channel_for_channel() {
        crate::decode::init();
        let dir = scratch("field");
        // Four channels, each with a pitch of its own; the second part's
        // are other pitches, and its track ends before its picture.
        let (first, second) = ([300.0, 500.0, 700.0, 900.0], [1100.0, 1300.0, 1500.0, 1700.0]);
        let parts = [
            part_with(&dir, "a", 64, 5, &first, 0.30),
            part_with(&dir, "b", 64, 5, &second, 0.12),
        ];
        assert_eq!(join_obstacle(&parts, JoinSound::Ambisonic).unwrap(), None);
        let out = dir.join("joined.mov");
        join_exports(&parts, &out, JoinSound::Ambisonic, |_| {}, Arc::new(AtomicBool::new(false))).unwrap();
        let joined = describe(&out).unwrap();
        assert_eq!((joined.frames, joined.four_channel_sound), (10, true));
        // Tagged as first-order ambisonics.
        let bytes = std::fs::read(&out).unwrap();
        assert!(bytes.windows(4).any(|w| w == b"SA3D"), "no SA3D box");
        // Every channel holds its own pitch and none of its neighbours',
        // under the picture it belongs to.
        for k in 0..4 {
            let sound = channel(&out, 0.4, 4, k);
            for (j, (a, b)) in first.iter().zip(&second).enumerate() {
                let (in_first, in_second) = (tone_level(&sound, 0.02, 0.18, *a), tone_level(&sound, 0.22, 0.30, *b));
                if j == k {
                    assert!(in_first > 0.4 && in_second > 0.4, "channel {k}: its own pitches at {in_first} and {in_second}");
                } else {
                    assert!(in_first < 0.02 && in_second < 0.02, "channel {k} holds channel {j}'s pitch: {in_first}, {in_second}");
                }
            }
            // The first part's pitch is gone where the second begins, and
            // the second's where its track ended.
            assert!(tone_level(&sound, 0.24, 0.38, first[k]) < 0.01);
            assert!(tone_level(&sound, 0.35, 0.40, second[k]) < 0.01);
        }
        // A part with stereo sound, or none, has no place in a sound field.
        let stereo = part(&dir, "stereo", 64, 5, Some((440.0, 0.2)));
        let silent = part(&dir, "silent", 64, 5, None);
        for odd in [stereo, silent] {
            let why = join_obstacle(&[parts[0].clone(), odd.clone()], JoinSound::Apac).unwrap().expect("not a sound field");
            assert!(why.contains(&name(&odd)) && why.contains("four-channel"), "{why}");
            // As stereo the same two parts join.
            assert_eq!(join_obstacle(&[parts[0].clone(), odd], JoinSound::Stereo).unwrap(), None);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn silent_parts_join_without_a_sound_track() {
        crate::decode::init();
        let dir = scratch("silent");
        let parts = [part(&dir, "a", 64, 3, None), part(&dir, "b", 64, 4, None)];
        let out = dir.join("joined.mov");
        join_exports(&parts, &out, JoinSound::Stereo, |_| {}, Arc::new(AtomicBool::new(false))).unwrap();
        let joined = describe(&out).unwrap();
        assert_eq!((joined.frames, joined.has_sound), (7, false));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_part_of_another_size_is_named() {
        crate::decode::init();
        let dir = scratch("size");
        let parts = [part(&dir, "wide", 64, 2, None), part(&dir, "small", 32, 2, None)];
        let why = join_obstacle(&parts, JoinSound::Stereo).unwrap().expect("two sizes do not join");
        assert!(why.contains("wide.mov") && why.contains("small.mov") && why.contains("64") && why.contains("32"), "{why}");
        let out = dir.join("joined.mov");
        assert!(join_exports(&parts, &out, JoinSound::Stereo, |_| {}, Arc::new(AtomicBool::new(false))).is_err());
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_part_sound_is_cut_filled_and_faded() {
        // 100 frames wanted; the fade is 5 ms = 240 frames, capped at half.
        let mut bytes: Vec<u8> = Vec::new();
        let mut sound = PartSound::new(&mut bytes, 2, 100);
        assert_eq!(sound.fade, 50);
        sound.push(&vec![1.0f32; 2 * 60]).unwrap();
        sound.push(&vec![1.0f32; 2 * 60]).unwrap();
        sound.finish().unwrap();
        let v: Vec<f32> = bytes.chunks_exact(8).map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]])).collect();
        assert_eq!(v.len(), 100);
        assert_eq!(v[0], 0.0);
        assert!(v[25] > 0.4 && v[25] < 0.6 && v[49] > 0.9);
        assert!(v[75] > 0.4 && v[75] < 0.6 && v[99] > 0.0 && v[99] < 0.05);

        // A track that ends before its picture: faded where it ends, then silence.
        let mut bytes: Vec<u8> = Vec::new();
        let mut sound = PartSound::new(&mut bytes, 2, 1000);
        sound.push(&vec![1.0f32; 2 * 600]).unwrap();
        sound.finish().unwrap();
        let v: Vec<f32> = bytes.chunks_exact(8).map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]])).collect();
        assert_eq!(v.len(), 1000);
        assert_eq!(v[300], 1.0);
        assert!(v[599] > 0.0 && v[599] < 0.01 && v[480] > 0.4 && v[480] < 0.6);
        assert!(v[600..].iter().all(|s| *s == 0.0));
    }
}
