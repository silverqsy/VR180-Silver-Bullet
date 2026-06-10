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
    let ictx = ffmpeg::format::input(&input)
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
    crate::decode::init();
    let info = probe_ambisonic(input)?
        .ok_or_else(|| Error::Ffmpeg(format!(
            "no ambisonic audio track in {input:?}"
        )))?;

    let mut ictx = ffmpeg::format::input(&input)
        .map_err(|e| Error::Ffmpeg(format!("open input: {e}")))?;
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

    let mut packet_count = 0u64;
    for (stream, mut packet) in ictx.packets() {
        if stream.index() != info.stream_index { continue; }
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

/// Pick the most useful audio stream in `path` — first non-attached-pic
/// audio track, regardless of channel count or codec. Returns the
/// stream index, channel count, and sample rate. `None` if the file
/// has no audio.
pub fn probe_any_audio(input: &Path) -> Result<Option<(usize, u32, u32)>> {
    crate::decode::init();
    let ictx = ffmpeg::format::input(&input)
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
pub fn mux_video_with_passthrough_audio(
    audio_src: &Path,
    video_src: &Path,
    final_out: &Path,
    audio_offset_s: f64,
    max_duration_s: Option<f64>,
) -> Result<u64> {
    // `audio_offset_s` / `max_duration_s` implement TRIMMED exports: the
    // video temp starts at source-time `trim_in` but its own timeline at 0,
    // so the source audio must be cut at `audio_offset_s` (= trim_in),
    // rebased to 0, and stopped after the video's duration. Without this a
    // trimmed export carried the FULL source audio at original timestamps —
    // wrong length and out of sync by trim_in seconds.
    crate::decode::init();

    let mut a_in = ffmpeg::format::input(&audio_src)
        .map_err(|e| Error::Ffmpeg(format!("open audio src {audio_src:?}: {e}")))?;
    let mut v_in = ffmpeg::format::input(&video_src)
        .map_err(|e| Error::Ffmpeg(format!("open video src {video_src:?}: {e}")))?;

    // Locate the first audio stream in the audio source.
    let a_idx = a_in.streams()
        .filter(|s| {
            s.parameters().medium() == ffmpeg::media::Type::Audio
        })
        .map(|s| s.index())
        .next();
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
