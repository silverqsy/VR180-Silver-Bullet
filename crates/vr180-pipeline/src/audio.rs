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
