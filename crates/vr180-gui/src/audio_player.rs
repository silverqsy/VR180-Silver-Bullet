//! Audio-only playback for the preview pipeline.
//!
//! Spawns a worker thread that:
//!   1. Opens the source file via ffmpeg-next.
//!   2. Finds the first audio stream and instantiates a decoder.
//!   3. Resamples decoded frames into the device's stereo f32 format.
//!   4. Pushes interleaved samples into a shared `VecDeque<f32>` buffer.
//!
//! A `cpal` output stream consumes from the same buffer in its
//! audio-thread callback.
//!
//! Public commands (`Pause`, `Resume`, `Seek`, `Stop`) flow through a
//! crossbeam channel. The cpal stream is started in `Pause`d state so
//! it doesn't burn cycles until the GUI flips playback on.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Commands sent from the main thread to the audio decoder worker.
#[derive(Debug)]
enum AudioCmd {
    Pause,
    Resume,
    /// Seek to `seconds` (clip-relative). Drains the buffer first.
    Seek(f64),
    Stop,
}

/// Shared sample queue between the decoder thread (producer) and the
/// cpal callback (consumer). Holds interleaved f32 samples in the
/// device's stereo layout.
type SampleBuf = Arc<Mutex<VecDeque<f32>>>;

/// Public handle. Drop = stop the worker + close the cpal stream.
pub struct AudioPlayer {
    cmd_tx: Sender<AudioCmd>,
    /// Held so the cpal stream stays alive. cpal streams stop on drop.
    _stream: cpal::Stream,
    /// Set true when the worker exits (EOF / error / stop command).
    worker_done: Arc<AtomicBool>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for AudioPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPlayer")
            .field("worker_done", &self.worker_done.load(Ordering::SeqCst))
            .finish()
    }
}

impl AudioPlayer {
    /// Open `path`, set up the cpal output stream, and spawn the decoder
    /// thread. Returns `Ok(None)` if the source has no audio track.
    /// `start_paused = true` keeps the cpal stream in `pause()` until
    /// `set_playing(true)` is called.
    pub fn open(path: PathBuf, start_paused: bool) -> Result<Option<Self>> {
        vr180_pipeline::decode::init();

        // Probe the source for an audio stream before allocating anything.
        let info = match vr180_pipeline::audio::probe_any_audio(&path) {
            Ok(Some(info)) => info,
            Ok(None) => {
                tracing::info!("audio_player: no audio in {}", path.display());
                return Ok(None);
            }
            Err(e) => {
                tracing::warn!("audio_player: probe failed: {e}");
                return Ok(None);
            }
        };
        let (stream_idx, src_channels, src_sample_rate) = info;
        tracing::info!(
            "audio_player: source has audio — stream {}, {} ch, {} Hz",
            stream_idx, src_channels, src_sample_rate
        );

        // Build the cpal default output stream — stereo f32 at whatever
        // sample rate the device picks. The decoder thread resamples
        // the source into this format.
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or_else(|| anyhow!("no default audio output device"))?;
        let supported = device.default_output_config()
            .map_err(|e| anyhow!("default_output_config: {e}"))?;
        let device_sample_rate = supported.sample_rate();
        let device_channels = supported.channels() as u32;
        tracing::info!(
            "audio_player: cpal device {:?} — {} ch @ {} Hz, fmt={:?}",
            device.name().unwrap_or_default(),
            device_channels, device_sample_rate, supported.sample_format()
        );

        // Shared sample queue between decoder (producer) and cpal (consumer).
        // Pre-allocate ~1 s of audio capacity so the decoder thread can
        // run ahead of cpal without the Vec growing in the audio callback.
        let cap_samples = (device_sample_rate as usize)
            * (device_channels as usize)
            * 2;
        let buf: SampleBuf = Arc::new(Mutex::new(VecDeque::with_capacity(cap_samples)));

        // Build the cpal stream. cpal handles all platform glue.
        let stream_config: cpal::StreamConfig = supported.clone().into();
        let err_fn = |e| tracing::warn!("audio_player: cpal stream error: {e}");
        let buf_for_cb = buf.clone();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |out: &mut [f32], _info| {
                    let mut guard = buf_for_cb.lock();
                    let want = out.len();
                    let have = guard.len().min(want);
                    for (slot, sample) in out.iter_mut().zip(guard.drain(..have)) {
                        *slot = sample;
                    }
                    // Pad with silence when underrun (decoder behind).
                    for slot in &mut out[have..] {
                        *slot = 0.0;
                    }
                },
                err_fn,
                None,
            ),
            other => {
                return Err(anyhow!(
                    "audio_player: device sample format {other:?} not supported yet \
                     (need to add resample to that format)"
                ));
            }
        }.map_err(|e| anyhow!("cpal build_output_stream: {e}"))?;

        if start_paused {
            let _ = stream.pause();
        } else {
            let _ = stream.play();
        }

        // Spawn the decoder/resampler thread.
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<AudioCmd>();
        let worker_done = Arc::new(AtomicBool::new(false));
        let worker_done_for_thread = worker_done.clone();
        let path_for_thread = path.clone();
        let buf_for_thread = buf.clone();
        let cap_samples_for_thread = cap_samples;
        let handle = std::thread::spawn(move || {
            if let Err(e) = decoder_worker(
                path_for_thread,
                device_sample_rate,
                device_channels,
                buf_for_thread,
                cap_samples_for_thread,
                cmd_rx,
                start_paused,
            ) {
                tracing::warn!("audio_player worker: {e}");
            }
            worker_done_for_thread.store(true, Ordering::SeqCst);
        });

        Ok(Some(AudioPlayer {
            cmd_tx,
            _stream: stream,
            worker_done,
            worker_handle: Some(handle),
        }))
    }

    /// Flip between playing (cpal `play()`) and paused (cpal `pause()`).
    /// Pause leaves the decoder worker idle but the file still open
    /// — `set_playing(true)` resumes from the same position.
    pub fn set_playing(&self, playing: bool) {
        let cmd = if playing { AudioCmd::Resume } else { AudioCmd::Pause };
        let _ = self.cmd_tx.send(cmd);
        if playing {
            let _ = self._stream.play();
        } else {
            let _ = self._stream.pause();
        }
    }

    /// Drain pending samples and re-seek the decoder to `seconds`.
    pub fn seek_to(&self, seconds: f64) {
        let _ = self.cmd_tx.send(AudioCmd::Seek(seconds.max(0.0)));
    }

    /// True after the worker thread reports EOF / error / stop.
    #[allow(dead_code)]
    pub fn is_done(&self) -> bool {
        self.worker_done.load(Ordering::SeqCst)
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(AudioCmd::Stop);
        if let Some(h) = self.worker_handle.take() {
            // Best-effort join.
            let _ = h.join();
        }
    }
}

/// Worker thread: open the source, find the audio stream, decode/resample
/// in a loop, push samples into the shared buffer. Reacts to Pause / Seek /
/// Stop commands between packets.
fn decoder_worker(
    path: PathBuf,
    device_sample_rate: u32,
    device_channels: u32,
    buf: SampleBuf,
    cap_samples: usize,
    cmd_rx: crossbeam_channel::Receiver<AudioCmd>,
    start_paused: bool,
) -> Result<()> {
    use ffmpeg_next as ffmpeg;
    use ffmpeg::software::resampling::Context as Resampler;

    let mut ictx = ffmpeg::format::input(&path)
        .map_err(|e| anyhow!("open {path:?}: {e}"))?;

    // Locate the audio stream.
    let audio_stream = ictx.streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!("no audio stream"))?;
    let audio_idx = audio_stream.index();

    // Decoder context from stream parameters.
    let dec_ctx = ffmpeg::codec::context::Context::from_parameters(audio_stream.parameters())
        .map_err(|e| anyhow!("from_parameters: {e}"))?;
    let mut decoder = dec_ctx.decoder().audio()
        .map_err(|e| anyhow!("decoder().audio(): {e}"))?;
    decoder.set_parameters(audio_stream.parameters())
        .map_err(|e| anyhow!("set_parameters: {e}"))?;

    let src_rate = decoder.rate();
    let src_format = decoder.format();
    let src_ch_layout = decoder.channel_layout();
    tracing::info!(
        "audio_player worker: decoder ready — src {} Hz, fmt {:?}, layout {:?}",
        src_rate, src_format, src_ch_layout
    );

    // Resampler: source → device's stereo f32 packed.
    let dst_ch_layout = if device_channels == 1 {
        ffmpeg::channel_layout::ChannelLayout::MONO
    } else {
        ffmpeg::channel_layout::ChannelLayout::STEREO
    };
    let mut resampler = Resampler::get(
        src_format,
        src_ch_layout,
        src_rate,
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
        dst_ch_layout,
        device_sample_rate,
    ).map_err(|e| anyhow!("resampler init: {e}"))?;

    let device_channels = device_channels as usize;

    // Main loop. We pull commands non-blockingly between each packet —
    // pause/seek/stop take effect within at most one decode step.
    let mut paused = start_paused;
    let mut packet_iter = ictx.packets();

    // Threshold above which we yield to the audio thread instead of
    // pulling more packets. Two cap_samples / 2 ≈ 1 s of buffer.
    let high_water = cap_samples / 2;

    loop {
        // Drain any pending commands.
        let mut should_seek: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                AudioCmd::Pause => paused = true,
                AudioCmd::Resume => paused = false,
                AudioCmd::Stop => return Ok(()),
                AudioCmd::Seek(t) => should_seek = Some(t),
            }
        }
        if let Some(t) = should_seek {
            // Drain queued samples so the cpal callback stops emitting
            // pre-seek audio (next `out` it touches will read silence
            // until the new decode catches up).
            {
                let mut g = buf.lock();
                g.clear();
            }
            // ffmpeg seek expects AV_TIME_BASE microseconds.
            let target_us = (t * 1_000_000.0) as i64;
            let _ = ictx.seek(target_us, ..target_us);
            decoder.flush();
            packet_iter = ictx.packets();
            continue;
        }

        // When paused, sleep briefly to avoid busy-looping the channel.
        if paused {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        // If the consumer is well ahead of us, sleep instead of
        // decoding more. Keeps the buffer at roughly 0.5 - 1 s.
        let queued = buf.lock().len();
        if queued >= high_water {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        }

        // Pull the next audio packet.
        let (stream, packet) = match packet_iter.next() {
            Some(x) => x,
            None => {
                tracing::info!("audio_player worker: EOF");
                return Ok(());
            }
        };
        if stream.index() != audio_idx { continue; }

        if decoder.send_packet(&packet).is_err() {
            continue;
        }

        // Pull all available decoded frames for this packet, resample,
        // push samples into the shared buffer.
        let mut frame = ffmpeg::frame::Audio::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            let mut out_frame = ffmpeg::frame::Audio::empty();
            if let Err(e) = resampler.run(&frame, &mut out_frame) {
                tracing::warn!("audio_player worker: resample: {e}");
                continue;
            }
            let n_samples = out_frame.samples() as usize;
            let stride = device_channels * std::mem::size_of::<f32>();
            let plane = out_frame.data(0);
            let needed_bytes = n_samples * stride;
            let bytes = &plane[..needed_bytes.min(plane.len())];
            let samples: &[f32] = bytemuck::cast_slice(bytes);

            let mut g = buf.lock();
            g.extend(samples.iter().copied());
        }
    }
}
