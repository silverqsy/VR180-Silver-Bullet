//! The preview worker without a window: `VR180_PREVIEW_PROBE=<clip>` makes the
//! binary start the same worker the app starts for that clip, ask it for a
//! few moments one after the other (paused, as a scrub does), and print one
//! hash per frame it hands back. Two builds given the same clip and the same
//! switches print the same lines exactly when their previews are the same
//! picture, which is what this is for: checking a change to the preview's
//! code the way an export is checked. No part of the UI reaches it, and it
//! touches neither the log file nor the saved settings.
//!
//! Switches (all optional):
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `VR180_PROBE_TIMES` | moments to render, seconds, comma separated | `0.5,1.0,1.5,2.0` |
//! | `VR180_PROBE_MODE` | `equirect`, `fisheye` or `reframe` | `equirect` |
//! | `VR180_PROBE_VIEW` | `sbs`, `anaglyph`, `overlay` or `single` | `sbs` |
//! | `VR180_PROBE_STAB` | `0` turns stabilization off | on |
//! | `VR180_PROBE_RS` | `0` turns the rolling-shutter correction off | the default |
//! | `VR180_PROBE_EYE` | working width of one eye, pixels | `1024` |
//! | `VR180_PROBE_LUT` | a `.cube` file, applied at full strength | none |
//! | `VR180_PROBE_AUTO_EYE` | `1` keeps the automatic eye match | off |
//! | `VR180_PROBE_JPEG` | a folder to write each frame into | none |
//! | `VR180_PROBE_STILL` | `1` also renders each moment's zoom still (the magnifier's and the snapshot's picture) | off |
//! | `VR180_PROBE_STILL_CAP` | width cap of that still's eye, `0` = native | `2048` |

use super::{
    start_decoder, DecodedFrame, DecoderCommand, DecoderConfig, DecoderControl, DetailCache,
    FisheyeOutputMode, PreviewMode, Settings,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vr180_pipeline::gpu::Device;

fn switch(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn settings_for(path: &Path) -> anyhow::Result<Settings> {
    let mut s = Settings::default();
    s.fisheye_output_mode = match switch("VR180_PROBE_MODE").as_deref() {
        None | Some("equirect") => FisheyeOutputMode::HalfEquirect,
        Some("fisheye") => FisheyeOutputMode::Fisheye,
        Some("reframe") => FisheyeOutputMode::Reframe,
        Some(other) => anyhow::bail!("VR180_PROBE_MODE: {other}"),
    };
    s.preview_mode = match switch("VR180_PROBE_VIEW").as_deref() {
        None | Some("sbs") => PreviewMode::Sbs,
        Some("anaglyph") => PreviewMode::Anaglyph,
        Some("overlay") => PreviewMode::Overlay50,
        Some("single") => PreviewMode::SingleEye,
        Some(other) => anyhow::bail!("VR180_PROBE_VIEW: {other}"),
    };
    s.stabilize = switch("VR180_PROBE_STAB").as_deref() != Some("0");
    if let Some(rs) = switch("VR180_PROBE_RS") {
        s.rs_correct = rs != "0";
    }
    // Its smoothing carries from frame to frame; off unless asked for.
    s.auto_eye_match = switch("VR180_PROBE_AUTO_EYE").as_deref() == Some("1");
    if let Some(lut) = switch("VR180_PROBE_LUT") {
        s.lut_path = lut;
        s.lut_intensity = 1.0;
    }
    let kind = vr180_pipeline::source_kind::detect(path)?;
    Ok(if vr180_pipeline::source_kind::is_single_eye(kind, path) { s.for_single_eye() } else { s })
}

/// The frame the worker settles on for the moment `t`: the first one at
/// that moment, then whatever replaces it within a quiet third of a second.
/// The request is repeated every two seconds until a frame answers it: one
/// that lands while the worker is busy (taking in the stabilization data it
/// just finished loading) is not acted on.
fn frame_at(
    frames: &crossbeam_channel::Receiver<DecodedFrame>,
    commands: &crossbeam_channel::Sender<DecoderCommand>,
    t: f64,
) -> anyhow::Result<DecodedFrame> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut asked = Instant::now();
    commands.send(DecoderCommand::Seek(t))?;
    let mut found: Option<DecodedFrame> = None;
    loop {
        let wait = if found.is_some() { Duration::from_millis(350) } else { Duration::from_millis(100) };
        match frames.recv_timeout(wait) {
            // A scrub lands on the frame nearest the moment asked for.
            Ok(f) if (f.timestamp_s - t).abs() <= 0.1 => found = Some(f),
            Ok(f) => tracing::debug!("probe: frame {} at {:.4} s is not t = {t}", f.frame_idx, f.timestamp_s),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if let Some(f) = found.take() {
                    return Ok(f);
                }
                anyhow::ensure!(Instant::now() < deadline, "no frame for t = {t} s");
                if asked.elapsed() > Duration::from_secs(2) {
                    asked = Instant::now();
                    commands.send(DecoderCommand::Seek(t))?;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return found.ok_or_else(|| anyhow::anyhow!("the worker ended before t = {t} s"));
            }
        }
    }
}

pub(crate) fn run(clip: &Path) -> anyhow::Result<()> {
    let path = PathBuf::from(clip);
    anyhow::ensure!(path.is_file(), "{}: not a file", path.display());
    let times: Vec<f64> = switch("VR180_PROBE_TIMES").unwrap_or_else(|| "0.5,1.0,1.5,2.0".into())
        .split(',').map(|t| t.trim().parse()).collect::<Result<_, _>>()?;
    let eye_w: u32 = switch("VR180_PROBE_EYE").map(|v| v.parse()).transpose()?.unwrap_or(1024);
    let jpeg_dir = switch("VR180_PROBE_JPEG").map(PathBuf::from);
    if let Some(dir) = &jpeg_dir {
        std::fs::create_dir_all(dir)?;
    }

    let settings = settings_for(&path)?;
    let stabilized = settings.stabilize;
    let pipeline = Arc::new(Device::new()?);
    let control = Arc::new(DecoderControl {
        paused: true.into(),
        settings: parking_lot::RwLock::new(settings.clone()),
        ..Default::default()
    });
    let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let cfg = DecoderConfig { path: path.clone(), segments: vec![path.clone()], settings, eye_w };
    let worker = {
        let (pipeline, control) = (pipeline.clone(), control.clone());
        std::thread::spawn(move || {
            let result = start_decoder(pipeline, cfg, control.clone(), frame_tx, cmd_rx);
            control.imu_progress.abort.store(true, Ordering::SeqCst);
            control.finished.store(true, Ordering::SeqCst);
            result
        })
    };

    let report = |label: &str, frame: &DecodedFrame| -> anyhow::Result<()> {
        let (w, h) = (frame.width, frame.height);
        let px = pipeline.read_texture_rgb8(&frame.texture, w, h)?;
        println!("{label} frame={:<5} ts={:<9.4} {w}x{h} {:016x}", frame.frame_idx, frame.timestamp_s, fnv(&px));
        if let Some(dir) = &jpeg_dir {
            // The texture holds B, G, R.
            let rgb: Vec<u8> = px.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]).collect();
            image::save_buffer(dir.join(format!("{}.jpg", label.trim())), &rgb, w, h, image::ColorType::Rgb8)?;
        }
        Ok(())
    };

    let mut first = frame_rx.recv_timeout(Duration::from_secs(120))
        .map_err(|_| anyhow::anyhow!("the worker produced no first frame"))?;
    // A photo has no moments: it is the one picture, once the worker has
    // stopped refining it.
    if vr180_pipeline::source_kind::detect(&path)?.is_still() {
        while let Ok(next) = frame_rx.recv_timeout(Duration::from_secs(3)) {
            first = next;
        }
        report("still ", &first)?;
        let _ = cmd_tx.send(DecoderCommand::Stop);
        return Ok(());
    }
    // The worker shows a video's first frame before the stabilization data
    // is in; every moment below is asked for once that data is there.
    drop(first);
    // (With stabilization off the worker never loads it, and never says so.)
    let ready_by = Instant::now() + Duration::from_secs(300);
    while stabilized && (!control.imu_ready.load(Ordering::SeqCst) || control.stab_loading.load(Ordering::SeqCst)) {
        anyhow::ensure!(!control.finished.load(Ordering::SeqCst), "the worker ended while loading");
        anyhow::ensure!(Instant::now() < ready_by, "stabilization data never arrived");
        std::thread::sleep(Duration::from_millis(50));
    }

    let still_settings = control.settings.read().clone();
    let mut zoom = (switch("VR180_PROBE_STILL").as_deref() == Some("1")).then(|| -> anyhow::Result<_> {
        let cap: u32 = switch("VR180_PROBE_STILL_CAP").map(|v| v.parse()).transpose()?.unwrap_or(2048);
        let kind = vr180_pipeline::source_kind::detect(&path)?;
        let fps = vr180_pipeline::decode::probe_video(&path)?.fps;
        Ok((DetailCache::new(path.clone(), vec![path.clone()], kind, fps,
            still_settings.fisheye_swap_eyes, egui::Context::default()), cap))
    }).transpose()?;
    // Let the frame it re-renders with that data come and go.
    while frame_rx.recv_timeout(Duration::from_millis(400)).is_ok() {}

    for &t in &times {
        while frame_rx.try_recv().is_ok() {}
        let frame = frame_at(&frame_rx, &cmd_tx, t)?;
        report(&format!("t={t:<6}"), &frame)?;
        if let Some((cache, cap)) = zoom.as_mut() {
            // Decoded on its own thread: ask until it is there.
            let by = Instant::now() + Duration::from_secs(60);
            let still = loop {
                if let Some(still) = cache.render(&pipeline, frame.timestamp_s, &still_settings, *cap, None) {
                    break still;
                }
                anyhow::ensure!(Instant::now() < by, "no zoom still for t = {t} s");
                std::thread::sleep(Duration::from_millis(20));
            };
            let (texture, width, height) = still;
            report(&format!("still t={t:<6}"), &DecodedFrame {
                texture: Arc::new(texture), width, height,
                frame_idx: frame.frame_idx, timestamp_s: frame.timestamp_s,
            })?;
        }
    }

    let _ = cmd_tx.send(DecoderCommand::Stop);
    drop(frame_rx);
    let stop_by = Instant::now() + Duration::from_secs(10);
    while !worker.is_finished() && Instant::now() < stop_by {
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}
