//! egui application state + UI.
//!
//! The state owns:
//! - the shared `vr180-pipeline` `Device` (built on top of eframe's
//!   own wgpu device so textures can flow between the two without
//!   any copies)
//! - a single decoder worker thread (see [`crate::decoder`]) that
//!   produces `wgpu::Texture`s on a channel
//! - the most recently displayed texture, registered with the egui
//!   renderer for display

use crossbeam_channel::{Receiver, Sender};
use egui::{Color32, RichText};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::decoder::{
    DecodedFrame, DecoderCommand, DecoderConfig, DecoderControl, Settings, start_decoder,
};

/// The single egui application.
pub struct App {
    /// Shared with the decoder worker. Built on eframe's wgpu device.
    pipeline: Arc<vr180_pipeline::gpu::Device>,
    /// Handle to the egui_wgpu renderer — we use it to register
    /// `wgpu::Texture`s for display. egui's RwLock wraps parking_lot's
    /// internally; we hold an Arc to share with future worker threads
    /// that want to register textures (currently the App holds it
    /// exclusively).
    egui_renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,

    // ─── User-visible state ──────────────────────────────────────
    loaded_path: Option<PathBuf>,
    clip: Option<ClipInfo>,
    /// Playing means the decoder thread is alive AND not paused.
    /// Paused-but-not-stopped (decoder thread parked) is `playing=false`
    /// with `decoder_alive=true`.
    playing: bool,
    /// True while a decoder thread is alive (started but not Stopped).
    /// Lets Pause toggle the `paused` flag instead of killing the worker.
    decoder_alive: bool,
    /// Currently displayed frame texture (kept alive while egui still
    /// references it) + the egui texture id we registered it under.
    current_display: Option<DisplayFrame>,
    /// Pending frames coming off the decoder. We drain on every
    /// repaint and pick the most recent one — naturally drops frames
    /// the UI couldn't keep up with.
    frame_rx: Option<Receiver<DecodedFrame>>,
    cmd_tx: Option<Sender<DecoderCommand>>,
    /// Shared with the decoder for pause / live-settings.
    control: Option<Arc<DecoderControl>>,
    /// Running fps measurement, refreshed every 500 ms.
    fps_stats: FpsStats,

    // ─── Settings panel ──────────────────────────────────────────
    settings: Settings,
    /// Last settings value we pushed to the shared control. Used to
    /// detect when to bump the generation counter (so the decoder
    /// recomputes per-eye bundles).
    last_pushed_settings: Settings,

    // ─── Export job (fisheye sources only for now) ───────────────
    /// `Some` while an export is in flight. `None` once the worker
    /// finishes (cleanly or via cancel) — the GUI re-enables the
    /// Export button.
    export_job: Option<ExportJob>,
}

struct ExportJob {
    output_path: PathBuf,
    handle: std::thread::JoinHandle<vr180_pipeline::Result<()>>,
    progress_rx: crossbeam_channel::Receiver<vr180_pipeline::fisheye_export::ExportProgress>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Most recent progress update — refreshed by drain on every UI
    /// frame so the progress bar / counters don't lag.
    last_progress: Option<vr180_pipeline::fisheye_export::ExportProgress>,
    started_at: std::time::Instant,
}

struct DisplayFrame {
    texture: Arc<wgpu::Texture>,
    egui_id: egui::TextureId,
    width: u32,
    height: u32,
    frame_idx: u32,
    timestamp_s: f64,
}

#[derive(Default)]
struct FpsStats {
    frames_in_window: u32,
    last_window_start: Option<std::time::Instant>,
    last_fps: f32,
}

#[derive(Debug, Clone)]
struct ClipInfo {
    width: u32,
    height: u32,
    fps: f32,
    duration_sec: f64,
    frame_count: u32,
    segments: Vec<PathBuf>,
    eac_tile_w: u32,
    cori_count: usize,
    grav_count: usize,
    srot_ms: Option<f32>,
    /// Detected source family. Drives which settings panels show in
    /// the sidebar — GoPro EAC keeps the stab / RS panels; fisheye
    /// sources (OSV / SBS / BRAW) get the camera-preset / FOV / KB
    /// panel instead.
    source_kind: vr180_pipeline::SourceKind,
    /// One eye dimensions (set for fisheye sources; (0,0) for EAC).
    /// Used by the GUI to label the FOV slider with a realistic
    /// "max recommended" hint based on the working resolution.
    fisheye_eye_w: u32,
    fisheye_eye_h: u32,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Force dark visuals regardless of the OS appearance preference.
        // egui's dark theme reads better against the half-equirect
        // preview (which is typically dark imagery itself) and matches
        // the rest of the colorist tooling our users come from
        // (DaVinci, FCP, Premiere, Insta360 Studio — all dark).
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        // Pull the wgpu Arcs that eframe set up. Sharing the device
        // means textures we produce in the pipeline live in the same
        // wgpu::Device the egui renderer draws from — registering
        // them with `Renderer::register_native_texture` is free.
        let wgpu_state = cc.wgpu_render_state.as_ref()
            .expect("eframe must be running with Renderer::Wgpu");
        // wgpu::Instance isn't kept by eframe's render state in v0.28
        // (it's dropped after device creation). We don't use the
        // instance in our pipeline after construction, so a fresh
        // throwaway one is fine.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let pipeline = vr180_pipeline::gpu::Device::from_existing(
            instance,
            wgpu_state.adapter.clone(),
            wgpu_state.device.clone(),
            wgpu_state.queue.clone(),
        ).expect("pipeline device init from shared wgpu device");
        let pipeline = Arc::new(pipeline);
        tracing::info!(
            backend = ?pipeline.adapter.get_info().backend,
            name = %pipeline.adapter.get_info().name,
            "pipeline running on the eframe-owned wgpu device"
        );

        let defaults = Settings::default();
        Self {
            pipeline,
            egui_renderer: wgpu_state.renderer.clone(),
            loaded_path: None,
            clip: None,
            playing: false,
            decoder_alive: false,
            current_display: None,
            frame_rx: None,
            cmd_tx: None,
            control: None,
            fps_stats: FpsStats::default(),
            settings: defaults.clone(),
            last_pushed_settings: defaults,
            export_job: None,
        }
    }

    /// Pick an output path and spawn the export worker. No-op if no
    /// clip is loaded, an export is already in flight, or the source
    /// is GoPro EAC (the existing CLI handles that family).
    fn start_export(&mut self) {
        if self.export_job.is_some() { return; }
        let (path, clip) = match (&self.loaded_path, &self.clip) {
            (Some(p), Some(c)) => (p.clone(), c.clone()),
            _ => return,
        };
        if !clip.source_kind.is_fisheye() {
            tracing::warn!("export: only fisheye sources supported (got {:?})", clip.source_kind);
            return;
        }

        // Default output path: same dir, _SBS.mp4 suffix.
        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let default_out = path.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_SBS.mp4"));

        let chosen = rfd::FileDialog::new()
            .add_filter("H.265 MP4", &["mp4", "mov"])
            .set_file_name(default_out.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output.mp4".into()))
            .set_directory(default_out.parent().unwrap_or(std::path::Path::new(".")))
            .save_file();
        let Some(output_path) = chosen else { return; };

        // Pick the backend: VT on macOS, libx265 elsewhere.
        let backend = if cfg!(target_os = "macos") {
            vr180_pipeline::encode::EncoderBackend::VideoToolbox
        } else {
            vr180_pipeline::encode::EncoderBackend::Libx265
        };

        // Output resolution: native source eye dims (no downscale).
        // For OSV that's 3840×3840 per eye → 7680×3840 SBS output.
        // For BRAW Pyxis it's whatever the .braw natively decodes to.
        // The fisheye projection shader handles any input resolution
        // because it samples KB-radially — runtime is roughly linear
        // in output pixel count.
        let (eye_w, eye_h) = if clip.fisheye_eye_w > 0 {
            (clip.fisheye_eye_w, clip.fisheye_eye_h)
        } else {
            // Fallback (shouldn't really hit for fisheye sources, but
            // be defensive — match the stream dims).
            (clip.width.max(512), clip.height.max(512))
        };

        // Default 200 Mbps. Bit depth = 10 → true end-to-end 10-bit
        // path: scaler outputs RGBA64LE, projection runs on a
        // Rgba16Unorm pipeline, compose at 16-bit, readback as
        // RGB48LE, encoder takes RGB48LE → Main10. Slower than the
        // 8-bit zero-copy path (no P010 IOSurface yet) but
        // genuinely 10-bit precise from source to codec.
        let bitrate_kbps: u32 = 200_000;
        let bit_depth: u8 = 10;

        let cfg = vr180_pipeline::fisheye_export::FisheyeExportConfig {
            source_path: path,
            output_path: output_path.clone(),
            source_kind: clip.source_kind,
            eye_w,
            eye_h,
            fps: clip.fps,
            bitrate_kbps,
            encoder: backend,
            bit_depth,
            stabilize: self.settings.stabilize,
            fisheye_preset: self.settings.fisheye_preset.clone(),
            fisheye_fov_deg: self.settings.fisheye_fov_deg,
            fisheye_k: self.settings.fisheye_k,
            fisheye_cx_offset_px: self.settings.fisheye_cx_offset_px,
            fisheye_cy_offset_px: self.settings.fisheye_cy_offset_px,
            fisheye_swap_eyes: self.settings.fisheye_swap_eyes,
            trim_in_s: self.settings.trim_in_s,
            trim_out_s: self.settings.trim_out_s,
        };

        // Stop preview playback for the duration of the export — both
        // share the same wgpu device + ffmpeg input contexts on the
        // same file, no point racing.
        self.stop_playback();

        let (progress_tx, progress_rx) = crossbeam_channel::bounded(256);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        let pipeline = self.pipeline.clone();
        let handle = std::thread::spawn(move || {
            vr180_pipeline::fisheye_export::export_fisheye(
                pipeline,
                cfg,
                move |p| { let _ = progress_tx.try_send(p); },
                cancel_for_thread,
            )
        });
        tracing::info!("export: spawned worker → {}", output_path.display());

        self.export_job = Some(ExportJob {
            output_path,
            handle,
            progress_rx,
            cancel,
            last_progress: None,
            started_at: std::time::Instant::now(),
        });
    }

    /// Drain progress channel, check if the worker has finished, and
    /// promote `export_job` to None when it has.
    fn poll_export_job(&mut self) {
        let Some(job) = self.export_job.as_mut() else { return; };
        // Drain the channel.
        while let Ok(p) = job.progress_rx.try_recv() {
            job.last_progress = Some(p);
        }
        // Check completion. JoinHandle::is_finished is stable in 1.61+.
        if job.handle.is_finished() {
            let job = self.export_job.take().unwrap();
            match job.handle.join() {
                Ok(Ok(())) => tracing::info!(
                    "export: done → {} (took {:.2?})",
                    job.output_path.display(), job.started_at.elapsed()
                ),
                Ok(Err(e)) => tracing::error!("export: failed: {e}"),
                Err(_) => tracing::error!("export: worker panicked"),
            }
        }
    }

    /// Drain whatever's in the frame channel, register the newest
    /// frame's texture with egui (freeing the previous one).
    ///
    /// We keep at most ONE frame alive: every received frame replaces
    /// the previous display. Older frames received in the same poll
    /// are dropped — which is the right behavior for "real-time, no
    /// buffering" playback.
    fn drain_frames(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.frame_rx else { return; };
        let mut newest: Option<DecodedFrame> = None;
        let mut drained = 0u32;
        while let Ok(f) = rx.try_recv() {
            self.fps_stats.frames_in_window += 1;
            newest = Some(f);
            drained += 1;
        }
        if drained > 0 && self.fps_stats.last_window_start.is_none() {
            tracing::info!("ui drain: first frame received (drained {} on first poll)", drained);
        }
        if let Some(f) = newest {
            self.replace_display(f);
            ctx.request_repaint();
        }
        // FPS aggregation.
        let now = std::time::Instant::now();
        match self.fps_stats.last_window_start {
            None => self.fps_stats.last_window_start = Some(now),
            Some(start) => {
                let dt = now.duration_since(start).as_secs_f32();
                if dt > 0.5 {
                    self.fps_stats.last_fps = self.fps_stats.frames_in_window as f32 / dt;
                    self.fps_stats.frames_in_window = 0;
                    self.fps_stats.last_window_start = Some(now);
                }
            }
        }
    }

    fn replace_display(&mut self, frame: DecodedFrame) {
        let is_first = self.current_display.is_none();
        let mut renderer = self.egui_renderer.write();
        // Free the previous texture id, if any.
        if let Some(prev) = self.current_display.take() {
            renderer.free_texture(&prev.egui_id);
        }
        // Register the new texture. Use a view of the SBS texture as
        // the egui-side handle. wgpu::Texture is Arc internally so
        // keeping our own Arc + handing eframe a view is fine.
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = renderer.register_native_texture(
            &self.pipeline.device, &view, wgpu::FilterMode::Linear,
        );
        if is_first {
            tracing::info!("ui: first frame registered with egui (id={:?}, {}×{})",
                id, frame.width, frame.height);
        }
        self.current_display = Some(DisplayFrame {
            texture: frame.texture,
            egui_id: id,
            width: frame.width,
            height: frame.height,
            frame_idx: frame.frame_idx,
            timestamp_s: frame.timestamp_s,
        });
    }

    // ─── File loading ────────────────────────────────────────────

    fn pick_and_load_file(&mut self) {
        let path = rfd::FileDialog::new()
            .add_filter(
                "All supported (.360 / .osv / .braw / .mp4 / .mov)",
                &["360", "osv", "OSV", "braw", "BRAW",
                  "mp4", "MP4", "mov", "MOV"],
            )
            .add_filter("GoPro Max (.360)", &["360"])
            .add_filter("DJI Osmo 360 (.osv)", &["osv", "OSV"])
            .add_filter("Blackmagic RAW (.braw)", &["braw", "BRAW"])
            .add_filter("Side-by-side fisheye (.mp4 / .mov)",
                &["mp4", "MP4", "mov", "MOV"])
            .pick_file();
        if let Some(p) = path { self.load_file(p); }
    }

    fn load_file(&mut self, path: PathBuf) {
        // ── Tear down everything tied to the previous clip ──────────
        // 1. Stop the decoder thread + drop the IPC channels.
        self.stop_playback();
        // 2. Free the egui-registered preview texture from the old
        //    clip. Without this the previous frame stays painted in
        //    the central panel until the new decoder produces output.
        if let Some(prev) = self.current_display.take() {
            let mut renderer = self.egui_renderer.write();
            renderer.free_texture(&prev.egui_id);
            // Dropping `prev.texture` (the Arc<wgpu::Texture>) below
            // releases the GPU memory when the last reference goes.
        }
        // 3. Reset volatile playback state. The user's persistent
        //    preferences (stabilize toggle, preview_eye_w) stay.
        self.fps_stats = FpsStats::default();
        self.settings.trim_in_s = None;
        self.settings.trim_out_s = None;
        self.settings.fisheye_preset.clear();
        self.settings.fisheye_fov_deg = 0.0;
        self.settings.fisheye_k = [0.0; 4];
        self.settings.fisheye_cx_offset_px = 0.0;
        self.settings.fisheye_cy_offset_px = 0.0;
        self.settings.fisheye_swap_eyes = false;
        self.settings.fisheye_output_mode = crate::decoder::FisheyeOutputMode::HalfEquirect;
        // 4. Drop ClipInfo + path so the sidebar shows "no file"
        //    momentarily if probe / detection below fails.
        self.clip = None;
        self.loaded_path = None;

        // ── Detect new clip and load metadata ──────────────────────
        let source_kind = vr180_pipeline::source_kind::detect(&path)
            .unwrap_or(vr180_pipeline::SourceKind::Unknown);
        tracing::info!("load_file: {} → kind={:?}", path.display(), source_kind);

        let probe = match vr180_pipeline::decode::probe_video(&path) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("probe failed: {e}");
                return;
            }
        };
        let dims = vr180_core::eac::Dims::new(probe.width, probe.height);

        // GoPro family: parse GPMF + GEOC for metadata display.
        let (cori_count, grav_count, srot_ms) = if source_kind == vr180_pipeline::SourceKind::GoProEac {
            match vr180_pipeline::decode::extract_gpmf_stream(&path) {
                Ok(gpmf) => {
                    let cori = vr180_core::gyro::parse_cori(&gpmf);
                    let raw = vr180_core::gyro::parse_raw_imu(&gpmf);
                    let srot = vr180_core::geoc::find_srot_ms(&gpmf)
                        .or_else(|| {
                            vr180_core::geoc::lookup_srot_s(&path, None)
                                .ok().flatten().map(|s| s * 1000.0)
                        });
                    (cori.len(), raw.grav.len(), srot)
                }
                Err(_) => (0, 0, None),
            }
        } else {
            (0, 0, None)
        };

        // Fisheye family: compute one-eye dimensions for FOV slider hints.
        let (fisheye_eye_w, fisheye_eye_h) = match source_kind {
            vr180_pipeline::SourceKind::SbsFisheye => (probe.width / 2, probe.height),
            vr180_pipeline::SourceKind::DjiOsv     => (probe.width, probe.height),
            vr180_pipeline::SourceKind::BlackmagicRaw => {
                // braw_helper --info may not be installed; fall back to probe.
                if let Ok(info) = vr180_braw::BrawInfo::probe(&path) {
                    if info.is_dual_track() {
                        (info.width / 2, info.height)
                    } else {
                        (info.width, info.height)
                    }
                } else {
                    (probe.width, probe.height)
                }
            }
            _ => (0, 0),
        };

        // Auto-fill the preset name in settings based on detection.
        // Unconditional — we just cleared it above, so this lands the
        // right default for the NEW clip regardless of what the prior
        // clip was.
        if source_kind != vr180_pipeline::SourceKind::GoProEac {
            let auto = match source_kind {
                vr180_pipeline::SourceKind::DjiOsv        => "DJI Osmo 360",
                vr180_pipeline::SourceKind::BlackmagicRaw => "Blackmagic Pyxis 12K",
                _                                         => "Custom",
            };
            self.settings.fisheye_preset = auto.to_string();
        }
        // Snapshot settings so the next spawn_decoder doesn't bump
        // generation unnecessarily on the first frame.
        self.last_pushed_settings = self.settings.clone();

        let segments = vr180_core::segments::detect_segments(&path);

        self.clip = Some(ClipInfo {
            width: probe.width, height: probe.height,
            fps: probe.fps, duration_sec: probe.duration_sec,
            frame_count: (probe.duration_sec * probe.fps as f64).round() as u32,
            segments,
            eac_tile_w: dims.tile_w(),
            cori_count, grav_count, srot_ms,
            source_kind,
            fisheye_eye_w, fisheye_eye_h,
        });
        self.loaded_path = Some(path);
    }

    // ─── Playback control ────────────────────────────────────────

    /// Spawn a fresh decoder thread. Used when starting playback on a
    /// freshly loaded clip (or when explicitly Stop'd and restarted).
    fn spawn_decoder(&mut self) {
        let Some(path) = self.loaded_path.clone() else {
            tracing::warn!("spawn_decoder: no path loaded");
            return;
        };
        self.stop_playback();
        tracing::info!("spawn_decoder: starting decoder for {}", path.display());
        let cfg = DecoderConfig {
            path,
            settings: self.settings.clone(),
            eye_w: self.settings.preview_eye_w,
        };
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let control = Arc::new(DecoderControl {
            paused: std::sync::atomic::AtomicBool::new(false),
            settings: parking_lot::RwLock::new(self.settings.clone()),
            settings_generation: std::sync::atomic::AtomicU64::new(0),
        });
        let pipeline = self.pipeline.clone();
        let control_for_thread = control.clone();
        std::thread::spawn(move || {
            if let Err(e) = start_decoder(pipeline, cfg, control_for_thread, frame_tx, cmd_rx) {
                tracing::error!("decoder error: {e}");
            } else {
                tracing::info!("decoder thread exited cleanly");
            }
        });
        self.frame_rx = Some(frame_rx);
        self.cmd_tx = Some(cmd_tx);
        self.control = Some(control);
        self.playing = true;
        self.decoder_alive = true;
        self.last_pushed_settings = self.settings.clone();
        self.fps_stats = FpsStats::default();
    }

    /// Toggle play/pause. If no decoder is alive yet, this kicks off
    /// a fresh one. Otherwise it just flips the paused flag — the
    /// decoder thread keeps its position so resume picks up
    /// exactly where pause left it.
    fn toggle_play_pause(&mut self) {
        if !self.decoder_alive {
            self.spawn_decoder();
            return;
        }
        if let Some(ctl) = &self.control {
            self.playing = !self.playing;
            ctl.paused.store(!self.playing, Ordering::SeqCst);
            tracing::info!("toggle_play_pause: playing={}", self.playing);
        }
    }

    /// Send a Seek command to the running decoder. If no decoder is
    /// alive, lazily spawns one (the seek then becomes the initial
    /// position via the regular pipeline).
    fn seek_to(&mut self, target_s: f64) {
        if !self.decoder_alive {
            self.spawn_decoder();
        }
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecoderCommand::Seek(target_s.max(0.0)));
        }
    }

    fn set_trim_in(&mut self, t: Option<f64>) {
        // Auto-clear if it goes past out. Don't let in > out.
        let t = t.map(|v| v.max(0.0));
        if let (Some(ti), Some(to)) = (t, self.settings.trim_out_s) {
            if ti >= to { return; }
        }
        self.settings.trim_in_s = t;
    }
    fn set_trim_out(&mut self, t: Option<f64>) {
        let t = t.map(|v| v.max(0.0));
        if let (Some(ti), Some(to)) = (self.settings.trim_in_s, t) {
            if to <= ti { return; }
        }
        self.settings.trim_out_s = t;
    }

    /// Render the timeline scrubber: a horizontal bar showing
    /// playhead position, with click/drag to seek and visual
    /// trim-in/out markers.
    fn draw_timeline(&mut self, ui: &mut egui::Ui) {
        let total = self.clip.as_ref().map(|c| c.duration_sec).unwrap_or(0.0);
        let can_seek = self.clip.is_some() && total > 0.0;

        // Reserve a strip with room for the track + labels.
        let desired = egui::vec2(ui.available_width(), 22.0);
        let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());

        // Background track.
        let track_y0 = rect.top() + 8.0;
        let track_y1 = rect.bottom() - 4.0;
        let track = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 6.0, track_y0),
            egui::pos2(rect.right() - 6.0, track_y1),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(track, 3.0, Color32::from_rgb(38, 42, 50));

        // Position helpers.
        let track_w = track.width().max(1.0);
        let t_to_x = |t: f64| -> f32 {
            let r = (t / total).clamp(0.0, 1.0) as f32;
            track.left() + r * track_w
        };
        let x_to_t = |x: f32| -> f64 {
            let r = ((x - track.left()) / track_w).clamp(0.0, 1.0) as f64;
            r * total
        };

        // Trim shaded region.
        if can_seek {
            let in_t  = self.settings.trim_in_s.unwrap_or(0.0);
            let out_t = self.settings.trim_out_s.unwrap_or(total);
            let x_in  = t_to_x(in_t);
            let x_out = t_to_x(out_t);
            let shaded = egui::Rect::from_min_max(
                egui::pos2(x_in, track_y0),
                egui::pos2(x_out, track_y1),
            );
            painter.rect_filled(shaded, 3.0, Color32::from_rgb(56, 84, 128));
            // Marker lines.
            if self.settings.trim_in_s.is_some() {
                painter.line_segment(
                    [egui::pos2(x_in, track_y0 - 3.0), egui::pos2(x_in, track_y1 + 3.0)],
                    egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 255)),
                );
            }
            if self.settings.trim_out_s.is_some() {
                painter.line_segment(
                    [egui::pos2(x_out, track_y0 - 3.0), egui::pos2(x_out, track_y1 + 3.0)],
                    egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 255)),
                );
            }

            // Playhead.
            let cur = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
            let x_cur = t_to_x(cur);
            painter.line_segment(
                [egui::pos2(x_cur, track_y0 - 4.0), egui::pos2(x_cur, track_y1 + 4.0)],
                egui::Stroke::new(2.0, Color32::from_rgb(255, 220, 80)),
            );
            painter.circle_filled(
                egui::pos2(x_cur, (track_y0 + track_y1) * 0.5),
                4.0, Color32::from_rgb(255, 220, 80),
            );
        }

        // Click / drag → seek.
        if can_seek {
            if let Some(pos) = resp.interact_pointer_pos() {
                if resp.clicked() || resp.dragged() {
                    let t = x_to_t(pos.x);
                    self.seek_to(t);
                }
            }
        }
    }

    /// Tear down the decoder thread for good. Resets position so the
    /// next Play starts from frame 0.
    fn stop_playback(&mut self) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecoderCommand::Stop);
        }
        // Make sure the decoder isn't waiting on a Pause flag right
        // before it sees the Stop — unpause so it processes the queue.
        if let Some(ctl) = &self.control {
            ctl.paused.store(false, Ordering::SeqCst);
        }
        self.frame_rx = None;
        self.cmd_tx = None;
        self.control = None;
        self.playing = false;
        self.decoder_alive = false;
    }

    /// Diff `self.settings` against `last_pushed_settings`; if changed,
    /// push to the shared control and bump the generation counter so
    /// the decoder rebuilds per-eye bundles on its next iteration.
    fn maybe_push_settings(&mut self) {
        if self.settings == self.last_pushed_settings {
            return;
        }
        if let Some(ctl) = &self.control {
            *ctl.settings.write() = self.settings.clone();
            ctl.settings_generation.fetch_add(1, Ordering::SeqCst);
            tracing::debug!("settings changed → generation bumped");
        }
        self.last_pushed_settings = self.settings.clone();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain decoder frames first so the most recent texture is
        // ready by the time we render the central panel.
        self.drain_frames(ctx);
        // Drain export-job progress + check for completion.
        self.poll_export_job();
        // Force a continuous repaint while an export is in flight so
        // the progress bar updates without needing mouse input.
        if self.export_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Handle dropped files (drag-and-drop into the window).
        ctx.input(|i| {
            for f in &i.raw.dropped_files {
                if let Some(p) = f.path.clone() {
                    self.load_file(p);
                    break;
                }
            }
        });

        // Keyboard shortcuts. Only fire when no text widget owns the
        // focus — egui's `wants_keyboard_input` handles that for us.
        // We capture trim-in/out actions inside the closure (borrows
        // `&i.modifiers` immutably) and apply them after.
        let mut pending_trim_in:  Option<Option<f64>> = None;
        let mut pending_trim_out: Option<Option<f64>> = None;
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if i.key_pressed(egui::Key::Space) {
                    self.toggle_play_pause();
                }
                if i.key_pressed(egui::Key::S) && i.modifiers.is_none() {
                    self.stop_playback();
                }
                if i.key_pressed(egui::Key::O)
                    && (i.modifiers.command || i.modifiers.ctrl)
                {
                    self.pick_and_load_file();
                }
                if i.key_pressed(egui::Key::F) && i.modifiers.is_none() {
                    let v = ctx.viewport_id();
                    let cur = i.viewport().fullscreen.unwrap_or(false);
                    ctx.send_viewport_cmd_to(v, egui::ViewportCommand::Fullscreen(!cur));
                }
                // I / O — mark in/out at current playhead.
                let cur = self.current_display.as_ref()
                    .map(|d| d.timestamp_s).unwrap_or(0.0);
                if i.key_pressed(egui::Key::I) && i.modifiers.is_none() {
                    pending_trim_in = Some(Some(cur));
                }
                if i.key_pressed(egui::Key::O) && i.modifiers.is_none() {
                    pending_trim_out = Some(Some(cur));
                }
            });
        }
        if let Some(t) = pending_trim_in  { self.set_trim_in(t); }
        if let Some(t) = pending_trim_out { self.set_trim_out(t); }

        // ─── Top toolbar (file + preview-size only; transport is bottom) ─
        egui::TopBottomPanel::top("toolbar")
            .exact_height(40.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(RichText::new("▶ VR180 Silver Bullet Neo")
                        .strong().color(Color32::from_rgb(90, 166, 255)));
                    ui.separator();
                    if ui.button("Open .360…").clicked() {
                        self.pick_and_load_file();
                    }

                    // Export — only enabled for fisheye sources with no
                    // export already in flight.
                    let can_export = self.export_job.is_none()
                        && matches!(
                            self.clip.as_ref().map(|c| c.source_kind),
                            Some(k) if k.is_fisheye()
                        );
                    ui.add_enabled_ui(can_export, |ui| {
                        if ui.button("Export SBS…").clicked() {
                            self.start_export();
                        }
                    });

                    ui.separator();
                    ui.label(RichText::new("Preview size").color(Color32::GRAY));
                    egui::ComboBox::from_id_source("preview_w")
                        .selected_text(format!("{}", self.settings.preview_eye_w))
                        .show_ui(ui, |ui| {
                            for &w in &[512_u32, 768, 1024, 1280, 1536] {
                                ui.selectable_value(&mut self.settings.preview_eye_w, w,
                                    format!("{w}"));
                            }
                        });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(env!("CARGO_PKG_VERSION"))
                            .small().color(Color32::GRAY));

                        // Export progress badge in the right corner —
                        // only when a job is in flight.
                        if let Some(job) = &self.export_job {
                            ui.separator();
                            if ui.small_button("Cancel").clicked() {
                                job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                            if let Some(p) = job.last_progress {
                                let pct = if p.total_frames > 0 {
                                    (p.frame_idx as f32 / p.total_frames as f32 * 100.0).min(100.0)
                                } else { 0.0 };
                                let eta = if p.fps_avg > 0.1 && p.total_frames > p.frame_idx {
                                    let remaining = (p.total_frames - p.frame_idx) as f32;
                                    let secs = remaining / p.fps_avg;
                                    format!("{:.0}s left", secs)
                                } else {
                                    "—".into()
                                };
                                ui.label(RichText::new(format!(
                                    "{pct:.1}% · {} / {} · {:.1} fps · {}",
                                    p.frame_idx, p.total_frames, p.fps_avg, eta
                                )).small().color(Color32::from_rgb(255, 176, 100)));
                            } else {
                                ui.label(RichText::new("starting…")
                                    .small().color(Color32::from_rgb(255, 176, 100)));
                            }
                        }
                    });
                });
            });

        // ─── Sidebar — settings panels, source-kind-aware ───────
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(300.0)
            .min_width(240.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                egui::CollapsingHeader::new(RichText::new("Source").strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_source_info(ui); });

                // Fisheye sources (OSV / SBS / BRAW) get the
                // camera-preset / FOV / KB panel. GoPro EAC keeps
                // the existing stabilization + RS panels.
                if matches!(kind, Some(k) if k.is_fisheye()) {
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Fisheye lens").strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_panel(ui); });
                } else {
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Stabilization").strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_stab_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Rolling shutter").strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_rs_panel(ui); });
                }
            });

        // ─── Bottom: status bar + transport ─────────────────────
        egui::TopBottomPanel::bottom("status").exact_height(24.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let status = if self.playing { "▶ playing" } else { "paused" };
                ui.label(RichText::new(status).small());
                ui.separator();
                ui.label(RichText::new(format!("{:.1} fps", self.fps_stats.last_fps))
                    .small().color(Color32::from_rgb(76, 208, 125)));
                if let Some(clip) = &self.clip {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!(
                            "{} × {} @ {:.2} fps · {} frames",
                            clip.width, clip.height, clip.fps, clip.frame_count,
                        )).small().color(Color32::GRAY));
                    });
                }
            });
        });

        // ─── Transport — buttons + trim controls ────────────────
        egui::TopBottomPanel::bottom("transport").exact_height(36.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let can_play = self.loaded_path.is_some();
                ui.add_enabled_ui(can_play, |ui| {
                    let label = if self.playing { "Pause" } else { "▶ Play" };
                    if ui.button(label).clicked() { self.toggle_play_pause(); }
                    if ui.button("■ Stop").clicked() { self.stop_playback(); }
                });
                ui.separator();

                // Time readout: current / total in mm:ss.ff format.
                let cur = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
                let total = self.clip.as_ref().map(|c| c.duration_sec).unwrap_or(0.0);
                ui.label(RichText::new(format!(
                    "{}  /  {}",
                    format_time(cur), format_time(total),
                )).monospace());

                ui.separator();
                if let Some(d) = &self.current_display {
                    let total_frames = self.clip.as_ref().map(|c| c.frame_count).unwrap_or(0);
                    ui.label(RichText::new(format!(
                        "frame {} / {}", d.frame_idx, total_frames,
                    )).small().color(Color32::GRAY));
                }

                ui.separator();
                // Trim in / out — capture current playhead.
                ui.add_enabled_ui(can_play, |ui| {
                    if ui.button("[ Mark In").on_hover_text("Set trim-in to current playhead (I)").clicked() {
                        self.set_trim_in(Some(cur));
                    }
                    if ui.button("Mark Out ]").on_hover_text("Set trim-out to current playhead (O)").clicked() {
                        self.set_trim_out(Some(cur));
                    }
                    if ui.button("Clear trim").on_hover_text("Remove both in/out points").clicked() {
                        self.set_trim_in(None);
                        self.set_trim_out(None);
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new("Space play · I in · O out · ⌘O open · F fullscreen")
                        .small().color(Color32::DARK_GRAY));
                });
            });
        });

        // ─── Timeline scrubber ──────────────────────────────────
        egui::TopBottomPanel::bottom("timeline").exact_height(34.0).show(ctx, |ui| {
            self.draw_timeline(ui);
        });

        // ─── Main preview area ──────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(d) = &self.current_display {
                let avail = ui.available_size();
                let aspect = d.width as f32 / d.height as f32;
                // Fit-to-area with aspect ratio preserved.
                let (w, h) = if avail.x / avail.y > aspect {
                    (avail.y * aspect, avail.y)
                } else {
                    (avail.x, avail.x / aspect)
                };
                let sized = egui::load::SizedTexture::new(d.egui_id, egui::vec2(w, h));
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(sized).fit_to_exact_size(egui::vec2(w, h)));
                });
            } else if self.loaded_path.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("Click ▶ Play to start preview.")
                        .size(14.0).color(Color32::GRAY));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("🎬").size(48.0));
                        ui.add_space(8.0);
                        ui.label(RichText::new("Drop a .360 file here, or click Open .360.")
                            .size(13.0).color(Color32::GRAY));
                    });
                });
            }
        });

        // Push slider edits to the running decoder.
        self.maybe_push_settings();

        // While the decoder is producing frames, keep the UI awake so
        // the next frame draws without waiting for a user input event.
        if self.playing {
            ctx.request_repaint_after(std::time::Duration::from_millis(8));
        }
    }
}

impl App {
    fn draw_source_info(&self, ui: &mut egui::Ui) {
        match (&self.loaded_path, &self.clip) {
            (Some(p), Some(c)) => {
                let fname = p.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "<no name>".into());
                ui.label(RichText::new(&fname).strong());

                // Source kind badge.
                let kind_color = if c.source_kind.is_fisheye() {
                    Color32::from_rgb(255, 176, 100) // amber
                } else if c.source_kind.is_eac() {
                    Color32::from_rgb(100, 196, 255) // sky
                } else {
                    Color32::GRAY
                };
                ui.label(RichText::new(c.source_kind.display()).color(kind_color).small());

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Stream").color(Color32::GRAY).small());
                    ui.label(format!("{} × {}", c.width, c.height));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("FPS").color(Color32::GRAY).small());
                    ui.label(format!("{:.3}", c.fps));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Duration").color(Color32::GRAY).small());
                    ui.label(format!("{:.2}s ({} frames)", c.duration_sec, c.frame_count));
                });
                if c.source_kind.is_eac() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("EAC tile").color(Color32::GRAY).small());
                        ui.label(format!("{} px", c.eac_tile_w));
                    });
                } else if c.source_kind.is_fisheye() && c.fisheye_eye_w > 0 {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Eye dims").color(Color32::GRAY).small());
                        ui.label(format!("{} × {}", c.fisheye_eye_w, c.fisheye_eye_h));
                    });
                }
                if c.segments.len() > 1 {
                    ui.label(RichText::new(format!(
                        "{} segments", c.segments.len()
                    )).small().color(Color32::from_rgb(90, 166, 255)));
                }
                if c.cori_count > 0 {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("CORI").color(Color32::GRAY).small());
                        ui.label(format!("{} samples", c.cori_count));
                    });
                }
                if let Some(s) = c.srot_ms {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("SROT").color(Color32::GRAY).small());
                        ui.label(format!("{:.3} ms", s));
                    });
                }
            }
            _ => {
                ui.label(RichText::new("No file loaded.").color(Color32::GRAY));
            }
        }
    }

    fn draw_stab_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.checkbox(&mut s.stabilize, "Enable stabilization");
        ui.add_enabled_ui(s.stabilize, |ui| {
            ui.horizontal(|ui| {
                ui.label("CORI source");
                egui::ComboBox::from_id_source("cori_src")
                    .selected_text(s.cori_source.as_str())
                    .show_ui(ui, |ui| {
                        use crate::decoder::CoriSource as C;
                        ui.selectable_value(&mut s.cori_source, C::Direct, "direct");
                        ui.selectable_value(&mut s.cori_source, C::Vqf, "vqf");
                        ui.selectable_value(&mut s.cori_source, C::Auto, "auto");
                    });
            });
            ui.add(egui::Slider::new(&mut s.smooth_ms, 0.0..=3000.0).text("Smooth (ms)"));
            ui.add(egui::Slider::new(&mut s.max_corr_deg, 0.0..=45.0).text("Max corr (°)"));
        });
        ui.add_space(6.0);
        ui.label(RichText::new(
            "Settings apply live during playback — the decoder rebuilds \
             per-eye bundles on the next iteration (small stutter ~300 ms \
             on long clips)."
        ).small().color(Color32::GRAY));
    }

    fn draw_rs_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.checkbox(&mut s.rs_correct, "Enable RS correction");
        ui.add_enabled_ui(s.rs_correct, |ui| {
            ui.horizontal(|ui| {
                ui.label("Mode");
                egui::ComboBox::from_id_source("rs_mode")
                    .selected_text(s.rs_mode.as_str())
                    .show_ui(ui, |ui| {
                        use crate::decoder::RsMode as R;
                        ui.selectable_value(&mut s.rs_mode, R::Firmware, "firmware");
                        ui.selectable_value(&mut s.rs_mode, R::NoFirmware, "no-firmware");
                    });
            });
            ui.add(egui::Slider::new(&mut s.rs_readout_ms, 5.0..=33.0)
                .text("SROT (ms)").fixed_decimals(3));
        });
    }

    /// Settings panel for fisheye sources (OSV / SBS / BRAW). Mirrors
    /// the Python app's lens parameters at vr180_gui.py:5500+. All
    /// sliders apply live during playback — bumping
    /// `settings_generation` makes the decoder re-resolve the
    /// `FisheyeCalib` on the next frame.
    fn draw_fisheye_panel(&mut self, ui: &mut egui::Ui) {
        let presets = vr180_fisheye::presets::presets();
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        let is_braw = matches!(kind, Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        let is_osv  = matches!(kind, Some(vr180_pipeline::SourceKind::DjiOsv));
        let s = &mut self.settings;

        // ── Stabilization (BRAW VQF + DJI OSV direct-quat). SBS
        //    fisheye is identity until per-camera gyro parsers land
        //    (Insta360 / Vuze / Canon each different).
        if is_braw || is_osv {
            let stab_label = if is_braw {
                "Stabilization (VQF 6D, BRAW)"
            } else {
                "Stabilization (DJI camera quats)"
            };
            ui.checkbox(&mut s.stabilize, stab_label);
            ui.add_enabled_ui(s.stabilize, |ui| {
                if is_osv {
                    // OSV is hard-locked to pure camera-lock: target
                    // is frame 0 for every frame, no per-frame
                    // correction cap. Neither slider applies here, so
                    // both are hidden — only the mode label.
                    ui.label(RichText::new(
                        "Mode: camera-lock (frame 0, no cap)"
                    ).small().color(Color32::from_rgb(180, 180, 180)));
                }
                // BRAW VQF currently has no smoothing/cap knobs —
                // the per-frame quat IS the smoothed output of VQF
                // itself. Future task could add a post-VQF EMA pass.
            });
            ui.label(RichText::new(
                "Stab toggle is live during playback — first time \
                 you flip it on, gyro extraction runs (~1 s stutter). \
                 Sliders apply live too."
            ).small().color(Color32::GRAY));
            ui.separator();
        }

        // ── Preset dropdown ─────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Camera");
            let current = if s.fisheye_preset.is_empty() {
                "(Auto)".to_string()
            } else {
                s.fisheye_preset.clone()
            };
            egui::ComboBox::from_id_source("fisheye_preset")
                .selected_text(current)
                .width(200.0)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(s.fisheye_preset.is_empty(), "(Auto)").clicked() {
                        s.fisheye_preset.clear();
                    }
                    for p in presets {
                        let selected = s.fisheye_preset == p.name;
                        if ui.selectable_label(selected, p.name).clicked() {
                            s.fisheye_preset = p.name.to_string();
                            // Reset overrides so the preset's values
                            // take effect immediately. The user can
                            // then nudge the FOV / KB sliders.
                            s.fisheye_fov_deg = 0.0;
                            s.fisheye_k = [0.0; 4];
                            s.fisheye_cx_offset_px = 0.0;
                            s.fisheye_cy_offset_px = 0.0;
                        }
                    }
                });
        });

        // ── FOV / lens-offset / KB sliders ──────────────────────
        ui.add_space(2.0);

        // FOV override. 0 = use preset default; otherwise this value wins.
        let mut fov_enabled = s.fisheye_fov_deg > 0.0;
        ui.checkbox(&mut fov_enabled, "Override FOV");
        if fov_enabled && s.fisheye_fov_deg <= 0.0 {
            // Seed with the preset's default.
            s.fisheye_fov_deg = presets
                .iter().find(|p| p.name == s.fisheye_preset)
                .map(|p| p.default_fov_deg as f32)
                .unwrap_or(180.0);
        } else if !fov_enabled {
            s.fisheye_fov_deg = 0.0;
        }
        ui.add_enabled_ui(fov_enabled, |ui| {
            ui.add(egui::Slider::new(&mut s.fisheye_fov_deg, 90.0..=230.0)
                .text("Full FOV (°)").fixed_decimals(2));
        });

        // Lens offset sliders.
        ui.collapsing("Lens center offset", |ui| {
            ui.add(egui::Slider::new(&mut s.fisheye_cx_offset_px, -200.0..=200.0)
                .text("cx Δpx").fixed_decimals(1));
            ui.add(egui::Slider::new(&mut s.fisheye_cy_offset_px, -200.0..=200.0)
                .text("cy Δpx").fixed_decimals(1));
        });

        // KB k1-k4 sliders (advanced).
        ui.collapsing("KB distortion (k1–k4)", |ui| {
            let mut k_override = s.fisheye_k.iter().any(|c| c.abs() > 1e-9);
            ui.checkbox(&mut k_override, "Override preset k");
            if k_override && s.fisheye_k.iter().all(|c| c.abs() < 1e-9) {
                // Seed from the preset.
                if let Some(p) = presets.iter().find(|p| p.name == s.fisheye_preset) {
                    for i in 0..4 {
                        s.fisheye_k[i] = p.calib.k[i] as f32;
                    }
                }
            } else if !k_override {
                s.fisheye_k = [0.0; 4];
            }
            ui.add_enabled_ui(k_override, |ui| {
                for (i, k) in s.fisheye_k.iter_mut().enumerate() {
                    ui.add(egui::Slider::new(k, -0.5..=0.5)
                        .text(format!("k{}", i + 1)).fixed_decimals(6));
                }
            });
        });

        // ── Swap eyes (OSV only) ────────────────────────────────
        let is_osv = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
        );
        if is_osv {
            ui.add_space(4.0);
            ui.checkbox(&mut s.fisheye_swap_eyes, "Swap L↔R eyes");
        }

        // ── Output mode (fisheye→fisheye is Task #10 pending) ───
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Output");
            egui::ComboBox::from_id_source("fisheye_output_mode")
                .selected_text(s.fisheye_output_mode.as_str())
                .show_ui(ui, |ui| {
                    use crate::decoder::FisheyeOutputMode as M;
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::HalfEquirect, "half-equirect");
                    ui.add_enabled(false,
                        egui::SelectableLabel::new(
                            s.fisheye_output_mode == M::Fisheye,
                            "fisheye (Task #10)"
                        )
                    );
                });
        });

        // ── Load Gyroflow lens profile ──────────────────────────
        ui.add_space(6.0);
        if ui.button("Load Gyroflow lens profile…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Gyroflow lens profile (.json)", &["json", "JSON"])
                .pick_file()
            {
                match vr180_fisheye::GyroflowLensProfile::load(&path) {
                    Ok(prof) => match prof.to_calibration() {
                        Ok(cal) => {
                            // Push the calibration's KB into settings (the
                            // FOV will be derived per the current preset).
                            for i in 0..4 {
                                s.fisheye_k[i] = cal.k[i] as f32;
                            }
                            // Compute approximate FOV from the rim radius
                            // (if the JSON specifies a calib_dimension).
                            if cal.calib_w > 0 {
                                let r_max = (cal.calib_w.min(cal.calib_h) as f64) * 0.5;
                                let fov_rad = cal.full_fov_from_rim(r_max);
                                s.fisheye_fov_deg = fov_rad.to_degrees() as f32;
                            }
                            tracing::info!(
                                "loaded Gyroflow lens profile: {} — fov≈{:.2}°, k={:?}",
                                path.display(), s.fisheye_fov_deg, s.fisheye_k
                            );
                        }
                        Err(e) => tracing::error!(
                            "Gyroflow lens profile invalid: {e}"
                        ),
                    },
                    Err(e) => tracing::error!(
                        "load Gyroflow JSON {}: {e}", path.display()
                    ),
                }
            }
        }

        ui.add_space(6.0);
        ui.label(RichText::new(
            "Camera presets autoload on file open. Override the FOV \
             or k coefficients above for non-standard lenses, or load \
             a Gyroflow lens profile for community-maintained \
             calibrations."
        ).small().color(Color32::GRAY));
    }
}

/// Format seconds as `mm:ss.ff` (frame-fraction-style, 2 decimals).
fn format_time(sec: f64) -> String {
    if !sec.is_finite() || sec < 0.0 {
        return "--:--.--".to_string();
    }
    let total = sec.max(0.0);
    let m = (total / 60.0) as u32;
    let s = total - (m as f64) * 60.0;
    format!("{:02}:{:05.2}", m, s)
}
