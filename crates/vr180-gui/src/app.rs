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
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
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
            .add_filter("GoPro 360 / MP4", &["360", "mp4", "MP4", "mov", "MOV"])
            .pick_file();
        if let Some(p) = path { self.load_file(p); }
    }

    fn load_file(&mut self, path: PathBuf) {
        // Stop any running playback before loading a new clip.
        self.stop_playback();

        let probe = match vr180_pipeline::decode::probe_video(&path) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("probe failed: {e}");
                return;
            }
        };
        let dims = vr180_core::eac::Dims::new(probe.width, probe.height);

        // Optional: parse GPMF + GEOC for metadata display.
        let (cori_count, grav_count, srot_ms) =
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
            };

        let segments = vr180_core::segments::detect_segments(&path);

        self.clip = Some(ClipInfo {
            width: probe.width, height: probe.height,
            fps: probe.fps, duration_sec: probe.duration_sec,
            frame_count: (probe.duration_sec * probe.fps as f64).round() as u32,
            segments,
            eac_tile_w: dims.tile_w(),
            cori_count, grav_count, srot_ms,
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

        // Handle dropped files (drag-and-drop into the window).
        ctx.input(|i| {
            for f in &i.raw.dropped_files {
                if let Some(p) = f.path.clone() {
                    self.load_file(p);
                    break;
                }
            }
        });

        // ─── Top toolbar ────────────────────────────────────────
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
                    let can_play = self.loaded_path.is_some();
                    let btn_label = if self.playing { "Pause" } else { "▶ Play" };
                    ui.add_enabled_ui(can_play, |ui| {
                        if ui.button(btn_label).clicked() {
                            self.toggle_play_pause();
                        }
                        if ui.button("■ Stop").clicked() {
                            self.stop_playback();
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
                    });
                });
            });

        // ─── Sidebar — stab / RS / display knobs ────────────────
        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(300.0)
            .min_width(240.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                egui::CollapsingHeader::new(RichText::new("Source").strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_source_info(ui); });

                ui.add_space(4.0);
                egui::CollapsingHeader::new(RichText::new("Stabilization").strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_stab_panel(ui); });

                ui.add_space(4.0);
                egui::CollapsingHeader::new(RichText::new("Rolling shutter").strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_rs_panel(ui); });
            });

        // ─── Bottom: status bar ─────────────────────────────────
        egui::TopBottomPanel::bottom("status").exact_height(24.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let status = if self.playing { "▶ playing" } else { "paused" };
                ui.label(RichText::new(status).small());
                ui.separator();
                if let Some(d) = &self.current_display {
                    ui.label(RichText::new(format!(
                        "frame {} · t={:.2}s · {} × {}",
                        d.frame_idx, d.timestamp_s, d.width, d.height,
                    )).small().color(Color32::GRAY));
                }
                ui.separator();
                ui.label(RichText::new(format!("{:.1} fps", self.fps_stats.last_fps))
                    .small().color(Color32::from_rgb(76, 208, 125)));
                if let Some(clip) = &self.clip {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!(
                            "{} × {} @ {:.2} fps · {:.1}s",
                            clip.width, clip.height, clip.fps, clip.duration_sec,
                        )).small().color(Color32::GRAY));
                    });
                }
            });
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
                ui.horizontal(|ui| {
                    ui.label(RichText::new("EAC tile").color(Color32::GRAY).small());
                    ui.label(format!("{} px", c.eac_tile_w));
                });
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
}
