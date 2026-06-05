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
    /// Last settings value written to disk, and when — used to debounce
    /// the persisted-settings save so a slider drag doesn't hammer disk.
    last_saved_settings: Settings,
    last_settings_save_at: std::time::Instant,

    // ─── Export job (fisheye sources only for now) ───────────────
    /// `Some` while an export is in flight. `None` once the worker
    /// finishes (cleanly or via cancel) — the GUI re-enables the
    /// Export button.
    export_job: Option<ExportJob>,

    /// Audio playback for the preview. `Some` when the clip's source
    /// has an audio track. Tied to the same play / pause / seek
    /// commands as the video decoder.
    audio_player: Option<crate::audio_player::AudioPlayer>,

    /// User-chosen codec/bitrate/metadata for the next export. Shown
    /// in a floating window when the user clicks Export…
    export_opts: ExportOptions,
    /// True while the export-options window is open.
    export_opts_visible: bool,

    /// Preview magnifier for alignment inspection. `preview_zoom` is the
    /// magnification (1.0 = fit-to-window); `preview_center` is the point
    /// of the image kept centered, in UV [0,1]. Scroll zooms toward the
    /// cursor, drag pans, double-click resets.
    preview_zoom: f32,
    preview_center: egui::Vec2,

    /// Full-resolution still for zoomed alignment: while paused + zoomed,
    /// the current frame is re-rendered at native source resolution on a
    /// worker thread so the magnifier shows real detail (the live preview
    /// is decoded at a capped working size). `full_res_display` holds the
    /// registered hi-res texture; `full_res_key` identifies the (frame,
    /// settings) it was rendered for; the receiver carries an in-flight
    /// job's result.
    full_res_display: Option<DisplayFrame>,
    /// Key (frame+settings) the current `full_res_display` was rendered for.
    full_res_key: u64,
    /// Key (frame+settings) we currently *want* a still for. While this
    /// differs from `full_res_key` the still is stale (mid drag/scrub) and
    /// the zoom view shows the live preview instead — which is responsive
    /// (decoder thread) and now frame-aligned with the still, so the swap on
    /// settle doesn't shift.
    full_res_desired_key: u64,
    /// Settle timer: the most recent (frame+settings) key and when it last
    /// changed. The native still is rendered only once the key has been
    /// stable for the settle window, so a drag/scrub shows the live preview
    /// throughout and the heavy render fires just once on settle.
    detail_last_key: u64,
    detail_key_changed_at: std::time::Instant,
    /// Native-res still renderer (synchronous, main-thread). Created on
    /// entering detail mode (paused + zoomed on a fisheye clip), dropped on
    /// leaving. Caches the decoded native frame so alignment tweaks only
    /// re-project (fast). NOT a background thread — see `DetailCache`.
    detail_cache: Option<crate::decoder::DetailCache>,
}

/// Codec choice for the export pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportCodec { H265, ProRes }

impl ExportCodec {
    fn label(self) -> &'static str {
        match self {
            ExportCodec::H265 => "H.265 (HEVC)",
            ExportCodec::ProRes => "ProRes",
        }
    }
}

/// ProRes profiles. Numbers match ffmpeg's `-profile:v N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProResProfile {
    Proxy = 0,
    Lt = 1,
    Standard = 2,
    Hq = 3,
    P4444 = 4,
    P4444Xq = 5,
}

impl ProResProfile {
    fn label(self) -> &'static str {
        match self {
            ProResProfile::Proxy    => "Proxy",
            ProResProfile::Lt       => "LT",
            ProResProfile::Standard => "Standard",
            ProResProfile::Hq       => "HQ",
            ProResProfile::P4444    => "4444",
            ProResProfile::P4444Xq  => "4444 XQ",
        }
    }
}

#[derive(Debug, Clone)]
struct ExportOptions {
    codec: ExportCodec,
    /// H.265 average bitrate in Mbps. Range 20..=500.
    h265_bitrate_mbps: u32,
    /// 8 or 10 (Main / Main10) for H.265.
    h265_bit_depth: u8,
    prores_profile: ProResProfile,
    /// Inject Apple Projected Media Profile (Vision Pro VR180).
    inject_apmp: bool,
    /// Inject Spatial Media V2 (YouTube VR180) — st3d + sv3d boxes.
    /// MUTUALLY EXCLUSIVE with `inject_apmp` (they overwrite the same
    /// atoms in the visual sample entry) — the GUI enforces it.
    inject_youtube: bool,
    /// Camera baseline in mm for APMP `cams/blin`. Typical 60..=70.
    apmp_baseline_mm: f32,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            codec: ExportCodec::H265,
            h265_bitrate_mbps: 200,
            h265_bit_depth: 10,
            prores_profile: ProResProfile::Hq,
            // APMP is the headline target (Vision Pro). They're
            // mutually exclusive; the dialog enforces it.
            inject_apmp: true,
            inject_youtube: false,
            apmp_baseline_mm: 65.0,
        }
    }
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

        // Restore the user's last-used settings from disk (defaults on first
        // run). Persisted across launches so every knob is remembered.
        let defaults = Settings::load_persisted();
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
            last_saved_settings: defaults.clone(),
            last_settings_save_at: std::time::Instant::now(),
            last_pushed_settings: defaults,
            export_job: None,
            audio_player: None,
            export_opts: ExportOptions::default(),
            export_opts_visible: false,
            preview_zoom: 1.0,
            preview_center: egui::vec2(0.5, 0.5),
            full_res_display: None,
            full_res_key: 0,
            full_res_desired_key: 0,
            detail_last_key: 0,
            detail_key_changed_at: std::time::Instant::now(),
            detail_cache: None,
        }
    }

    /// Pick an output path and spawn the export worker. No-op if no
    /// clip is loaded, an export is already in flight, or the source
    /// is GoPro EAC (the existing CLI handles that family).
    /// Top-toolbar Export button — opens the options window. The actual
    /// export kicks off from `commit_export()` once the user confirms.
    fn start_export(&mut self) {
        if self.export_job.is_some() { return; }
        if !matches!(self.clip.as_ref().map(|c| c.source_kind),
            Some(k) if k.is_fisheye())
        {
            tracing::warn!("export: only fisheye sources supported");
            return;
        }
        self.export_opts_visible = true;
    }

    /// Called from the export-options window's "Start Export…" button.
    /// Pops up the save-file dialog, builds the FisheyeExportConfig
    /// from `self.export_opts`, and spawns the encode worker.
    fn commit_export(&mut self) {
        self.export_opts_visible = false;
        if self.export_job.is_some() { return; }
        let (path, clip) = match (&self.loaded_path, &self.clip) {
            (Some(p), Some(c)) => (p.clone(), c.clone()),
            _ => return,
        };
        if !clip.source_kind.is_fisheye() {
            return;
        }

        // Default output path: same dir, _SBS.mp4 / .mov suffix.
        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext_default = match self.export_opts.codec {
            ExportCodec::H265 => "mp4",
            ExportCodec::ProRes => "mov", // ProRes is conventional in MOV
        };
        let default_out = path.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_SBS.{ext_default}"));

        let chosen = rfd::FileDialog::new()
            .add_filter(
                match self.export_opts.codec {
                    ExportCodec::H265 => "H.265 video",
                    ExportCodec::ProRes => "ProRes QuickTime",
                },
                &["mp4", "mov"],
            )
            .set_file_name(default_out.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output.mp4".into()))
            .set_directory(default_out.parent().unwrap_or(std::path::Path::new(".")))
            .save_file();
        let Some(output_path) = chosen else { return; };

        // Pick the backend based on the chosen codec + platform.
        let backend = match (self.export_opts.codec, cfg!(target_os = "macos")) {
            (ExportCodec::H265, true) =>
                vr180_pipeline::encode::EncoderBackend::VideoToolbox,
            (ExportCodec::H265, false) =>
                vr180_pipeline::encode::EncoderBackend::Libx265,
            (ExportCodec::ProRes, true) =>
                vr180_pipeline::encode::EncoderBackend::ProResVideoToolbox,
            (ExportCodec::ProRes, false) =>
                vr180_pipeline::encode::EncoderBackend::ProResKs,
        };

        // Output resolution depends on the projection target. For
        // half-equirect VR180 we use the source eye dims (or fallback
        // for non-fisheye sources). For fisheye pass-through we use
        // the raw source eye dims since the projection is bypassed.
        let (mut eye_w, mut eye_h) = if clip.fisheye_eye_w > 0 {
            (clip.fisheye_eye_w, clip.fisheye_eye_h)
        } else {
            (clip.width.max(512), clip.height.max(512))
        };

        let projection = match self.settings.fisheye_output_mode {
            crate::decoder::FisheyeOutputMode::HalfEquirect =>
                vr180_pipeline::fisheye_export::FisheyeExportProjection::HalfEquirect,
            crate::decoder::FisheyeOutputMode::Fisheye =>
                vr180_pipeline::fisheye_export::FisheyeExportProjection::Fisheye,
        };
        // Fisheye output is a CIRCLE in a SQUARE frame — force eye_h = eye_w
        // so SBS becomes (2*side × side). For OSV (3840×3840) this is a
        // no-op; for non-square sources it picks the smaller side so
        // the fisheye disk fits.
        if projection == vr180_pipeline::fisheye_export::FisheyeExportProjection::Fisheye {
            let side = eye_w.min(eye_h);
            eye_w = side;
            eye_h = side;
        }
        let color_stack = self.settings.build_color_stack();

        // Bitrate / bit-depth come from the options window. ProRes
        // ignores the bitrate field (the profile picks the rate)
        // and is always 10-bit / 12-bit per profile.
        let (bitrate_kbps, bit_depth) = match self.export_opts.codec {
            ExportCodec::H265 => (
                self.export_opts.h265_bitrate_mbps.saturating_mul(1000),
                self.export_opts.h265_bit_depth,
            ),
            ExportCodec::ProRes => {
                // The encoder ignores bitrate for ProRes but the config
                // field is non-optional — pass a sane default.
                let pix_is_12bit = matches!(
                    self.export_opts.prores_profile,
                    ProResProfile::P4444 | ProResProfile::P4444Xq,
                );
                (0, if pix_is_12bit { 10 } else { 10 })
            }
        };

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
            prores_profile: self.export_opts.prores_profile as i32,
            inject_apmp: self.export_opts.inject_apmp,
            inject_youtube_vr180: self.export_opts.inject_youtube,
            apmp_baseline_mm: self.export_opts.apmp_baseline_mm,
            stabilize: self.settings.stabilize,
            dji_max_corr_deg: self.settings.dji_max_corr_deg,
            dji_smooth_ms: self.settings.dji_smooth_ms,
            view_adjust: vr180_pipeline::panomap::ViewAdjust {
                pano_yaw_deg: self.settings.pano_yaw_deg,
                pano_pitch_deg: self.settings.pano_pitch_deg,
                pano_roll_deg: self.settings.pano_roll_deg,
                stereo_yaw_deg: self.settings.stereo_yaw_deg,
                stereo_pitch_deg: self.settings.stereo_pitch_deg,
                stereo_roll_deg: self.settings.stereo_roll_deg,
            },
            fisheye_preset: self.settings.fisheye_preset.clone(),
            fisheye_override_left: self.settings.fisheye_override_left,
            fisheye_override_right: self.settings.fisheye_override_right,
            fisheye_fov_deg_left: self.settings.fisheye_fov_deg_left,
            fisheye_fov_deg_right: self.settings.fisheye_fov_deg_right,
            fisheye_k_left: self.settings.fisheye_k_left,
            fisheye_k_right: self.settings.fisheye_k_right,
            fisheye_cx_norm_left: self.settings.fisheye_cx_norm_left,
            fisheye_cy_norm_left: self.settings.fisheye_cy_norm_left,
            fisheye_cx_norm_right: self.settings.fisheye_cx_norm_right,
            fisheye_cy_norm_right: self.settings.fisheye_cy_norm_right,
            fisheye_swap_eyes: self.settings.fisheye_swap_eyes,
            trim_in_s: self.settings.trim_in_s,
            trim_out_s: self.settings.trim_out_s,
            color_stack,
            projection,
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

    /// Drive the full-resolution still used by the zoom magnifier. While
    /// paused + zoomed on a fisheye clip, the current frame is decoded +
    /// re-projected at native source resolution and shown in the magnifier
    /// (the live preview decodes at a capped working size). No-op otherwise.
    ///
    /// The native render runs SYNCHRONOUSLY on this (main) thread — see
    /// `DetailCache` for why a background thread deadlocks against eframe's
    /// renderer on the shared Metal device. To hide the resulting one-frame
    /// hitch it is DEBOUNCED: it fires only once the desired (frame,
    /// settings) key has been stable for the settle window. During an active
    /// drag/scrub the key keeps changing, so the preview shows the live
    /// low-res frame and we skip the heavy render until you stop.
    fn poll_full_res(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        // Keep the detail cache alive whenever we're paused on a fisheye clip
        // — independent of zoom. Zoom only gates whether we RENDER a still,
        // not whether the cache (and its decoded native frame) exists. That
        // way zooming out and back in reuses the already-decoded frame
        // instead of re-decoding it (a GOP walk at native res), which showed
        // up as a fresh slowdown on every zoom-in.
        let keep_alive = self.decoder_alive
            && !self.playing
            && self.clip.as_ref().map(|c| c.source_kind.is_fisheye()).unwrap_or(false);

        if !keep_alive {
            if let Some(prev) = self.full_res_display.take() {
                self.egui_renderer.write().free_texture(&prev.egui_id);
            }
            self.full_res_key = 0;
            self.full_res_desired_key = 0;
            self.detail_last_key = 0;
            self.detail_cache = None; // close the native decoder
            return;
        }

        let (path, kind, fps) = match (&self.loaded_path, &self.clip) {
            (Some(p), Some(c)) => (p.clone(), c.source_kind, c.fps),
            _ => return,
        };
        // Spin up the native-res renderer on entering detail mode. It owns a
        // background decode thread; `ctx` lets that thread poke a repaint
        // when a frame finishes decoding so the still appears promptly.
        if self.detail_cache.is_none() {
            self.detail_cache = Some(crate::decoder::DetailCache::new(
                path, kind, fps, self.settings.fisheye_swap_eyes, ctx.clone()));
            self.full_res_key = 0;
            self.full_res_desired_key = 0;
            self.detail_last_key = 0;
        }

        // Only render a still while zoomed in. When zoomed out we keep the
        // cache + the last rendered still (the 1× view uses the live
        // preview), so returning here makes zoom-in instant when nothing
        // changed — and a cheap re-projection (no decode) if settings moved.
        if self.preview_zoom <= 1.001 {
            return;
        }

        // Key identifies the (frame, settings) we want a still for. Zoom is
        // excluded — one still covers the whole frame; the UV sub-rect
        // samples it.
        let ts = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
        let gen = self.control.as_ref()
            .map(|c| c.settings_generation.load(Ordering::SeqCst)).unwrap_or(0);
        let ts_ms = (ts * 1000.0).round() as i64 as u64;
        let key = ts_ms.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(gen).max(1);

        self.full_res_desired_key = key;

        // Settle timer: reset whenever the wanted (frame+settings) changes.
        // We render the native still ONLY once the key has been stable for
        // the window. While you're dragging a slider (or scrubbing) the key
        // keeps moving, so `full_res_key != full_res_desired_key` and the
        // zoom view shows the live preview — which renders on the DECODER
        // thread (parallel, non-blocking → responsive) and is now frame-
        // aligned with the still, so the swap on settle doesn't shift.
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(150);
        if key != self.detail_last_key {
            self.detail_last_key = key;
            self.detail_key_changed_at = std::time::Instant::now();
        }

        // Already showing this exact key → done.
        if key == self.full_res_key {
            return;
        }
        // Not settled yet → leave the live preview up; wake when it settles.
        if self.detail_key_changed_at.elapsed() < SETTLE {
            ctx.request_repaint_after(SETTLE);
            return;
        }

        // Settled. `render` now does the GPU project + compose on this thread
        // (~10 ms) IF the native frame is already decoded; otherwise it kicks
        // the background decode and returns None (cheap). So calling it every
        // poll while pending is fine — no heavy work on the main thread until
        // the frame is ready. On None we poll again (the worker also pokes a
        // repaint the instant it finishes decoding, so there's no needless
        // wait). This also means a transient decode failure simply retries
        // instead of latching "no still".
        let pipeline = self.pipeline.clone(); // Arc — avoids borrowing self twice
        let settings = self.settings.clone();
        let rendered = self.detail_cache.as_mut().unwrap()
            .render(&pipeline, ts, &settings, 0);
        if rendered.is_none() {
            // Decoding (or a transient failure) — keep the live preview up
            // and poll again shortly.
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        if let Some((tex, w, h)) = rendered {
            let mut rend = self.egui_renderer.write();
            if let Some(prev) = self.full_res_display.take() {
                rend.free_texture(&prev.egui_id);
            }
            // sRGB view so egui decodes our Rec.709-gamma SBS to linear on
            // sample (egui re-applies the OETF) — prevents the double-gamma
            // that made the preview look washed out vs the export.
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..Default::default()
            });
            let id = rend.register_native_texture(
                &self.pipeline.device, &view, wgpu::FilterMode::Linear);
            drop(rend);
            self.full_res_display = Some(DisplayFrame {
                texture: Arc::new(tex), egui_id: id,
                width: w, height: h,
                frame_idx: 0, timestamp_s: ts,
            });
            self.full_res_key = key;
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
        // sRGB view so egui decodes our Rec.709-gamma SBS to linear on
        // sample (egui re-applies the OETF) — prevents the double-gamma that
        // made the preview look washed out vs the export.
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });
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
                "All supported video",
                &["360", "osv", "OSV", "braw", "BRAW",
                  "mp4", "MP4", "mov", "MOV"],
            )
            .add_filter("DJI Osmo 360 (.osv)", &["osv", "OSV"])
            .add_filter("Blackmagic RAW (.braw)", &["braw", "BRAW"])
            .add_filter("Side-by-side fisheye (.mp4 / .mov)",
                &["mp4", "MP4", "mov", "MOV"])
            .pick_file();
        if let Some(p) = path { self.load_file(p); }
    }

    fn load_file(&mut self, path: PathBuf) {
        let t_load = std::time::Instant::now();
        // ── Tear down everything tied to the previous clip ──────────
        // 1. Stop the decoder thread + drop the IPC channels.
        self.stop_playback();
        tracing::info!("load_timing: after stop_playback @ {:?}", t_load.elapsed());
        // 2. Free the egui-registered preview texture from the old
        //    clip. Without this the previous frame stays painted in
        //    the central panel until the new decoder produces output.
        if let Some(prev) = self.current_display.take() {
            let mut renderer = self.egui_renderer.write();
            renderer.free_texture(&prev.egui_id);
            // Dropping `prev.texture` (the Arc<wgpu::Texture>) below
            // releases the GPU memory when the last reference goes.
        }
        // 3. Reset volatile playback state. Everything in `Settings` is now
        //    REMEMBERED across loads (and across launches, via
        //    `Settings::save_persisted`) — stabilization, view-adjust,
        //    per-eye override / calibration, output mode, color, LUT — so a
        //    new clip keeps your last setup. The only exception is the trim
        //    range, which is clip-specific (time offsets) and would be
        //    meaningless on a different clip. Per-clip camera defaults
        //    (preset + built-in LUT) below only fill in still-unset values,
        //    so an explicit choice always wins.
        self.fps_stats = FpsStats::default();
        self.settings.trim_in_s = None;
        self.settings.trim_out_s = None;
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
        tracing::info!("load_timing: after probe_video @ {:?}", t_load.elapsed());
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
        // Per-camera default preset — only when none is set yet, so a
        // remembered/explicit choice is preserved.
        if source_kind != vr180_pipeline::SourceKind::GoProEac
            && self.settings.fisheye_preset.is_empty()
        {
            let auto = match source_kind {
                vr180_pipeline::SourceKind::DjiOsv        => "DJI Osmo 360",
                vr180_pipeline::SourceKind::BlackmagicRaw => "Blackmagic Pyxis 12K",
                _                                         => "Custom",
            };
            self.settings.fisheye_preset = auto.to_string();
        }
        // Autoload the embedded DJI Osmo 360 "D-Log M → Rec.709" LUT for OSV
        // footage so the preview and export show corrected Rec.709 color by
        // default — but only when no LUT is set, so a remembered choice (a
        // different LUT, or a deliberately cleared one within a session) is
        // preserved. The user can clear or swap it in the Color panel.
        if source_kind == vr180_pipeline::SourceKind::DjiOsv
            && self.settings.lut_path.is_empty()
        {
            self.settings.lut_path = crate::decoder::BUILTIN_OSMO_LUT_PATH.to_string();
            self.settings.lut_intensity = 1.0;
        }
        // Snapshot settings so the next spawn_decoder doesn't bump
        // generation unnecessarily on the first frame.
        self.last_pushed_settings = self.settings.clone();

        let segments = vr180_core::segments::detect_segments(&path);
        tracing::info!("load_timing: after detect_segments @ {:?}", t_load.elapsed());

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
        // Kick off the decoder in paused state right away so the first
        // frame paints without forcing the user to press Play. The pause
        // loop in the decoder is wired to skip its sleep on the FIRST
        // pulled pair, so the user sees the clip's opening frame and
        // can immediately use the view-adjust / stabilization sliders.
        self.spawn_decoder(true);
        tracing::info!("load_timing: load_file TOTAL @ {:?}", t_load.elapsed());
    }

    // ─── Playback control ────────────────────────────────────────

    /// Spawn a fresh decoder thread. Used when starting playback on a
    /// freshly loaded clip (or when explicitly Stop'd and restarted).
    /// `start_paused = true` puts the decoder in paused state from the
    /// outset — the worker still pulls the first pair and the main
    /// loop still renders it once (because the pause loop is skipped
    /// when a new pair has just been pulled), so the user sees the
    /// first frame without having to press Play.
    fn spawn_decoder(&mut self, start_paused: bool) {
        let Some(path) = self.loaded_path.clone() else {
            tracing::warn!("spawn_decoder: no path loaded");
            return;
        };
        self.stop_playback();
        tracing::info!(
            "spawn_decoder: starting decoder for {} (paused={})",
            path.display(), start_paused
        );
        let cfg = DecoderConfig {
            path,
            settings: self.settings.clone(),
            eye_w: self.settings.preview_eye_w,
        };
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let control = Arc::new(DecoderControl {
            paused: std::sync::atomic::AtomicBool::new(start_paused),
            settings: parking_lot::RwLock::new(self.settings.clone()),
            settings_generation: std::sync::atomic::AtomicU64::new(0),
            detected_calib: parking_lot::Mutex::new(None),
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
        self.playing = !start_paused;
        self.decoder_alive = true;
        self.last_pushed_settings = self.settings.clone();
        self.fps_stats = FpsStats::default();

        // Spawn the audio player alongside the video decoder. It runs
        // its own thread + cpal output stream and is held by the App
        // for the lifetime of the playback session.
        if let Some(audio_path) = self.loaded_path.clone() {
            let t_audio = std::time::Instant::now();
            self.audio_player = match crate::audio_player::AudioPlayer::open(
                audio_path, start_paused,
            ) {
                Ok(Some(player)) => Some(player),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("audio_player: open failed: {e}");
                    None
                }
            };
            tracing::info!("load_timing: AudioPlayer::open @ {:?}", t_audio.elapsed());
        }
    }

    /// Toggle play/pause. If no decoder is alive yet, this kicks off
    /// a fresh one. Otherwise it just flips the paused flag — the
    /// decoder thread keeps its position so resume picks up
    /// exactly where pause left it.
    fn toggle_play_pause(&mut self) {
        if !self.decoder_alive {
            self.spawn_decoder(false); // Play button → start playing
            return;
        }
        if let Some(ctl) = &self.control {
            self.playing = !self.playing;
            ctl.paused.store(!self.playing, Ordering::SeqCst);
            tracing::info!("toggle_play_pause: playing={}", self.playing);
        }
        if let Some(player) = &self.audio_player {
            player.set_playing(self.playing);
        }
    }

    /// Send a Seek command to the running decoder. If no decoder is
    /// alive, lazily spawns one (the seek then becomes the initial
    /// position via the regular pipeline).
    fn seek_to(&mut self, target_s: f64) {
        if !self.decoder_alive {
            self.spawn_decoder(false);
        }
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecoderCommand::Seek(target_s.max(0.0)));
        }
        if let Some(player) = &self.audio_player {
            player.seek_to(target_s.max(0.0));
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
        // Drop the audio player: cpal stream stops, worker thread
        // shuts down (Stop command).
        self.audio_player = None;
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

    /// Persist settings to disk when they change. Debounced: writes at most
    /// every ~1.2 s during continuous edits (a slider drag), and flushes
    /// IMMEDIATELY when the window is closing so a last-moment tweak isn't
    /// lost (eframe's own storage is disabled here).
    fn persist_settings(&mut self, ctx: &egui::Context) {
        if self.settings == self.last_saved_settings {
            return;
        }
        let closing = ctx.input(|i| i.viewport().close_requested());
        if closing
            || self.last_settings_save_at.elapsed() >= std::time::Duration::from_millis(1200)
        {
            self.settings.save_persisted();
            self.last_saved_settings = self.settings.clone();
            self.last_settings_save_at = std::time::Instant::now();
        }
    }

    /// Draw the floating export-options window when `export_opts_visible`
    /// is set. The window is dismissable; "Start Export…" hands off to
    /// `commit_export()` which pops the save-file dialog and spawns the
    /// encode worker.
    fn draw_export_options_window(&mut self, ctx: &egui::Context) {
        if !self.export_opts_visible { return; }
        let mut open = true;
        let mut commit = false;
        egui::Window::new("Export options")
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(360.0)
            .open(&mut open)
            .show(ctx, |ui| {
                let opts = &mut self.export_opts;
                ui.label(RichText::new("Codec").strong());
                egui::ComboBox::from_id_source("export_codec")
                    .selected_text(opts.codec.label())
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut opts.codec, ExportCodec::H265,   ExportCodec::H265.label());
                        ui.selectable_value(&mut opts.codec, ExportCodec::ProRes, ExportCodec::ProRes.label());
                    });
                ui.add_space(8.0);

                match opts.codec {
                    ExportCodec::H265 => {
                        ui.label(RichText::new("H.265 bitrate").strong());
                        ui.add(egui::Slider::new(&mut opts.h265_bitrate_mbps, 20..=500)
                            .text("Mbps"));
                        ui.add_space(4.0);
                        // 10-bit (Main10) is the default and stays 10-bit
                        // end-to-end; 8-bit (Main) is offered for users who
                        // need the wider-compatibility profile.
                        ui.label(RichText::new("Bit depth").strong());
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut opts.h265_bit_depth, 10u8, "10-bit (Main10)");
                            ui.selectable_value(&mut opts.h265_bit_depth, 8u8,  "8-bit (Main)");
                        });
                    }
                    ExportCodec::ProRes => {
                        ui.label(RichText::new("ProRes profile").strong());
                        egui::ComboBox::from_id_source("prores_profile")
                            .selected_text(opts.prores_profile.label())
                            .width(220.0)
                            .show_ui(ui, |ui| {
                                use ProResProfile::*;
                                for p in [Proxy, Lt, Standard, Hq, P4444, P4444Xq] {
                                    ui.selectable_value(&mut opts.prores_profile, p, p.label());
                                }
                            });
                    }
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(RichText::new("VR180 metadata").strong());
                // Mutually exclusive — they overwrite the same atoms.
                // Show as radio so it's obvious only one is active.
                #[derive(PartialEq, Eq, Copy, Clone)]
                enum MetaTarget { None, Apmp, Youtube }
                let mut target = match (opts.inject_apmp, opts.inject_youtube) {
                    (true, _)        => MetaTarget::Apmp,
                    (false, true)    => MetaTarget::Youtube,
                    _                => MetaTarget::None,
                };
                ui.radio_value(&mut target, MetaTarget::Apmp,
                    "Apple Vision Pro (APMP — vexu + hfov)");
                if matches!(target, MetaTarget::Apmp) {
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        ui.label("Camera baseline");
                        ui.add(egui::Slider::new(&mut opts.apmp_baseline_mm, 30.0..=120.0)
                            .suffix(" mm").fixed_decimals(1));
                    });
                }
                ui.radio_value(&mut target, MetaTarget::Youtube,
                    "YouTube (Spherical V2 — st3d + sv3d)");
                ui.radio_value(&mut target, MetaTarget::None,
                    "None (no VR180 metadata)");
                opts.inject_apmp    = matches!(target, MetaTarget::Apmp);
                opts.inject_youtube = matches!(target, MetaTarget::Youtube);
                ui.label(RichText::new(
                    "APMP and Spatial V2 conflict in the same atoms, so \
                     only one can be active per file. Re-run the export \
                     with the other target if you need a second copy."
                ).small().color(Color32::GRAY));

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        commit = false;
                        self.export_opts_visible = false;
                    }
                    ui.add_space(8.0);
                    let label = match opts.codec {
                        ExportCodec::H265   => "Start Export… (H.265)",
                        ExportCodec::ProRes => "Start Export… (ProRes)",
                    };
                    if ui.button(RichText::new(label).strong()).clicked() {
                        commit = true;
                    }
                });
            });
        if !open { self.export_opts_visible = false; }
        if commit { self.commit_export(); }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain decoder frames first so the most recent texture is
        // ready by the time we render the central panel.
        self.drain_frames(ctx);
        // Drain export-job progress + check for completion.
        self.poll_export_job();
        // Drive the full-res still for the zoom magnifier.
        self.poll_full_res(ctx);
        // Export options floating window (shown on Export click).
        self.draw_export_options_window(ctx);
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
                    if ui.button("Load video…").clicked() {
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

                    ui.separator();
                    ui.label(RichText::new("View").color(Color32::GRAY));
                    egui::ComboBox::from_id_source("preview_mode_combo")
                        .selected_text(self.settings.preview_mode.as_str())
                        .show_ui(ui, |ui| {
                            use crate::decoder::PreviewMode as M;
                            ui.selectable_value(&mut self.settings.preview_mode,
                                M::Sbs, M::Sbs.as_str());
                            ui.selectable_value(&mut self.settings.preview_mode,
                                M::Anaglyph, M::Anaglyph.as_str());
                            ui.selectable_value(&mut self.settings.preview_mode,
                                M::Overlay50, M::Overlay50.as_str());
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
              // Scroll the controls vertically so every section stays
              // reachable even when many are expanded / the window is short.
              egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                ui.add_space(8.0);
                egui::CollapsingHeader::new(RichText::new("Source").strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_source_info(ui); });

                // Fisheye sources (OSV / SBS / BRAW): Stabilization +
                // Output are top-level sections; the camera-preset /
                // per-eye FOV / center / KB panel is its own (collapsed)
                // section. GoPro EAC keeps the existing stab + RS panels.
                if matches!(kind, Some(k) if k.is_fisheye()) {
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Stabilization").strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_stab_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Output").strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_output_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("Fisheye lens").strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_fisheye_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new("View adjustment").strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_view_adjust_panel(ui); });
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

                ui.add_space(4.0);
                egui::CollapsingHeader::new(
                    RichText::new("Color").strong()
                )
                .default_open(false)
                .show(ui, |ui| { self.draw_color_panel(ui); });
                }); // ScrollArea
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
            // Copy out the display handle so we can borrow `self` mutably for
            // the zoom/pan state below. When zoomed + paused we show the
            // `DetailCache` still — but only while it's CURRENT for both the
            // live frame and the live settings (`full_res_key == desired`).
            // During a drag/scrub the desired key moves ahead, so we show the
            // live preview instead: it renders on the decoder thread (so the
            // adjustment stays responsive) and is now frame-aligned with the
            // still, so the swap to the crisp still on settle doesn't shift.
            let cur_ts = self.current_display.as_ref()
                .map(|d| d.timestamp_s).unwrap_or(0.0);
            let use_full_res = self.preview_zoom > 1.001 && !self.playing
                && self.full_res_key == self.full_res_desired_key
                && self.full_res_display.as_ref()
                    .map(|d| (d.timestamp_s - cur_ts).abs() < 0.02).unwrap_or(false);
            let disp = if use_full_res {
                self.full_res_display.as_ref()
            } else {
                self.current_display.as_ref()
            }.map(|d| (d.egui_id, d.width, d.height));
            if let Some((egui_id, dw, dh)) = disp {
                let avail = ui.available_size();
                let aspect = dw as f32 / dh as f32;
                // Fit-to-area with aspect ratio preserved.
                let (w, h) = if avail.x / avail.y > aspect {
                    (avail.y * aspect, avail.y)
                } else {
                    (avail.x, avail.x / aspect)
                };

                // Centered image rect + an interaction region over exactly
                // that rect (for scroll-zoom and drag-pan).
                let img_rect = egui::Rect::from_center_size(
                    ui.max_rect().center(), egui::vec2(w, h));
                let resp = ui.interact(
                    img_rect, ui.id().with("preview_zoom"),
                    egui::Sense::click_and_drag());

                let zoom = &mut self.preview_zoom;
                let center = &mut self.preview_center;

                // Scroll → zoom toward the cursor.
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.01 {
                        let old_z = *zoom;
                        let new_z = (*zoom * (1.0 + scroll * 0.002)).clamp(1.0, 64.0);
                        if let Some(p) = resp.hover_pos() {
                            // UV under the cursor, kept fixed across the zoom.
                            let f = egui::vec2(
                                ((p.x - img_rect.left()) / w).clamp(0.0, 1.0) - 0.5,
                                ((p.y - img_rect.top()) / h).clamp(0.0, 1.0) - 0.5,
                            );
                            let uv_cur = *center + f / old_z;
                            *zoom = new_z;
                            *center = uv_cur - f / new_z;
                        } else {
                            *zoom = new_z;
                        }
                    }
                }
                // Drag → pan (only meaningful when zoomed in).
                if resp.dragged() && *zoom > 1.0 {
                    let d = resp.drag_delta();
                    center.x -= d.x / (w * *zoom);
                    center.y -= d.y / (h * *zoom);
                }
                // Double-click → reset to fit.
                if resp.double_clicked() {
                    *zoom = 1.0;
                    *center = egui::vec2(0.5, 0.5);
                }

                // Clamp the visible window inside [0,1] so we never show
                // outside the image.
                let half = 0.5 / *zoom;
                center.x = center.x.clamp(half, 1.0 - half);
                center.y = center.y.clamp(half, 1.0 - half);
                let uv = egui::Rect::from_min_max(
                    egui::pos2(center.x - half, center.y - half),
                    egui::pos2(center.x + half, center.y + half),
                );

                // Paint the (sub-rect of the) texture into the fitted rect.
                let sized = egui::load::SizedTexture::new(egui_id, egui::vec2(w, h));
                egui::Image::new(sized).uv(uv).paint_at(ui, img_rect);

                // Zoom readout + reset hint, top-left overlay.
                let painter = ui.painter_at(img_rect);
                painter.text(
                    img_rect.left_top() + egui::vec2(8.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    format!("{:.0}%  (scroll = zoom, drag = pan, dbl-click = reset)",
                        *zoom * 100.0),
                    egui::FontId::proportional(12.0),
                    Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                );
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
                        ui.label(RichText::new("Drop a video file here, or click Load video.")
                            .size(13.0).color(Color32::GRAY));
                    });
                });
            }
        });

        // Push slider edits to the running decoder.
        self.maybe_push_settings();
        // Remember settings across launches (debounced; flushes on close).
        self.persist_settings(ctx);

        // Keep the UI awake whenever a decoder is running so the next
        // frame draws without waiting for a user input event.
        // - playing: 125 Hz, matches the source frame budget
        // - paused but alive: 20 Hz, enough to display slider-driven
        //   re-renders the decoder thread pushes into frame_rx.
        if self.decoder_alive {
            let delay_ms = if self.playing { 8 } else { 50 };
            ctx.request_repaint_after(std::time::Duration::from_millis(delay_ms));
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

    /// Global pano-map + per-eye stereo offset. Ports the Python
    /// app's sliders at `vr180_gui.py:10654-10681` plus the per-eye
    /// composition at `vr180_gui.py:12789-12804`. Applied after
    /// stabilization, so default-0 is a no-op even when stab is on.
    /// Color grading panel — port of the CDL / LUT / temp-tint-sat
    /// controls from the Python app (`vr180_gui.py:11132-11290`).
    /// All knobs apply live to the preview through the
    /// `Settings::build_color_stack()` plumbing.
    fn draw_color_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;

        // CDL: lift / gamma / gain / shadow / highlight.
        ui.label(RichText::new("Tone (CDL)").strong());
        ui.add(egui::Slider::new(&mut s.lift, -1.0..=1.0)
            .text("Lift").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.gamma, 0.2..=3.0)
            .text("Gamma").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.gain, 0.2..=3.0)
            .text("Gain").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.shadow, -1.0..=1.0)
            .text("Shadow").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.highlight, -1.0..=1.0)
            .text("Highlight").fixed_decimals(2));
        if ui.button("Reset tone").clicked() {
            s.lift = 0.0; s.gamma = 1.0; s.gain = 1.0;
            s.shadow = 0.0; s.highlight = 0.0;
        }

        ui.separator();

        // White balance + saturation.
        ui.label(RichText::new("White balance").strong());
        ui.add(egui::Slider::new(&mut s.temperature, -1.0..=1.0)
            .text("Temperature").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.tint, -1.0..=1.0)
            .text("Tint").fixed_decimals(2));
        ui.add(egui::Slider::new(&mut s.saturation, 0.0..=2.0)
            .text("Saturation").fixed_decimals(2));
        if ui.button("Reset color").clicked() {
            s.temperature = 0.0; s.tint = 0.0; s.saturation = 1.0;
        }

        ui.separator();

        // 3D LUT.
        ui.label(RichText::new("3D LUT (.cube)").strong());
        ui.horizontal(|ui| {
            let mut label = if s.lut_path.is_empty() {
                "(none)".to_string()
            } else if s.lut_path == crate::decoder::BUILTIN_OSMO_LUT_PATH {
                crate::decoder::BUILTIN_OSMO_LUT_NAME.to_string()
            } else {
                std::path::Path::new(&s.lut_path)
                    .file_name().and_then(|f| f.to_str()).unwrap_or("(invalid)")
                    .to_string()
            };
            ui.add(egui::TextEdit::singleline(&mut label).interactive(false).desired_width(160.0));
            if ui.button("Browse…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Cube LUT", &["cube", "CUBE"])
                    .pick_file()
                {
                    s.lut_path = p.to_string_lossy().to_string();
                }
            }
            if ui.button("Clear").clicked() {
                s.lut_path.clear();
            }
        });
        // Quick re-apply of the embedded Osmo 360 D-LogM → Rec.709 LUT
        // (autoloaded for OSV clips; offered here in case it was cleared).
        if s.lut_path != crate::decoder::BUILTIN_OSMO_LUT_PATH
            && ui.button("Use built-in Osmo 360 D-LogM→709").clicked()
        {
            s.lut_path = crate::decoder::BUILTIN_OSMO_LUT_PATH.to_string();
            s.lut_intensity = 1.0;
        }
        ui.add_enabled_ui(!s.lut_path.is_empty(), |ui| {
            ui.add(egui::Slider::new(&mut s.lut_intensity, 0.0..=1.0)
                .text("Intensity").fixed_decimals(2));
        });
    }

    fn draw_view_adjust_panel(&mut self, ui: &mut egui::Ui) {
        let zoom = self.preview_zoom;
        let s = &mut self.settings;
        ui.label(RichText::new("Global (both eyes)").small().color(Color32::GRAY));
        if zoom > 1.001 {
            ui.label(RichText::new(format!("fine drag ×{:.0} (zoomed)", zoom))
                .small().color(Color32::from_rgb(120, 180, 120)));
        }
        // Rotational alignment: extra-fine (0.15×) on top of the zoom
        // scaling, since these need sub-degree precision.
        let rot_fine = 0.15;
        // Global pano-map: Up/Down arrows step 0.1° per press.
        let pano_key_step = 0.1;
        fine_slider(ui, zoom, &mut s.pano_yaw_deg, -180.0..=180.0, "Yaw (°)", 3, rot_fine, pano_key_step);
        fine_slider(ui, zoom, &mut s.pano_pitch_deg, -90.0..=90.0, "Pitch (°)", 3, rot_fine, pano_key_step);
        fine_slider(ui, zoom, &mut s.pano_roll_deg, -45.0..=45.0, "Roll (°)", 3, rot_fine, pano_key_step);
        ui.add_space(4.0);
        ui.label(RichText::new("Stereo offset (right = +, left = −)")
            .small().color(Color32::GRAY));
        // Stereo offset: Up/Down arrows step 0.01° per press.
        let stereo_key_step = 0.01;
        fine_slider(ui, zoom, &mut s.stereo_yaw_deg, -10.0..=10.0, "Yaw (°)", 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_pitch_deg, -10.0..=10.0, "Pitch (°)", 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_roll_deg, -10.0..=10.0, "Roll (°)", 3, rot_fine, stereo_key_step);
        ui.add_space(4.0);
        if ui.button("Reset to 0").clicked() {
            s.pano_yaw_deg = 0.0;
            s.pano_pitch_deg = 0.0;
            s.pano_roll_deg = 0.0;
            s.stereo_yaw_deg = 0.0;
            s.stereo_pitch_deg = 0.0;
            s.stereo_roll_deg = 0.0;
        }
    }

    /// Stabilization panel for fisheye sources (own sidebar section).
    fn draw_fisheye_stab_panel(&mut self, ui: &mut egui::Ui) {
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        let is_braw = matches!(kind, Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        let is_osv  = matches!(kind, Some(vr180_pipeline::SourceKind::DjiOsv));
        let s = &mut self.settings;

        if !(is_braw || is_osv) {
            ui.label(RichText::new(
                "No gyro stabilization available for this source yet."
            ).small().color(Color32::GRAY));
            return;
        }
        let stab_label = if is_braw {
            "Stabilization (VQF 6D, BRAW)"
        } else {
            "Stabilization (DJI camera quats)"
        };
        ui.checkbox(&mut s.stabilize, stab_label);
        ui.add_enabled_ui(s.stabilize, |ui| {
            if is_osv {
                // smooth_ms = 0 → sharp camera-lock (legacy).
                // smooth_ms > 0 → soft-stab (GoPro-style).
                ui.add(egui::Slider::new(&mut s.dji_smooth_ms, 0.0..=3000.0)
                    .text("Smooth (ms)  0 = sharp lock, > 0 = soft-stab"));
                ui.add(egui::Slider::new(&mut s.dji_max_corr_deg, 0.0..=45.0)
                    .text("Max corr (°)  0 = no cap"));
            }
        });
        ui.label(RichText::new(
            "Stab toggle is live during playback — first time you flip \
             it on, gyro extraction runs (~1 s stutter). Sliders apply \
             live too."
        ).small().color(Color32::GRAY));
    }

    /// Output panel for fisheye sources — projection target + L↔R swap.
    fn draw_fisheye_output_panel(&mut self, ui: &mut egui::Ui) {
        let is_osv = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
        );
        let s = &mut self.settings;

        ui.horizontal(|ui| {
            ui.label("Format");
            egui::ComboBox::from_id_source("fisheye_output_mode")
                .selected_text(s.fisheye_output_mode.as_str())
                .show_ui(ui, |ui| {
                    use crate::decoder::FisheyeOutputMode as M;
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::HalfEquirect, "Half-equirect (VR180)");
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::Fisheye, "Fisheye SBS");
                });
        });
        ui.label(RichText::new(
            "Fisheye SBS = stabilized circular fisheye per eye in a 2×side \
             SBS frame. Stab, panomap, stereo-offset all apply just like \
             the VR180 mode; per-eye FOV controls output FOV."
        ).small().color(Color32::GRAY));

        if is_osv {
            ui.add_space(4.0);
            ui.checkbox(&mut s.fisheye_swap_eyes, "Swap L↔R eyes");
        }
    }

    /// Per-eye lens calibration panel (camera preset + per-eye FOV /
    /// center / KB + Gyroflow load). Mirrors the Python app's lens
    /// parameters. All sliders apply live during playback.
    fn draw_fisheye_panel(&mut self, ui: &mut egui::Ui) {
        let presets = vr180_fisheye::presets::presets();
        // Native per-eye pixel dims, used to display the principal point
        // as absolute pixels (stored normalized internally). Falls back
        // to 1.0 (so we'd show the raw normalized value) if unknown.
        let (native_w, native_h) = self.clip.as_ref()
            .map(|c| (c.fisheye_eye_w.max(1) as f32, c.fisheye_eye_h.max(1) as f32))
            .unwrap_or((1.0, 1.0));
        // In-file per-eye calibration the decoder resolved (OSV protobuf).
        // Used to display the actual values + seed the Override fields.
        let detected = self.control.as_ref()
            .and_then(|c| *c.detected_calib.lock());
        let zoom = self.preview_zoom;
        let s = &mut self.settings;

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
                    // Only the cameras relevant to this project are offered:
                    // the DJI Osmo 360 and a Custom slot. (Empty preset still
                    // auto-resolves to the Osmo 360 for OSV clips.)
                    for p in presets.iter()
                        .filter(|p| p.name == "DJI Osmo 360" || p.name == "Custom")
                    {
                        let selected = s.fisheye_preset == p.name;
                        if ui.selectable_label(selected, p.name).clicked() {
                            s.fisheye_preset = p.name.to_string();
                        }
                    }
                });
        });

        let preset_default_fov = presets
            .iter().find(|p| p.name == s.fisheye_preset)
            .map(|p| p.default_fov_deg as f32)
            .unwrap_or(180.0);
        let preset_k: [f32; 4] = presets
            .iter().find(|p| p.name == s.fisheye_preset)
            .map(|p| [p.calib.k[0] as f32, p.calib.k[1] as f32,
                      p.calib.k[2] as f32, p.calib.k[3] as f32])
            .unwrap_or([0.0; 4]);

        ui.add_space(4.0);
        draw_eye_lens(
            ui, "L", "Left eye", native_w, native_h, zoom,
            detected.map(|d| d.left),
            &mut s.fisheye_override_left,
            &mut s.fisheye_fov_deg_left,
            &mut s.fisheye_cx_norm_left,
            &mut s.fisheye_cy_norm_left,
            &mut s.fisheye_k_left,
            preset_default_fov, preset_k,
        );
        ui.separator();
        draw_eye_lens(
            ui, "R", "Right eye", native_w, native_h, zoom,
            detected.map(|d| d.right),
            &mut s.fisheye_override_right,
            &mut s.fisheye_fov_deg_right,
            &mut s.fisheye_cx_norm_right,
            &mut s.fisheye_cy_norm_right,
            &mut s.fisheye_k_right,
            preset_default_fov, preset_k,
        );

        // ── Load Gyroflow lens profile (applies to BOTH eyes; enables
        //    override on both so the loaded values take effect) ────
        ui.add_space(6.0);
        if ui.button("Load Gyroflow lens profile (both eyes)…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Gyroflow lens profile (.json)", &["json", "JSON"])
                .pick_file()
            {
                match vr180_fisheye::GyroflowLensProfile::load(&path) {
                    Ok(prof) => match prof.to_calibration() {
                        Ok(cal) => {
                            let kk = [cal.k[0] as f32, cal.k[1] as f32,
                                      cal.k[2] as f32, cal.k[3] as f32];
                            s.fisheye_override_left = true;
                            s.fisheye_override_right = true;
                            s.fisheye_k_left = kk;
                            s.fisheye_k_right = kk;
                            if cal.calib_w > 0 {
                                let r_max = (cal.calib_w.min(cal.calib_h) as f64) * 0.5;
                                let fov = cal.full_fov_from_rim(r_max).to_degrees() as f32;
                                s.fisheye_fov_deg_left = fov;
                                s.fisheye_fov_deg_right = fov;
                            }
                            tracing::info!(
                                "loaded Gyroflow lens profile: {} — fov≈{:.2}°, k={:?}",
                                path.display(), s.fisheye_fov_deg_left, kk
                            );
                        }
                        Err(e) => tracing::error!("Gyroflow lens profile invalid: {e}"),
                    },
                    Err(e) => tracing::error!("load Gyroflow JSON {}: {e}", path.display()),
                }
            }
        }

        ui.add_space(6.0);
        ui.label(RichText::new(
            "With Override off, each eye uses the in-file (OSV) / preset \
             calibration. Turn Override on for an eye to set its FOV, \
             principal point and distortion by hand."
        ).small().color(Color32::GRAY));
    }
}

/// Draw one eye's lens-calibration controls. A single "Override" toggle
/// gates everything: off → in-file/preset calibration; on → manual FOV,
/// absolute principal point, and KB k1–k4.
///
/// The principal point is stored normalized (`*cx_norm` ∈ [0,1]) but
/// displayed/edited as absolute pixels at the native per-eye resolution
/// (`native_w` × `native_h`), so the value is resolution-independent.
///
/// Free function so each eye's distinct `Settings` fields can be borrowed
/// independently. `id` must be unique per eye ("L"/"R").
fn draw_eye_lens(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    native_w: f32,
    native_h: f32,
    zoom: f32,
    detected: Option<crate::decoder::EyeCalibSeed>,
    over: &mut bool,
    fov: &mut f32,
    cx_norm: &mut f32,
    cy_norm: &mut f32,
    k: &mut [f32; 4],
    preset_default_fov: f32,
    preset_k: [f32; 4],
) {
    ui.push_id(id, |ui| {
        // Header row: eye label + the single master Override toggle.
        let was_over = *over;
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(over, "Override");
            });
        });

        // On the off→on transition, seed the principal point from the ACTUAL
        // in-file calibration when we have it (real per-lens cx/cy). FOV is
        // taken from the PRESET — we no longer load it from the file
        // calibration — and k is the preset too, matching Override-off.
        if *over && !was_over {
            *fov = if preset_default_fov > 0.0 { preset_default_fov } else { 180.0 };
            if let Some(d) = detected {
                *cx_norm = d.cx_norm;
                *cy_norm = d.cy_norm;
            } else {
                *cx_norm = 0.5;
                *cy_norm = 0.5;
            }
            *k = preset_k;
        }

        if !*over {
            // FOV comes from the PRESET (not the file calibration); the
            // principal point is the actual in-file cx/cy. k is the preset.
            let fov_disp = if preset_default_fov > 0.0 { preset_default_fov } else { 180.0 };
            if let Some(d) = detected {
                ui.label(RichText::new(format!(
                    "Preset FOV {:.1}°  ·  in-file cx {:.1}, cy {:.1} px",
                    fov_disp, d.cx_norm * native_w, d.cy_norm * native_h,
                )).small().color(Color32::GRAY));
            } else {
                ui.label(RichText::new(format!("Preset FOV {:.1}°  ·  center cx/cy", fov_disp))
                    .small().color(Color32::GRAY));
            }
            return;
        }

        if zoom > 1.001 {
            ui.label(RichText::new(format!("fine drag ×{:.0} (zoomed)", zoom))
                .small().color(Color32::from_rgb(120, 180, 120)));
        }

        // FOV (zoom-scaled fine drag).
        fine_slider(ui, zoom, fov, 90.0..=230.0, "Full FOV (°)", 2, 1.0, 0.0);

        // Principal point — absolute pixels at native res, stored as norm.
        // DragValue speed scales with 1/zoom for sub-pixel control.
        let drag_speed = (0.5 / zoom.max(1.0)) as f64;
        let mut cx_px = *cx_norm * native_w;
        let mut cy_px = *cy_norm * native_h;
        ui.horizontal(|ui| {
            ui.label("cx (px)");
            if ui.add(egui::DragValue::new(&mut cx_px).speed(drag_speed)
                .range(0.0..=native_w)).changed()
            {
                *cx_norm = (cx_px / native_w).clamp(0.0, 1.0);
            }
            ui.label("cy (px)");
            if ui.add(egui::DragValue::new(&mut cy_px).speed(drag_speed)
                .range(0.0..=native_h)).changed()
            {
                *cy_norm = (cy_px / native_h).clamp(0.0, 1.0);
            }
        });
        if ui.small_button("Center").clicked() {
            *cx_norm = 0.5;
            *cy_norm = 0.5;
        }

        // KB distortion (zoom-scaled fine drag).
        ui.collapsing("KB distortion (k1–k4)", |ui| {
            for (i, ki) in k.iter_mut().enumerate() {
                fine_slider(ui, zoom, ki, -0.5..=0.5, &format!("k{}", i + 1), 6, 1.0, 0.0);
            }
            if ui.small_button("Reset k to preset").clicked() {
                *k = preset_k;
            }
        });
    });
}

/// A slider whose effective drag is scaled by `1/zoom` when zoomed in
/// (`zoom > 1`), giving sub-pixel-fine value control for alignment while
/// the slider still spans the full range. At `zoom == 1` it's a plain
/// slider. The trick: after egui maps the cursor position to a value, we
/// keep only `1/zoom` of this frame's change, so the value creeps at
/// `1/zoom` of cursor speed.
fn fine_slider(
    ui: &mut egui::Ui,
    zoom: f32,
    val: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
    decimals: usize,
    extra: f32,
    key_step: f32,
) -> egui::Response {
    let (lo, hi) = (*range.start(), *range.end());
    let before = *val;
    let mut slider = egui::Slider::new(val, range)
        .text(text)
        .fixed_decimals(decimals)
        .smart_aim(false);
    // `step_by` makes the keyboard step EXACTLY `key_step` per press — for
    // both the track (Left/Right) and the numeric value box (Up/Down). Its
    // only side effect is quantizing a drag to multiples of `key_step`, which
    // here is far finer than the slider's pixel resolution (0.1° over a ~2°/px
    // track, 0.001° over a ~0.1/px track), so the drag feel is unchanged.
    if key_step > 0.0 {
        slider = slider.step_by(key_step as f64);
    }
    let resp = ui.add(slider);
    // `extra < 1.0` makes a control even finer (e.g. rotational alignment
    // needs sub-degree control). Effective per-frame gain = extra / zoom.
    if zoom > 1.001 && resp.dragged() {
        *val = before + (*val - before) * extra / zoom;
    }
    // Up/Down arrow keys nudge by exactly `key_step` while focused — e.g.
    // 0.1° for the global pano-map, 0.001° for the stereo offset. The value
    // box handles Up/Down (stepping by `step_by` above) when IT has focus and
    // CONSUMES them — so `count_and_consume_key` here returns 0 and never
    // double-steps. When the track is focused instead (it ignores Up/Down,
    // using Left/Right), the keys are free and we apply the step ourselves.
    if key_step > 0.0 && resp.has_focus() {
        let net = ui.input_mut(|i| {
            i.count_and_consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) as i32
                - i.count_and_consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) as i32
        });
        if net != 0 {
            *val = (*val + net as f32 * key_step).clamp(lo, hi);
        }
    }
    resp
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
