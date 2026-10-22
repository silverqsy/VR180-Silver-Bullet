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

/// What a worker-thread file picker came back with. One variant per call
/// site; `Cancelled` covers the user dismissing the panel.
enum DialogResult {
    /// Video sources to open (the "Load video…" button and Cmd+O).
    Videos(Vec<std::path::PathBuf>),
    /// Batch output directory.
    BatchOutDir(std::path::PathBuf),
    /// A .cube LUT for the colour panel.
    Lut(std::path::PathBuf),
    /// A Gyroflow lens profile, applied to both eyes.
    GyroflowProfile(std::path::PathBuf),
    Cancelled,
}

pub struct App {
    /// The UI context, handed to decoder workers so they can repaint the
    /// window when a frame lands (`DecoderControl::wake_ui`).
    egui_ctx: egui::Context,
    /// Shared with the decoder worker. Built on eframe's wgpu device.
    pipeline: Arc<vr180_pipeline::gpu::Device>,
    /// Handle to the egui_wgpu renderer — we use it to register
    /// `wgpu::Texture`s for display. egui's RwLock wraps parking_lot's
    /// internally; we hold an Arc to share with future worker threads
    /// that want to register textures (currently the App holds it
    /// exclusively).
    egui_renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    /// Texture-view format used to register our Rec.709-gamma SBS with egui.
    /// Always `Rgba8Unorm` (pass-through): egui-wgpu 0.34 samples display
    /// textures as gamma/raw bytes ("NOT sRGB-aware"), so an sRGB *view*
    /// would over-darken the preview to `sRGB⁻¹(v)`. See the constructor.
    preview_view_format: wgpu::TextureFormat,

    // ─── User-visible state ──────────────────────────────────────
    loaded_path: Option<PathBuf>,
    /// Files named on the command line (`vr180-gui a.insv b.dng`, or macOS
    /// `open App.app --args …`): opened on the first frame, once the
    /// window exists.
    pending_open: Vec<PathBuf>,
    /// Last (swap, upside_down) pair the decoder was opened with —
    /// a change triggers an automatic clip reload (the dual-stream
    /// iterators bind eye order at open).
    eye_orientation_applied: (bool, bool),
    clip: Option<ClipInfo>,
    /// Playing means the decoder thread is alive AND not paused.
    /// Paused-but-not-stopped (decoder thread parked) is `playing=false`
    /// with `decoder_alive=true`.
    playing: bool,
    /// True while a decoder thread is alive (started but not Stopped).
    /// Lets Pause toggle the `paused` flag instead of killing the worker.
    decoder_alive: bool,
    /// One-shot (reset by `spawn_decoder`): the "settings changed with no
    /// decoder alive" warning has been logged for the current stop.
    no_decoder_warned: bool,
    /// Still photo: the magnifier currently asks for more pixels per eye
    /// than the binned working size has (computed by the central panel from
    /// zoom × on-screen eye size; fed to `Settings::still_native_zoom`).
    still_native_wanted: bool,
    /// True from `spawn_decoder` until the decoder delivers its FIRST frame.
    /// While set, play/seek respawns are ignored — clicking Play repeatedly
    /// during a slow load would otherwise pile up detached decoder threads
    /// all contending for the same file (the "click play too fast" freeze).
    decoder_starting: bool,
    /// A decoder that exited BEFORE delivering its first frame has already
    /// been respawned once (see `poll_load`); cleared when a first frame lands.
    decoder_retried: bool,
    /// In-flight async clip load. The blocking metadata reads (probe, format
    /// detect, GoPro GPMF, segment scan, per-segment probes) run on a worker
    /// thread instead of the UI thread, so a cold/slow source — notably a
    /// spun-down SMB/NAS share, where the first access stalls for seconds
    /// while the disks wake — never freezes the GUI. `update` polls `rx`;
    /// `finish_load` applies the result. `None` when no load is in flight.
    load_pending: Option<LoadPending>,
    /// Monotonic load id. Bumped on every `load_file_inner`; a delivered
    /// `LoadedMeta` is applied only if its token still matches, so a newer
    /// load (clip switch mid-read) cleanly supersedes an older one.
    load_token: u64,
    /// Set when the user presses Play DURING the async-load window (the clip
    /// isn't ready yet, so the press can't spawn a decoder). `finish_load`
    /// honors it by spawning unpaused instead of paused, so the intent isn't
    /// lost. Reset on each fresh load.
    play_when_ready: bool,
    /// Currently displayed frame texture (kept alive while egui still
    /// references it) + the egui texture id we registered it under.
    current_display: Option<DisplayFrame>,
    /// Pending frames coming off the decoder. We drain on every
    /// repaint and pick the most recent one — naturally drops frames
    /// the UI couldn't keep up with.
    frame_rx: Option<Receiver<DecodedFrame>>,
    /// Native-res zoom stills for EAC (.360) clips. The decoder thread
    /// emits these while `control.want_detail` is set (paused + zoomed);
    /// we register them into `full_res_display`. Fisheye sources use the
    /// main-thread `DetailCache` instead and leave this idle.
    detail_frame_rx: Option<Receiver<DecodedFrame>>,
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
    /// Per-source-kind settings template (`"osv"` / `"eac"` / `"sbs"` /
    /// `"braw"`). A newly-added clip inherits ITS kind's template (not the
    /// previously-active clip's, which may be a different camera), and the
    /// active clip's edits flow back into its kind's slot — so OSV and
    /// `.360` setups stay independent and each persists across same-kind files.
    kind_settings: std::collections::HashMap<String, Settings>,

    // ─── Export job (fisheye sources only for now) ───────────────
    /// `Some` while an export is in flight. `None` once the worker
    /// finishes (cleanly or via cancel) — the GUI re-enables the
    /// Export button.
    export_job: Option<ExportJob>,

    // ─── Native file pickers ──────────────────────────────────────
    //
    // These MUST NOT be run from inside `update()`. rfd's blocking API
    // ends in `-[NSSavePanel runModal]`, which spins a nested modal
    // runloop while winit's event handler is still borrowed. winit posts
    // its queued events with `CFRunLoopPerformBlock` in
    // `kCFRunLoopDefaultMode` and deliberately avoids the modal and
    // event-tracking modes so that a blocking modal is normally safe —
    // but a Core Drag defeats that: `NSCoreDragTrackingProc` pumps
    // `-[NSRunLoop runUntilDate:]` in DEFAULT mode, which drains the
    // queued block straight back into the borrowed handler. winit's
    // `try_borrow_mut` then fails and panics across an ObjC nounwind
    // boundary, i.e. an immediate abort. Confirmed from the crash report
    // for pid 84516 (frames: App::update → pick_and_load_file → rfd
    // run_modal → NSSavePanel runModal → NSCoreDragTrackingProc →
    // CFRunLoopDoBlocks → winit's block → abort).
    //
    // Running the picker on a worker thread fixes it: rfd wraps its
    // blocking body in `run_on_main`, so off-thread it marshals through
    // `dispatch::Queue::main().exec_sync` and `runModal` executes from a
    // main-queue block — a SIBLING callout of the observer that drives
    // winit, never nested inside it. No borrow is held for the dialog's
    // life, so nothing is ever queued to be drained re-entrantly.
    //
    // NB `rfd::AsyncFileDialog` is NOT the fix. `ModalFuture::new` only
    // takes the sheet path when a parent window resolves; otherwise it
    // falls back to a plain `runModal` on the main thread and silently
    // reintroduces this exact crash.
    dialog_tx: crossbeam_channel::Sender<DialogResult>,
    dialog_rx: crossbeam_channel::Receiver<DialogResult>,
    /// A picker is on screen. The window keeps repainting behind it now
    /// (the event loop is no longer blocked), so without this a second
    /// click would stack a second panel.
    dialog_open: bool,

    // ─── Auto-update (see updater.rs; export-job channel pattern) ──
    updater_tx: crossbeam_channel::Sender<crate::updater::UpdateEvent>,
    updater_rx: crossbeam_channel::Receiver<crate::updater::UpdateEvent>,
    /// Newer version advertised by the feed for this platform.
    update_available: Option<crate::updater::UpdateInfo>,
    /// Download progress `(got, total)` while installing.
    update_progress: Option<(u64, u64)>,
    /// True from "Install & Restart" until the process is replaced
    /// (or a Failed event lands).
    update_installing: bool,
    /// Error/status line shown in the popover (check failures for
    /// MANUAL checks, install failures, or "up to date").
    update_status: Option<String>,
    show_update_popover: bool,
    /// The in-flight check was user-initiated → surface its result
    /// even when it's "up to date" / an error.
    update_manual_check: bool,
    /// Auto-open the popover once per discovered version per session.
    update_prompted_version: Option<String>,
    update_last_check: std::time::Instant,
    updater_prefs: crate::updater::UpdaterPrefs,
    /// Windows: verified installer waiting for app teardown — set by
    /// `UpdateEvent::ReadyToInstall`, consumed in `on_exit` (the app
    /// closes itself gracefully, then the installer runs + relaunches).
    pending_update_install: Option<crate::updater::StagedUpdate>,
    /// Release notes for the RUNNING version, from the feed (available
    /// once a check finds us up to date). Shown in the update popover as
    /// "What's new in this version".
    current_version_notes: Option<String>,

    /// Audio playback for the preview. `Some` when the clip's source
    /// has an audio track. Tied to the same play / pause / seek
    /// commands as the video decoder.
    audio_player: Option<crate::audio_player::AudioPlayer>,

    /// The UNIFIED export output settings (codec / bitrate / resolution /
    /// metadata), shared by "Export selected" and "Export all" and edited
    /// inline in the bottom export bar. Frozen into `batch_opts` at run
    /// start so mid-run edits can't change later items.
    export_opts: ExportOptions,

    // ─── Source clip list + export queue ──────────────────────────
    /// The working set: every loaded file, each with its own Settings
    /// (see `BatchItem`). Shown in the Source panel; the export queue runs it.
    batch: Vec<BatchItem>,
    /// Index of the ACTIVE clip (preview + sidebar). Kept in sync with
    /// `loaded_path`; `None` when nothing is loaded or the active file
    /// was removed from the list.
    active_clip: Option<usize>,
    /// In-flight "Auto align" stereo-offset measurement (one-shot worker);
    /// `Some` while running. Result is the correction to ADD to the
    /// stereo sliders.
    align_rx: Option<crossbeam_channel::Receiver<Result<vr180_pipeline::stereo_align::StereoAlignResult, String>>>,
    /// Last Auto align outcome, shown under the button.
    align_status: Option<String>,
    /// True while the export queue is draining (one item exports at a time).
    batch_running: bool,
    /// Index of the item the current `export_job` belongs to (only
    /// meaningful while `batch_running`).
    batch_current: Option<usize>,
    /// `export_opts` frozen at run start so mid-run edits can't change
    /// later items.
    batch_opts: Option<ExportOptions>,
    /// Output folder override. `None` = next to each source file.
    batch_out_dir: Option<PathBuf>,
    /// Set by the "Skip current" button so completion marks the item
    /// Skipped instead of Done.
    batch_skip_flag: bool,
    /// Items armed for the CURRENT run (set at start) — drives the overall
    /// progress bar + ETA in the export bar.
    batch_run_indices: Vec<usize>,
    /// When the current run started (for elapsed + the completion summary).
    batch_started_at: Option<std::time::Instant>,
    /// ETA rate anchor: (time of the run's first written frame, overall
    /// done-frames at that moment). Rate = frames since / seconds since,
    /// so load / decoder / encoder start-up never counts as encode time.
    run_rate_anchor: Option<(std::time::Instant, u64)>,
    /// Last finished run's outcome, shown as a banner until dismissed/replaced.
    last_batch_summary: Option<BatchSummary>,
    /// Whether the export-bar's Format settings popover (a non-modal window)
    /// is open. A toggle, not a fragile dropdown menu.
    show_export_settings: bool,
    /// UI language (English / 中文), persisted next to settings.json.
    lang: Lang,

    /// Per-clip segment override chosen at import: representative path →
    /// the files that make up that clip. `clip_segments` consults this
    /// before falling back to `detect_segments` (GoPro auto-chapters), so
    /// the import "merge vs individual" choice flows to decode AND export.
    merge_groups: std::collections::HashMap<PathBuf, Vec<PathBuf>>,
    /// An import awaiting the user's merge/individual decision (drives the
    /// modal). `None` when no decision is pending.
    pending_import: Option<PendingImport>,

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
    /// External 3D display output (side-by-side monitor / AR glasses).
    three_d: ThreeDDisplay,
    /// Last snapshot result for the status bar: (message, when).
    snapshot_status: Option<(String, std::time::Instant)>,
    /// A snapshot waiting for the export-size still of the paused frame.
    snapshot_pending: Option<SnapshotReq>,
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
    /// When the native still was last actually rendered — throttles the
    /// same-frame (slider-drag) re-render so the UI thread doesn't pay a
    /// full-res render per pointer event.
    full_res_rendered_at: std::time::Instant,
    /// Native-res still renderer (synchronous, main-thread). Created on
    /// entering detail mode (paused + zoomed on a fisheye clip), dropped on
    /// leaving. Caches the decoded native frame so alignment tweaks only
    /// re-project (fast). NOT a background thread — see `DetailCache`.
    detail_cache: Option<crate::decoder::DetailCache>,
    /// Last in-file calibration we re-seeded the Override fields from. When a
    /// new clip publishes a different `detected_calib`, any eye currently in
    /// Override is re-seeded from it (so Override tracks the new clip's
    /// factory calib instead of the previous clip's stale values).
    fisheye_last_seeded: Option<crate::decoder::DetectedLensCalib>,
    /// One-shot guard set by `select_clip`: the just-restored settings
    /// carry per-clip Override values — skip the next auto-reseed.
    suppress_reseed_once: bool,
}

/// Codec choice for the export pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportCodec { H265, ProRes }

/// Output container for a STILL source (Insta360 X6 `.dng`). Videos
/// use `ExportCodec`; stills use this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StillFormatChoice { Png16, Jpeg, LinearDng }

impl StillFormatChoice {
    fn label(self) -> &'static str {
        match self {
            StillFormatChoice::Png16 => "PNG (16-bit)",
            StillFormatChoice::Jpeg => "JPEG",
            StillFormatChoice::LinearDng => "DNG (linear raw)",
        }
    }
    fn extension(self) -> &'static str {
        match self { Self::Png16 => "png", Self::Jpeg => "jpg", Self::LinearDng => "dng" }
    }
    fn to_pipeline(self) -> vr180_pipeline::still_export::StillFormat {
        use vr180_pipeline::still_export::StillFormat as F;
        match self {
            Self::Png16 => F::Png16,
            Self::Jpeg => F::Jpeg { quality: 92 },
            Self::LinearDng => F::LinearDng { compression: vr180_pipeline::dng_write::DngCompression::LosslessJpegTiles },
        }
    }
}

/// Output resolution target for the export. `Native` renders at the
/// source's per-eye dimensions — labeled by the OSV case (3840² per eye
/// → 7680×3840 SBS); a `.360` source actually renders its cross-native
/// 3936² → 7872×3936. `R8k` (the default) renders the PROJECTION at
/// 4096² per eye → 8192×4096 SBS — the projection kernel samples the
/// native-res source directly into the 4096 grid (one resample,
/// end-to-end at output res; sharpen / color / encode all run at 4096),
/// NOT a finished-frame upscale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportResolution { Native, R8k }

impl ExportResolution {
    fn label(self) -> &'static str {
        match self {
            ExportResolution::Native => "7680 × 3840",
            ExportResolution::R8k    => "8192 × 4096",
        }
    }
}

/// Per-eye size of a reframed export (the side-by-side frame is twice as
/// wide): 1080 / 1440 / 2160 lines for video; a STILL may also pick 3240
/// (16:9 → 5760×3240 per eye) or, for the 1:1 view, 4320 (4320² per eye) —
/// a photo renders from the native lens half, so the only cost is the file.
/// The width follows the view's aspect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReframeSize { P1080, P1440, P2160, P3240, P4320 }

impl ReframeSize {
    fn height(self) -> u32 {
        match self {
            Self::P1080 => 1080, Self::P1440 => 1440, Self::P2160 => 2160,
            Self::P3240 => 3240, Self::P4320 => 4320,
        }
    }
    /// The sizes offered for a source: video stops at 2160 lines; a still
    /// adds 3240 for both aspects and 4320 for the 1:1 view only (16:9 at
    /// 4320 would be 7680×4320 per eye).
    fn choices(is_still: bool, aspect: crate::decoder::ReframeAspect) -> &'static [ReframeSize] {
        match (is_still, aspect) {
            (false, _) => &[Self::P1080, Self::P1440, Self::P2160],
            (true, crate::decoder::ReframeAspect::Wide) => &[Self::P1080, Self::P1440, Self::P2160, Self::P3240],
            (true, crate::decoder::ReframeAspect::Square) => &[Self::P1080, Self::P1440, Self::P2160, Self::P3240, Self::P4320],
        }
    }
    /// The size clamped to what `choices` offers for the source — a
    /// still-only pick remembered from a photo must not size a video.
    fn clamped(self, is_still: bool, aspect: crate::decoder::ReframeAspect) -> Self {
        let allowed = Self::choices(is_still, aspect);
        if allowed.contains(&self) { self } else { *allowed.last().unwrap_or(&Self::P2160) }
    }
    fn eye_dims(self, aspect: crate::decoder::ReframeAspect) -> (u32, u32) {
        let h = self.height();
        let w = (((h as f32 * aspect.ratio()).round() as u32) + 1) & !1;
        (w, h)
    }
    fn label(self, aspect: crate::decoder::ReframeAspect) -> String {
        let (w, h) = self.eye_dims(aspect);
        format!("{} × {}  ({w} × {h} per eye)", 2 * w, h)
    }
    /// Recommended H.265 average bitrate (Mbps) for a reframed export of
    /// this size: ~0.28 bits per pixel per frame over the side-by-side
    /// frame at 30 fps, ×1.5 at 60 fps (linear between, flat below 30),
    /// rounded to 5 and kept inside the slider's 10..=300 range. A flat
    /// rectilinear frame is ordinary video — every pixel is seen and none
    /// are stretched — so it needs a fraction of the VR180 rate:
    /// 3840×1080 → 35, 5120×1440 → 60, 7680×2160 → 140 Mbps at 30 fps.
    fn recommended_bitrate_mbps(self, aspect: crate::decoder::ReframeAspect, fps: f32) -> u32 {
        let (w, h) = self.eye_dims(aspect);
        let pixels = (2 * w * h) as f32;
        let fps_factor = 1.0 + 0.5 * ((fps - 30.0) / 30.0).clamp(0.0, 1.0);
        let mbps = pixels * 30.0 * 0.28 * fps_factor / 1.0e6;
        (((mbps / 5.0).round() as u32) * 5).clamp(10, 300)
    }
}

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

#[derive(Debug, Clone, Copy)]
struct ExportOptions {
    codec: ExportCodec,
    /// Still-photo container (used only when the source is a still).
    still_format: StillFormatChoice,
    /// Output resolution target (Native / 8192×4096).
    resolution: ExportResolution,
    /// H.265 average bitrate in Mbps for VR180 output. Range 20..=500.
    h265_bitrate_mbps: u32,
    /// H.265 average bitrate in Mbps for REFRAMED (Flat 3D) output. Range
    /// 10..=300, kept apart from the VR180 rate: a flat rectilinear frame
    /// needs a fraction of what a half-equirect does, and the two must not
    /// bleed into each other when the Format switches.
    reframe_bitrate_mbps: u32,
    /// `false` → the reframed rate follows the size / aspect / frame-rate
    /// recommendation (`ReframeSize::recommended_bitrate_mbps`), re-derived
    /// per clip at export time. A manual slider change pins the value
    /// (`true`) until "Use recommended" is pressed.
    reframe_bitrate_custom: bool,
    /// 8 or 10 (Main / Main10) for H.265.
    h265_bit_depth: u8,
    /// H.265 encoder: hardware (NVIDIA NVENC) vs software (libx265).
    /// Hardware is the default on Windows — it's many times faster at
    /// VR180 resolutions. macOS always uses VideoToolbox regardless.
    /// Falls back to libx265 automatically if NVENC can't be opened.
    h265_hardware: bool,
    prores_profile: ProResProfile,
    /// Inject Apple Projected Media Profile (Vision Pro VR180).
    inject_apmp: bool,
    /// Inject Spatial Media V2 (YouTube VR180) — st3d + sv3d boxes.
    /// MUTUALLY EXCLUSIVE with `inject_apmp` (they overwrite the same
    /// atoms in the visual sample entry) — the GUI enforces it.
    inject_youtube: bool,
    /// Reframed exports: per-eye size (the aspect comes from the clip).
    reframe_size: ReframeSize,
    /// Camera baseline in mm for APMP `cams/blin`. Typical 60..=70.
    apmp_baseline_mm: f32,
    /// "BeyondVR Hack": at the final output stage, scale each eye of the
    /// SBS half-equirect frame about its own half-center, padding the
    /// border with black. No-op for fisheye output and in preview.
    beyondvr_hack: bool,
    /// Per-eye scale in percent when the hack is on.
    beyondvr_scale_pct: f32,
}

#[cfg(test)]
mod reframe_bitrate_tests {
    use super::ReframeSize;
    use crate::decoder::ReframeAspect;

    #[test]
    fn recommended_rates_match_the_table() {
        let cases = [
            (ReframeSize::P1080, ReframeAspect::Square, 30.0, 20),
            (ReframeSize::P1080, ReframeAspect::Wide,   30.0, 35),
            (ReframeSize::P1440, ReframeAspect::Square, 30.0, 35),
            (ReframeSize::P1440, ReframeAspect::Wide,   30.0, 60),
            (ReframeSize::P2160, ReframeAspect::Square, 30.0, 80),
            (ReframeSize::P2160, ReframeAspect::Wide,   30.0, 140),
            (ReframeSize::P1080, ReframeAspect::Wide,   60.0, 50),
            (ReframeSize::P2160, ReframeAspect::Wide,   60.0, 210),
            (ReframeSize::P1080, ReframeAspect::Square, 24.0, 20), // flat below 30 fps
        ];
        for (sz, aspect, fps, want) in cases {
            assert_eq!(sz.recommended_bitrate_mbps(aspect, fps), want, "{sz:?} {aspect:?} @ {fps}");
        }
        // Always inside the slider's range.
        for sz in [ReframeSize::P1080, ReframeSize::P1440, ReframeSize::P2160] {
            for fps in [23.976_f32, 30.0, 50.0, 59.94, 120.0] {
                let r = sz.recommended_bitrate_mbps(ReframeAspect::Wide, fps);
                assert!((10..=300).contains(&r) && r % 5 == 0, "{sz:?} @ {fps}: {r}");
            }
        }
    }
}

impl ExportOptions {
    /// The H.265 rate an export actually uses: the VR180 slider value, or
    /// in Reframed mode the reframed rate — the recommendation for this
    /// size / aspect / `fps` unless the user pinned a custom value.
    fn effective_h265_mbps(&self, reframe: Option<crate::decoder::ReframeAspect>, fps: f32) -> u32 {
        match reframe {
            Some(_) if self.reframe_bitrate_custom => self.reframe_bitrate_mbps,
            Some(aspect) => self.reframe_size.recommended_bitrate_mbps(aspect, fps),
            None => self.h265_bitrate_mbps,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            codec: ExportCodec::H265,
            still_format: StillFormatChoice::Png16,
            resolution: ExportResolution::R8k,
            h265_bitrate_mbps: 200,
            reframe_bitrate_mbps: 35,
            reframe_bitrate_custom: false,
            h265_bit_depth: 10,
            h265_hardware: true,
            prores_profile: ProResProfile::Standard,
            // APMP is the headline target (Vision Pro). They're
            // mutually exclusive; the dialog enforces it.
            inject_apmp: true,
            inject_youtube: false,
            apmp_baseline_mm: 65.0,
            reframe_size: ReframeSize::P1080,
            beyondvr_hack: false,
            beyondvr_scale_pct: 85.0,
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

/// One entry in the SOURCE CLIP LIST — the app's primary working set.
/// Every loaded file lives here with its OWN `settings`; the ACTIVE
/// entry (`App::active_clip`) is what the preview shows and what the
/// sidebar edits (its settings mirror `App::settings` live, and sync
/// back on clip switch / batch start). The batch export simply runs
/// this list. Probe data is captured at add time (cheap header read).
struct BatchItem {
    path: PathBuf,
    fps: f32,
    duration_sec: f64,
    source_kind: vr180_pipeline::SourceKind,
    fisheye_eye_w: u32,
    fisheye_eye_h: u32,
    width: u32,
    height: u32,
    settings: crate::decoder::Settings,
    status: BatchStatus,
    /// Ticked in the Source list to include in "Export selected".
    selected: bool,
}

#[derive(Clone, PartialEq)]
enum BatchStatus {
    /// Never exported (or explicitly re-armed). The default.
    Idle,
    /// Armed for the current batch run.
    Queued,
    Running,
    Done,
    Skipped,
    Failed(String),
}

impl BatchStatus {
    fn label(&self) -> String {
        match self {
            BatchStatus::Idle       => "—".into(),
            BatchStatus::Queued     => "queued".into(),
            BatchStatus::Running    => "running…".into(),
            BatchStatus::Done       => "done".into(),
            BatchStatus::Skipped    => "skipped".into(),
            BatchStatus::Failed(e)  => format!("failed: {e}"),
        }
    }
}

/// Outcome of a finished export run, shown as a dismissable banner in the
/// export bar (and used to fire the completion notification).
struct BatchSummary {
    done: usize,
    failed: usize,
    elapsed: std::time::Duration,
}

/// UI language. English is the source of truth; `tr` maps the core chrome
/// strings to Chinese, falling back to the English literal for anything not
/// in the table (so untranslated strings degrade gracefully).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lang {
    En,
    Zh,
}

/// Current UI language, stored as a process global so `tr` can be called
/// anywhere (in any egui closure) without threading the language through.
/// Set once per frame at the top of `update` from `self.lang`.
static UI_LANG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn cur_lang() -> Lang {
    if UI_LANG.load(std::sync::atomic::Ordering::Relaxed) == 1 { Lang::Zh } else { Lang::En }
}
fn set_cur_lang(l: Lang) {
    UI_LANG.store(if l == Lang::Zh { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
}

/// Translate a (static English) UI string for the current language.
fn tr(en: &'static str) -> &'static str {
    if cur_lang() == Lang::En {
        return en;
    }
    match en {
        // Top bar
        "Load video…" => "加载视频…",
        "Drop a video file here, or click Load video." => "将视频文件拖到此处，或点击“加载视频”。",
        "Loading" => "加载中",
        "Loading stabilization data…" => "正在加载防抖数据……",
        "Reading from the source — network drives can take a moment." => "正在读取源文件——网络驱动器可能需要片刻。",
        "Preview size" => "预览尺寸",
        // Side-panel section headers
        "Source" => "素材",
        "Color" => "调色",
        "Noise Reduction" => "降噪",
        "Detail" => "细节",
        "Audio" => "音频",
        "Stabilization" => "防抖",
        "Rolling shutter" => "果冻效应",
        "View adjust" => "视角调整",
        "View adjustment" => "视角调整",
        "Fisheye lens" => "鱼眼镜头",
        "Input" => "输入",
        "Dewarp fisheye input" => "对鱼眼输入去畸变",
        "On: each half is a raw fisheye image — dewarp it with the Fisheye lens settings. Off (default): the file is already VR180 side-by-side half-equirect and is used as-is." => "开：左右两半是原始鱼眼图像，按“鱼眼镜头”设置去畸变。关（默认）：文件已经是 VR180 左右半等距柱状格式，按原样使用。",
        "Fisheye dewarp is off for this side-by-side file (Source ▸ Input) — the lens settings below are not applied." => "此左右格式文件已关闭鱼眼去畸变（素材 ▸ 输入）——下方镜头设置不生效。",
        "Output" => "输出",
        "View" => "视图",
        "📷 Snapshot" => "📷 截图",
        "Save this frame as a JPEG in the export folder (S)" => "把当前帧存为 JPEG 到导出文件夹（S）",
        "Nothing to snapshot yet" => "还没有可截图的画面",
        "Saved" => "已保存",
        "Rendering snapshot" => "正在渲染截图",
        "preview size — full-size still never arrived" => "预览尺寸——未能获得全尺寸画面",
        "Snapshot failed" => "截图失败",
        "3D display" => "3D 显示器",
        "Show the stereo pair on a side-by-side 3D monitor or AR glasses that appear as one wide screen (3840×1080): left eye in the left half, right eye in the right. Goes fullscreen on that screen automatically when one is connected; otherwise opens a window to drag onto it (F = fullscreen). Esc closes it." => "在以单个宽屏（3840×1080）形式出现的左右格式 3D 显示器或 AR 眼镜上显示立体画面：左半为左眼，右半为右眼。检测到这样的屏幕时自动在其上全屏；否则打开一个窗口，拖到该屏幕后按 F 全屏。按 Esc 关闭。",
        "SBS while the 3D display is on" => "3D 显示器开启时固定为 SBS",
        "3D display — play a clip to see it here" => "3D 显示器——播放素材后在此显示",
        "Drag this window onto the 3D screen · F = fullscreen · Esc = close" => "把此窗口拖到 3D 屏幕上 · F = 全屏 · Esc = 关闭",
        "Click ▶ Play to start preview." => "点击 ▶ 播放以开始预览。",
        // Source clip list
        "+ Add clips…" => "+ 添加片段…",
        "Apply settings to all" => "应用设置到全部",
        "Remove from list" => "从列表移除",
        "Re-arm for export" => "重新加入导出",
        "Tick to include in \"Export selected\"" => "勾选以加入“导出所选”",
        // Export bar
        "Export all" => "全部导出",
        "Export selected" => "导出所选",
        "Stop" => "停止",
        "Skip current" => "跳过当前",
        "next to source" => "源文件旁",
        "starting…" => "开始中…",
        "Output folder for exports" => "导出文件夹",
        "Choose folder…" => "选择文件夹…",
        "Next to each source" => "各源文件旁",
        "Output format, resolution & metadata" => "输出格式、分辨率与元数据",
        "Export format" => "导出格式",
        "Export the ticked clips" => "导出已勾选的片段",
        "Export the active clip" => "导出当前片段",
        "Click to dismiss" => "点击关闭",
        // Transport
        "▶ Play" => "▶ 播放",
        "Pause" => "暂停",
        "■ Stop" => "■ 停止",
        // Noise Reduction panel
        "Strength" => "强度",
        "Reset" => "重置",
        // Export format window (export_options_ui)
        "Resolution" => "分辨率",
        "Still format" => "照片格式",
        "Still photo — no timeline" => "静态照片——没有时间线",
        "Restart preview" => "重启预览",
        "Recorded in 360° Audio: first-order ambisonics (AmbiX). A plain copy plays lopsided, so Stereo decodes a front-facing pair from it." => "以 360° 音频录制：一阶 Ambisonics（AmbiX）。直接拷贝播放会左右失衡，所以立体声是从它解码出的朝前声像对。",
        "Stereo AAC decoded from the ambisonics (virtual cardioids ±45°, camera front) — universal." => "由 Ambisonics 解码的立体声 AAC（±45° 虚拟心形，以相机正前为准）——通用。",
        "Shared by this camera's video and stills — one camera, one eye alignment." => "同一台相机的视频与照片共用——同一台机器，同一个双眼错位。",
        "Still photo — use Export for the full-size file" => "静态照片——请用导出获得全尺寸文件",
        "DNG output: the 3D LUT, sharpening and automatic eye matching cannot travel as DNG metadata — hidden, and not applied to the preview." => "DNG 输出：3D LUT、锐化和自动眼间匹配无法作为 DNG 元数据保存——已隐藏，预览中也不应用。",
        "Still" => "照片",
        "Your grade travels as metadata: white balance → AsShotNeutral, tone (lift/gamma/gain/shadow/highlight) → profile tone curve, saturation → profile look table (Adobe); per-eye trims are baked. Not carried: 3D LUT, sharpen." => "调色以元数据形式写入：白平衡 → AsShotNeutral，影调（提升/伽马/增益/阴影/高光）→ 配置文件色调曲线，饱和度 → 配置文件 Look 表（Adobe）；双眼微调会烘入像素。不带入：3D LUT、锐化。",
        "single photo, 14-bit raw" => "单张照片，14 位 RAW",
        "DNG keeps the camera-native linear data with the raw colour tags — grade it in a raw developer." => "DNG 保留相机原生线性数据和 RAW 色彩标签——可在 RAW 软件里继续调色。",
        "Still photo — no stabilization. Use the View adjust / stereo controls to level and align." => "静态照片——没有防抖。用视角调整 / 立体控制来调平和对齐。",
        "Codec" => "编码",
        "Bitrate (Mbps)" => "码率 (Mbps)",
        "Bit depth" => "位深",
        "ProRes profile" => "ProRes 配置",
        "VR180 metadata" => "VR180 元数据",
        "H.265 bitrate" => "H.265 码率",
        "Recommended" => "推荐",
        "Use recommended" => "使用推荐值",
        "Encoder" => "编码器",
        "10-bit (Main10)" => "10 位 (Main10)",
        "8-bit (Main)" => "8 位 (Main)",
        "Hardware (NVENC)" => "硬件 (NVENC)",
        "Software (libx265)" => "软件 (libx265)",
        "Camera baseline" => "相机基线",
        "None (no VR180 metadata)" => "无（不写入 VR180 元数据）",
        "Eye scale" => "单眼缩放",
        // Auto align (View adjustment panel)
        "Auto align" => "自动对齐",
        "Aligning…" => "对齐中…",
        "Auto align failed" => "自动对齐失败",
        "Measures the disparity between the eyes on a few frames and auto-fills the stereo pitch/roll/yaw. Adds to the current values — tweak after, or run again to refine."
            => "在若干帧上测量双眼间的视差，自动填充立体俯仰/滚转/偏航偏移。在当前值基础上累加——可再手动微调，或再次运行以进一步收敛。",
        "Scales each eye about its own center at the final output stage, padding with black. Half-equirect (VR180) output only."
            => "在最终输出阶段将每只眼睛的画面围绕各自中心缩放，四周填充黑色。仅对半等距柱状（VR180）输出生效。",
        // Source info
        "Stream" => "视频流",
        "FPS" => "帧率",
        "Duration" => "时长",
        "EAC tile" => "EAC 瓦片",
        "Eye dims" => "单眼尺寸",
        "No file loaded." => "未加载文件。",
        // Stabilization
        "Enable stabilization" => "启用防抖",
        "CORI source" => "CORI 来源",
        "Smooth (ms)" => "平滑 (ms)",
        "Camera lock" => "锁定视角",
        "Lock the view to the first frame; smoothing and max correction are ignored." => "锁定到首帧视角；平滑与最大校正均忽略。",
        // Reframed view
        "Reframed view" => "重新取景视图",
        "Aspect" => "画幅比例",
        "Zoom — horizontal FOV (°)" => "变焦 — 水平视场角 (°)",
        "Defish (0 = rectilinear, 1 = fisheye)" => "去鱼眼（0 = 直线透视，1 = 鱼眼）",
        "Pan (°)" => "平移 (°)",
        "Tilt (°)" => "俯仰 (°)",
        "Field of view" => "视场角",
        "Reset view" => "重置视图",
        "FOV" => "视场角",
        "pan" => "平移",
        "tilt" => "俯仰",
        "drag = pan, scroll / pinch = zoom, dbl-click = recenter" => "拖动 = 平移，滚轮 / 捏合 = 缩放，双击 = 回中",
        "Per-eye size" => "单眼尺寸",
        "Reframed output is not VR180 — no projection metadata is written." => "重新取景输出不是 VR180 —— 不写入投影元数据。",
        "Reframed mode aims the view with the Reframed view pan; these angles apply to VR180 output only."
            => "重新取景模式下由“重新取景视图”的平移控制朝向；这些角度仅作用于 VR180 输出。",
        "Max corr (°)" => "最大校正 (°)",
        "Response" => "响应",
        "direct" => "直接",
        "vqf" => "VQF",
        "auto" => "自动",
        // Audio
        "Export track" => "导出音轨",
        // Rolling shutter
        "Enable RS correction" => "启用果冻效应校正",
        "Mode" => "模式",
        "firmware" => "固件",
        "no-firmware" => "无固件",
        // Color
        "Tone (CDL)" => "影调 (CDL)",
        "White balance" => "白平衡",
        "Lift" => "提升",
        "Gamma" => "伽马",
        "Gain" => "增益",
        "Shadow" => "阴影",
        "Highlight" => "高光",
        "Temperature" => "色温",
        "Tint" => "色调",
        "Saturation" => "饱和度",
        "Intensity" => "强度",
        "Sharpen" => "锐化",
        "Radius" => "半径",
        "Reset tone" => "重置影调",
        "Reset color" => "重置色彩",
        // Blackmagic RAW (the SDK's colour processing)
        "Reset to clip" => "恢复素材原值",
        "Developed by the Blackmagic RAW SDK from the sensor data — these replace the Tone and White balance controls for this camera. A dot marks a value changed from the clip's."
            => "由 Blackmagic RAW SDK 直接从传感器数据解算——对此机型它们取代“影调”和“白平衡”控件。带点的项表示已偏离素材原值。",
        "Color science" => "色彩科学",
        "Gamut" => "色域",
        "Exposure (stops)" => "曝光（档）",
        "Highlight recovery" => "高光恢复",
        "Gamut compression" => "色域压缩",
        "Tone curve" => "色调曲线",
        "Contrast" => "对比度",
        "Midpoint" => "中点",
        "Highlights" => "高光",
        "Shadows" => "阴影",
        "Black level" => "黑电平",
        "White level" => "白电平",
        "Video black level" => "视频黑电平",
        "The clip's Film→video LUT fits the Film gamma only — it is switched off while another gamma is selected above."
            => "素材自带的 Film→视频 LUT 只适用于 Film 伽马——上方选择其他伽马时会自动关闭。",
        "Use the clip's embedded Film→video LUT" => "使用素材内嵌的 Film→视频 LUT",
        "Blackmagic RAW colour controls need the clip probe's colour block — rebuild the helper (helpers/braw/build_braw.sh)."
            => "Blackmagic RAW 调色控件需要探测结果中的颜色信息——请重新编译 helper（helpers/braw/build_braw.sh）。",
        // Matching Eyes (inter-eye trim)
        "Matching Eyes" => "双目匹配",
        "Fixes a difference between the two lenses — CT, Tint and Exposure apply oppositely to each eye."
            => "修正两个镜头之间的差异——色温、色调和曝光对两眼反向施加。",
        "Eye CT (±)" => "双目色温 (±)",
        "Eye Tint (±)" => "双目色调 (±)",
        "Eye Exposure (±EV)" => "双目曝光 (±EV)",
        "Reset matching" => "重置匹配",
        "Auto-match eye exposure (per frame)" => "逐帧自动匹配双目曝光",
        "The X6's two sensors auto-expose independently; this measures the brightness difference between the eyes every frame and cancels it before the LUT."
            => "X6 的两个传感器独立自动曝光；此功能逐帧测量两眼亮度差，并在套用 LUT 之前抵消。",
        "Reference" => "基准",
        "Meet in the middle" => "两眼折中",
        "Match to left eye" => "以左眼为准",
        "Match to right eye" => "以右眼为准",
        "Measured L−R" => "实测 左−右",
        "applied" => "已施加",
        "Measuring…" => "测量中…",
        "Applied per frame — the live readout shows while previewing." => "逐帧施加中——实测值在预览时显示。",
        "No usable content in the frame (clipped or too dark) — no correction applied." => "画面中没有可用内容（过曝或过暗）——未施加校正。",
        "Browse…" => "浏览…",
        "Clear" => "清除",
        "Use built-in Osmo 360 D-LogM→709" => "使用内置 Osmo 360 D-LogM→709",
        "Use built-in GoPro GP-Log→709" => "使用内置 GoPro GP-Log→709",
        "Use built-in Insta360 X6 I-Log→709" => "使用内置 Insta360 X6 I-Log→709",
        "Reset detail" => "重置细节",
        // View adjust
        "Global (both eyes)" => "全局（双眼）",
        "Yaw (°)" => "偏航 (°)",
        "Pitch (°)" => "俯仰 (°)",
        "Roll (°)" => "翻滚 (°)",
        "Stereo offset (right = +, left = −)" => "立体偏移（右 = +，左 = −）",
        "Tip: select a field and press ↑ / ↓ to step its value precisely." => "提示：选中字段后按 ↑ / ↓ 键可精细微调数值。",
        "Reset to 0" => "重置为 0",
        // Auto-update UI
        "Software update" => "软件更新",
        "Installed:" => "已安装：",
        "New version:" => "新版本：",
        "Install & Restart" => "安装并重启",
        "Skip this version" => "跳过此版本",
        "Check for updates" => "检查更新",
        "You're on the latest version." => "已是最新版本。",
        "What's new in" => "更新内容",
        "update" => "更新",
        "Downloading…" => "下载中…",
        "Installing — the app will restart…" => "正在安装 — 应用即将重启…",
        "Check failed:" => "检查失败：",
        "No update feed published yet." => "尚未发布更新源。",
        "Update failed:" => "更新失败：",
        // Fisheye output / stab / lens
        "Format" => "格式",
        "Swap L↔R eyes" => "交换左右眼",
        "Upside-down mount (180°)" => "倒装 (180°)",
        "Stabilization (VQF 6D, BRAW)" => "防抖 (VQF 6D, BRAW)",
        "Stabilization (DJI camera quats)" => "防抖（DJI 相机四元数）",
        "Stabilization (Insta360 gyro)" => "防抖（Insta360 陀螺仪）",
        "No gyro stabilization available for this source yet." => "此素材暂无陀螺仪防抖。",
        "Camera" => "相机",
        "(Auto)" => "（自动）",
        // Status bar
        "▶ playing" => "▶ 播放中",
        "paused" => "已暂停",
        // Import-merge modal
        "Import — merge segments?" => "导入 — 合并片段？",
        "files look like one continuous recording:" => "个文件像是同一段连续录制：",
        "Merge into one clip" => "合并为一个片段",
        "Import individually" => "分别导入",
        "Cancel" => "取消",
        // Audio hints
        "Stereo AAC — universal compatibility." => "立体声 AAC — 通用兼容。",
        "4-channel 1st-order ambisonic (AmbiX) + SA3D — YouTube VR / Quest head-track it."
            => "4 声道一阶 Ambisonic (AmbiX) + SA3D — YouTube VR / Quest 可头部跟踪。",
        "Apple Positional Audio — Vision Pro head-tracked spatial (macOS only; re-encodes the ambisonic)."
            => "Apple 位置音频 — Vision Pro 头部跟踪空间音频（仅 macOS；会重新编码 Ambisonic）。",
        // Noise reduction explainer
        "Temporal noise reduction (Apple VideoToolbox). Averages sensor noise across neighbouring frames before projection — most effective on low-light / high-ISO footage. Runs fully in 10-bit."
            => "时域降噪（Apple VideoToolbox）。投影前对相邻帧的传感器噪点取平均 — 对低光 / 高 ISO 素材最有效。全程以 10 位运行。",
        "⚠ Applied on export only — not shown in the preview." => "⚠ 仅在导出时应用 — 预览中不显示。",
        "⚠ Slows the export down (Apple's temporal filter runs at full source resolution)." => "⚠ 会拖慢导出速度（Apple 的时域滤波器以完整源分辨率运行）。",
        // Import-merge body
        "Merge plays + exports each recording as ONE clip with continuous stabilization across the join. Individual imports every file as its own clip."
            => "合并：将每段录制作为一个片段播放与导出，接合处连续防抖。分别：每个文件各自作为独立片段导入。",
        // Long help tooltips / notes
        "APMP and Spatial V2 conflict in the same atoms, so only one can be active per file. Re-run the export with the other target if you need a second copy."
            => "APMP 与 Spatial V2 占用相同的 atom，因此每个文件只能启用其一。如需第二份副本，请改用另一种目标重新导出。",
        "Copy the active clip's settings to every clip OF THE SAME TYPE (OSV→OSV, .360→.360, …) — other camera types keep their own settings. Per-clip trim & RS mode are kept."
            => "将当前片段的设置复制到所有相同类型的片段（OSV→OSV、.360→.360 ……）— 其他相机类型保留各自的设置。各片段的裁剪与卷帘快门模式保持不变。",
        "Velocity response curve: <1 follows motion early, 1 linear, >1 holds longer then catches up (cinematic lag)."
            => "速度响应曲线：<1 提前跟随运动，1 为线性，>1 保持更久后再追上（电影感延迟）。",
        "Settings apply live during playback — the decoder rebuilds per-eye bundles on the next iteration (small stutter ~300 ms on long clips)."
            => "设置在播放时实时生效 — 解码器会在下一次迭代重建每只眼的数据包（长片段会有约 300 毫秒的轻微卡顿）。",
        "Equirect-aware unsharp mask (cos-latitude weighted). 0.5 subtle, 1.0 moderate, 2.0 strong."
            => "等距柱状感知的 USM 锐化（按纬度余弦加权）。0.5 轻微，1.0 适中，2.0 强烈。",
        "With Override off, each eye uses the in-file (OSV) / preset calibration. Turn Override on for an eye to set its FOV, principal point and distortion by hand."
            => "关闭“覆盖”时，每只眼使用文件内置（OSV）/ 预设标定。为某只眼开启“覆盖”后可手动设置其 FOV、主点与畸变。",
        // Short hover tips + trim transport
        "Switch eye (keeps zoom/pan)" => "切换眼别（保留缩放/平移）",
        "Click to make this the active clip" => "点击以设为当前片段",
        "[ Mark In" => "[ 标记入点",
        "Mark Out ]" => "标记出点 ]",
        "Clear trim" => "清除裁剪",
        "Set trim-in to current playhead (I)" => "将入点设为当前播放头（I）",
        "Set trim-out to current playhead (O)" => "将出点设为当前播放头（O）",
        "Remove both in/out points" => "清除入点与出点",
        _ => en,
    }
}

/// Load a system CJK font as a FALLBACK so Chinese (中文) renders instead of
/// tofu boxes. Pushed after the default Latin fonts for both families, so
/// Latin keeps the bundled font and only CJK glyphs fall through to this.
/// No-op (logs a warning) if no system Chinese font is found.
fn install_cjk_font(ctx: &egui::Context) {
    // (path, face index within a .ttc collection). Every face in these is a
    // Chinese face, so index 0 always yields CJK glyphs.
    let candidates: &[(&str, u32)] = &[
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", 0),
        ("C:\\Windows\\Fonts\\msyh.ttc", 0),
        ("C:\\Windows\\Fonts\\simhei.ttf", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ];
    let Some((path, bytes, index)) = candidates.iter().find_map(|(p, i)| {
        std::fs::read(p).ok().map(|b| (*p, b, *i))
    }) else {
        tracing::warn!("CJK: no system Chinese font found — 中文 will show as boxes");
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    let mut fd = egui::FontData::from_owned(bytes);
    fd.index = index;
    // CJK faces sit lower than egui's Latin baseline (more descent), so mixed
    // rows look misaligned. The needed nudge is font-specific: macOS PingFang
    // centres at 0.24, but Windows' Microsoft YaHei (msyh) is already aligned
    // there — 0.24 pushes 中文 too low. Windows → 0; macOS untouched.
    let y_offset_factor = if cfg!(target_os = "windows") { 0.0 } else { 0.24 };
    fd.tweak = egui::FontTweak {
        y_offset_factor,
        ..Default::default()
    };
    fonts.font_data.insert("cjk".to_owned(), std::sync::Arc::new(fd));
    for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(fam).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
    tracing::info!("CJK: loaded fallback font {path}");
}

struct DisplayFrame {
    texture: Arc<wgpu::Texture>,
    egui_id: egui::TextureId,
    width: u32,
    height: u32,
    frame_idx: u32,
    timestamp_s: f64,
}

/// External **3D display** output: a second, chrome-less viewport that shows
/// the SBS preview with the left eye in the left half and the right eye in
/// the right half — the stereo input a side-by-side 3D monitor or AR glasses
/// (one wide screen such as 3840×1080) take. See `draw_3d_display`.
#[derive(Default)]
struct ThreeDDisplay {
    enabled: bool,
    displays: Vec<crate::displays::DisplayInfo>,
    /// Index into `displays` the output sits on; `None` → the windowed
    /// fallback (drag it onto the 3D screen, `F` = fullscreen).
    target: Option<usize>,
    /// Current fullscreen state (`F` toggles it in the output window).
    fullscreen: bool,
    /// Auto-placed output wants NATIVE fullscreen (macOS with separate
    /// Spaces per display — a borderless window would sit under that
    /// display's menu bar). It is requested in a second step, once the
    /// borderless window is confirmed on the target screen, so macOS
    /// fullscreens THAT screen (asking for it at creation fullscreened
    /// the main screen and covered the app).
    native_fullscreen: bool,
    /// The two-step fullscreen request has been sent.
    fullscreen_sent: bool,
    /// Settings changed on enable, restored on disable.
    restore_preview_w: Option<u32>,
    restore_preview_mode: Option<crate::decoder::PreviewMode>,
}

/// Largest rect of `aspect` (w/h) centred inside `outer`.
fn fit_aspect(outer: egui::Rect, aspect: f32) -> egui::Rect {
    let (ow, oh) = (outer.width(), outer.height());
    let (w, h) = if ow / oh.max(1e-3) > aspect { (oh * aspect, oh) } else { (ow, ow / aspect.max(1e-3)) };
    egui::Rect::from_center_size(outer.center(), egui::vec2(w, h))
}

/// A snapshot in flight: the still path renders the paused frame at the
/// export's per-eye size `out`; `poll_snapshot` writes it once it lands.
struct SnapshotReq {
    out: (u32, u32),
    since: std::time::Instant,
}

#[derive(Default)]
struct FpsStats {
    frames_in_window: u32,
    last_window_start: Option<std::time::Instant>,
    last_fps: f32,
}

/// One recording detected among the files the user just picked. GoPro
/// `.360` members are auto-detected chapters (`detect_segments`); OSV
/// members are exactly the files the user multi-selected (OSV segments are
/// indistinguishable from separate clips by name, so the user chooses
/// them). A group with >1 member is offered "merge into one clip vs import
/// individually".
struct ImportGroup {
    /// Files the user actually selected for this group (segment order).
    picked: Vec<PathBuf>,
    /// Full recording in segment order — chapters for GoPro, == picked for OSV.
    members: Vec<PathBuf>,
    /// Short human label for the prompt (e.g. "GoPro · 4 chapters").
    label: String,
}

impl ImportGroup {
    fn mergeable(&self) -> bool { self.members.len() > 1 }
}

/// An import awaiting the user's merge/individual decision (modal).
struct PendingImport {
    groups: Vec<ImportGroup>,
    /// First picked path — re-activated after the import completes.
    first: PathBuf,
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
    /// GoPro `.360` GEOC calibration dimension (square sensor cal space,
    /// 4216 for GoPro Max). `None` for non-EAC sources or when the file
    /// has no GEOC. The ".360 Lens calibration" panel displays cx/cy in
    /// this pixel space (matching CTRX/CTRY units).
    eac_cal_dim: Option<u32>,
    /// Insta360 X6 recorded in "360° Audio" mode: the source carries a
    /// 4-channel first-order AmbiX track (so the Audio section applies).
    foa_audio: bool,
    /// Kandao QooCam with its 4-channel microphone track (mixed to stereo
    /// for playback and export — `audio::QuadEncoder`).
    quad_audio: bool,
    /// Blackmagic RAW: the SDK's colour processing as shot (values, ranges,
    /// lists) for the Color panel's Blackmagic RAW section.
    braw_color: Option<std::sync::Arc<vr180_braw::BrawColorInfo>>,
    /// Blackmagic RAW: the clip's embedded display LUT written out as a
    /// `.cube` (auto-applied; switched off under a non-Film gamma).
    braw_lut: Option<String>,
}

/// An async clip load in flight (see `App::load_pending`). The worker
/// thread does all the blocking source I/O and posts a [`LoadOutcome`]
/// back on `rx`; the UI thread shows a "Loading…" state until it arrives.
struct LoadPending {
    token: u64,
    path: PathBuf,
    rx: Receiver<LoadOutcome>,
    started: std::time::Instant,
}

/// Result of the off-thread metadata read: either the resolved clip
/// metadata to apply, or a human-readable error string to log.
type LoadOutcome = std::result::Result<LoadedMeta, String>;

/// One BRAW eye at load: the layout's (memoised per clip; side by side
/// unless the first frame holds one centred circle), from the probe's
/// composed frame size.
fn braw_eye_dims(path: &std::path::Path, frame_w: u32, frame_h: u32) -> (u32, u32) {
    match vr180_braw::BrawInfo::probe(path) {
        Ok(info) => vr180_pipeline::braw_imu::braw_layout(path, &info).eye_dims(&info),
        Err(_) => (frame_w / 2, frame_h),
    }
}

/// Everything `finish_load` needs to build the `ClipInfo`, seed per-clip
/// settings, and spawn the decoder — all gathered off the UI thread.
struct LoadedMeta {
    token: u64,
    path: PathBuf,
    preserve_clip_settings: bool,
    source_kind: vr180_pipeline::SourceKind,
    width: u32,
    height: u32,
    fps: f32,
    duration_sec: f64,
    frame_count: u32,
    segments: Vec<PathBuf>,
    cori_count: usize,
    grav_count: usize,
    srot_ms: Option<f32>,
    detected_rs_mode: crate::decoder::RsMode,
    fisheye_eye_w: u32,
    fisheye_eye_h: u32,
    eac_cal_dim: Option<u32>,
    /// Insta360 X6 recorded in "360° Audio" mode: the source carries a
    /// 4-channel first-order AmbiX track (so the Audio section applies).
    foa_audio: bool,
    /// Kandao QooCam 4-channel microphone track (see `ClipInfo::quad_audio`).
    quad_audio: bool,
    /// The stream is HLG (the QooCam's HDR mode) — auto-apply the HLG LUT.
    hlg: bool,
    /// Blackmagic RAW: the clip's embedded display LUT written out as a
    /// `.cube` by the helper (off the UI thread), to auto-apply.
    braw_lut: Option<String>,
    /// Blackmagic RAW: the SDK's colour processing as shot (the probe's
    /// `color` block), for the Color panel.
    braw_color: Option<vr180_braw::BrawColorInfo>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Force dark visuals regardless of the OS appearance preference.
        // egui's dark theme reads better against the half-equirect
        // preview (which is typically dark imagery itself) and matches
        // the rest of the colorist tooling our users come from
        // (DaVinci, FCP, Premiere, Insta360 Studio — all dark).
        // egui 0.34 follows the OS theme by default, so forcing the
        // visuals alone isn't enough — pin the theme *preference* to Dark
        // (matches the previous build, which was always dark).
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        // egui's bundled fonts have no CJK glyphs, so 中文 renders as tofu
        // boxes — load a system Chinese font as a fallback.
        install_cjk_font(&cc.egui_ctx);

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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let pipeline = vr180_pipeline::gpu::Device::from_existing(
            instance,
            std::sync::Arc::new(wgpu_state.adapter.clone()),
            std::sync::Arc::new(wgpu_state.device.clone()),
            std::sync::Arc::new(wgpu_state.queue.clone()),
        ).expect("pipeline device init from shared wgpu device");
        let pipeline = Arc::new(pipeline);
        tracing::info!(
            backend = ?pipeline.adapter.get_info().backend,
            name = %pipeline.adapter.get_info().name,
            "pipeline running on the eframe-owned wgpu device"
        );
        // The egui surface (swapchain) format decides how egui encodes its
        // output to the screen, which in turn decides whether our
        // Rgba8UnormSrgb texture *view* (sRGB-decode-on-sample) is cancelled
        // by a matching encode-on-store. On macOS this is *Srgb (validated
        // correct); if Windows/Vulkan hands back a non-sRGB surface the decode
        // isn't re-encoded and the preview shows sRGB⁻¹(v) — too dark.
        tracing::info!(
            target_format = ?wgpu_state.target_format,
            is_srgb = wgpu_state.target_format.is_srgb(),
            "egui surface (swapchain) format"
        );

        // Restore the user's last-used settings from disk (defaults on first
        // run). Persisted across launches so every knob is remembered.
        let defaults = Settings::load_persisted();
        // egui-wgpu 0.34's fragment shader (both the linear- and
        // gamma-framebuffer variants) explicitly "expect normal textures that
        // are NOT sRGB-aware": it samples them as gamma/raw bytes, and on an
        // sRGB swapchain it linearizes at the very end so the hardware
        // re-encode is identity. Our SBS already holds Rec.709-gamma bytes, so
        // we register it with a PLAIN Unorm view on every platform. An sRGB
        // *view* would sRGB-decode the sample; egui then treats that linear
        // value as gamma and displays sRGB⁻¹(v) — too dark (the Windows
        // symptom; the old sRGB view was right only under egui-wgpu 0.28).
        let _ = wgpu_state.target_format; // logged above; view choice is fixed
        let preview_view_format = wgpu::TextureFormat::Rgba8Unorm;

        // Auto-update: silent check on launch (errors swallowed — the
        // user can always check manually via the version label).
        let (updater_tx, updater_rx) = crossbeam_channel::unbounded();
        crate::updater::spawn_check(updater_tx.clone());
        // Worker-thread file pickers (see the `dialog_tx` field comment).
        let (dialog_tx, dialog_rx) = crossbeam_channel::unbounded();

        Self {
            egui_ctx: cc.egui_ctx.clone(),
            pipeline,
            egui_renderer: wgpu_state.renderer.clone(),
            preview_view_format,
            loaded_path: None,
            pending_open: std::env::args_os().skip(1).map(PathBuf::from)
                .filter(|p| p.is_file()).collect(),
            eye_orientation_applied: (defaults.fisheye_swap_eyes, defaults.camera_upside_down),
            clip: None,
            playing: false,
            decoder_alive: false,
            no_decoder_warned: false,
            still_native_wanted: false,
            decoder_starting: false,
            decoder_retried: false,
            current_display: None,
            frame_rx: None,
            detail_frame_rx: None,
            cmd_tx: None,
            control: None,
            fps_stats: FpsStats::default(),
            settings: defaults.clone(),
            last_saved_settings: defaults.clone(),
            last_settings_save_at: std::time::Instant::now(),
            kind_settings: Settings::load_kind_map(),
            last_pushed_settings: defaults,
            export_job: None,
            updater_tx,
            updater_rx,
            dialog_tx,
            dialog_rx,
            dialog_open: false,
            update_available: None,
            update_progress: None,
            update_installing: false,
            update_status: None,
            show_update_popover: false,
            update_manual_check: false,
            update_prompted_version: None,
            update_last_check: std::time::Instant::now(),
            updater_prefs: crate::updater::load_prefs(),
            pending_update_install: None,
            current_version_notes: None,
            audio_player: None,
            export_opts: ExportOptions::default(),
            batch: Vec::new(),
            active_clip: None,
            align_rx: None,
            align_status: None,
            batch_running: false,
            batch_current: None,
            batch_opts: None,
            batch_out_dir: None,
            batch_skip_flag: false,
            batch_run_indices: Vec::new(),
            batch_started_at: None,
            run_rate_anchor: None,
            last_batch_summary: None,
            show_export_settings: false,
            lang: Self::load_lang(),
            merge_groups: std::collections::HashMap::new(),
            pending_import: None,
            preview_zoom: 1.0,
            preview_center: egui::vec2(0.5, 0.5),
            full_res_display: None,
            three_d: ThreeDDisplay::default(),
            snapshot_status: None,
            snapshot_pending: None,
            full_res_key: 0,
            full_res_desired_key: 0,
            full_res_rendered_at: std::time::Instant::now(),
            detail_last_key: 0,
            detail_key_changed_at: std::time::Instant::now(),
            detail_cache: None,
            fisheye_last_seeded: None,
            suppress_reseed_once: false,
            load_pending: None,
            load_token: 0,
            play_when_ready: false,
        }
    }

    /// Pick an output path and spawn the export worker. No-op if no
    /// clip is loaded, an export is already in flight, or the source
    /// is GoPro EAC (the existing CLI handles that family).
    /// Top-toolbar Export button — opens the options window. The actual
    /// export kicks off from `commit_export()` once the user confirms.
    /// Export just the ACTIVE clip — a "batch of one". Routes through the
    /// SAME queue runner as "Export all": a single export is a one-item run,
    /// so there's one code path, one set of output settings, one progress UI.
    /// `…/VR180SilverBullet2.0/lang` — the UI-language preference file.
    fn lang_config_path() -> Option<PathBuf> {
        crate::decoder::Settings::config_path()
            .and_then(|p| p.parent().map(|d| d.join("lang")))
    }

    fn load_lang() -> Lang {
        match Self::lang_config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.trim().to_string())
            .as_deref()
        {
            Some("zh") => Lang::Zh,
            _ => Lang::En,
        }
    }

    fn save_lang(&self) {
        if let Some(p) = Self::lang_config_path() {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, if self.lang == Lang::Zh { "zh" } else { "en" });
        }
    }

    fn export_selected(&mut self) {
        if self.export_job.is_some() || self.batch_running { return; }
        self.sync_active_clip_settings();
        // The ticked clips; or, if none are ticked, the active clip alone.
        let mut run: Vec<usize> = self.batch.iter().enumerate()
            .filter(|(_, b)| b.selected && b.source_kind.is_exportable())
            .map(|(i, _)| i)
            .collect();
        if run.is_empty() {
            if let Some(idx) = self.active_clip {
                if self.batch.get(idx).map_or(false, |b| b.source_kind.is_exportable()) {
                    run.push(idx);
                }
            }
        }
        if run.is_empty() {
            tracing::warn!("export: nothing selected to export");
            return;
        }
        for &i in &run {
            self.batch[i].status = BatchStatus::Queued;
        }
        self.begin_run(run);
    }

    /// Common run setup: freeze the output options, record the run-set +
    /// start time (for overall progress / ETA / completion), kick the first
    /// queued item.
    fn begin_run(&mut self, run_indices: Vec<usize>) {
        if !self.batch.iter().any(|b| b.status == BatchStatus::Queued) { return; }
        self.batch_opts = Some(self.export_opts);
        self.batch_run_indices = run_indices;
        self.batch_started_at = Some(std::time::Instant::now());
        self.run_rate_anchor = None;
        self.last_batch_summary = None;
        self.batch_running = true;
        self.batch_skip_flag = false;
        self.start_next_batch_item();
    }

    /// macOS desktop notification when an export run finishes — people walk
    /// away during long batches (more so now that NR triples some clips).
    /// Fires only for multi-item or long (>20 s) runs so a quick single
    /// export doesn't nag. No-op off macOS.
    fn notify_run_complete(
        &self,
        done: usize,
        failed: usize,
        elapsed: std::time::Duration,
        n_run: usize,
    ) {
        #[cfg(target_os = "macos")]
        {
            if n_run <= 1 && elapsed.as_secs() < 20 {
                return;
            }
            let when = {
                let (m, s) = (elapsed.as_secs() / 60, elapsed.as_secs() % 60);
                if m > 0 { format!("{m}m {s}s") } else { format!("{s}s") }
            };
            let body = if failed > 0 {
                format!("{done} exported, {failed} failed · {when}")
            } else {
                format!("{done} exported · {when}")
            }
            .replace('"', "'");
            // osascript is always present on macOS; spawn detached so the UI
            // never blocks on it.
            let _ = std::process::Command::new("osascript")
                .arg("-e")
                .arg(format!(
                    "display notification \"{body}\" with title \"VR180 Silver Bullet\" \
                     sound name \"Glass\""
                ))
                .spawn();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (done, failed, elapsed, n_run);
        }
    }

    /// Frames one run item contributes to the totals: its TRIMMED length
    /// (every entry carries its own trim), or — for the item currently
    /// exporting — the pipeline's authoritative trim-aware count. Using the
    /// untrimmed clip length here was why a trimmed export's bar stalled
    /// short of 100% and its ETA ran long.
    fn run_item_frames(&self, i: usize, running: bool) -> u64 {
        if running {
            if let Some(p) = self.export_job.as_ref().and_then(|j| j.last_progress.as_ref()) {
                if p.total_frames > 0 { return p.total_frames; }
            }
        }
        self.batch.get(i).map(|b| {
            let t_in = b.settings.trim_in_s.unwrap_or(0.0).max(0.0);
            let t_out = b.settings.trim_out_s.unwrap_or(b.duration_sec).min(b.duration_sec);
            ((t_out - t_in).max(0.0) * b.fps as f64).round().max(1.0) as u64
        }).unwrap_or(1)
    }

    /// `(done_frames, total_frames, running_item_pos)` for the current run;
    /// `None` when no run is in flight.
    fn run_frame_tally(&self) -> Option<(u64, u64, usize)> {
        if !self.batch_running || self.batch_run_indices.is_empty() {
            return None;
        }
        let (mut done, mut total, mut item_pos) = (0u64, 0u64, 0usize);
        for (pos, &i) in self.batch_run_indices.iter().enumerate() {
            let status = self.batch.get(i).map(|b| &b.status);
            let running = matches!(status, Some(BatchStatus::Running));
            let frames = self.run_item_frames(i, running);
            total += frames;
            match status {
                Some(BatchStatus::Done | BatchStatus::Failed(_) | BatchStatus::Skipped) => {
                    done += frames;
                }
                Some(BatchStatus::Running) => {
                    item_pos = pos;
                    if let Some(p) = self.export_job.as_ref().and_then(|j| j.last_progress.as_ref()) {
                        done += p.frame_idx.min(frames);
                    }
                }
                _ => {}
            }
        }
        Some((done, total, item_pos))
    }

    /// Overall progress of the current run for the export bar:
    /// `(done_frames, total_frames, eta_secs, item_position, item_total)`.
    /// `None` when no run is in flight. The ETA rate is measured from the
    /// run's first written frame (see `run_rate_anchor`) across items, so
    /// slow-NR and fast plain clips self-correct and start-up time is never
    /// mistaken for encode time; before that anchor exists it falls back to
    /// the live encode rate.
    fn overall_run_progress(&self) -> Option<(u64, u64, Option<f64>, usize, usize)> {
        let (done_frames, total_frames, item_pos) = self.run_frame_tally()?;
        let remaining = total_frames.saturating_sub(done_frames);
        let anchored = self.run_rate_anchor.and_then(|(t0, d0)| {
            let secs = t0.elapsed().as_secs_f64();
            (done_frames > d0 && secs > 0.5)
                .then(|| remaining as f64 * secs / (done_frames - d0) as f64)
        });
        let eta = anchored.or_else(|| {
            self.export_job.as_ref()
                .and_then(|j| j.last_progress.as_ref())
                .map(|p| p.fps_avg)
                .filter(|f| *f > 0.1)
                .map(|f| remaining as f64 / f as f64)
        });
        Some((done_frames, total_frames, eta, item_pos + 1, self.batch_run_indices.len()))
    }

    /// `<source_stem>_SBS.<ext-for-codec>` next to the source.
    fn default_output_for(source: &std::path::Path, codec: ExportCodec) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let ext = match codec {
            ExportCodec::H265 => "mp4",
            ExportCodec::ProRes => "mov", // ProRes is conventional in MOV
        };
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_SBS.{ext}"))
    }

    /// `<source_stem>_SBS.<png|jpg|dng>` next to the source (stills).
    fn default_still_output_for(source: &std::path::Path, format: StillFormatChoice) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_SBS.{}", format.extension()))
    }

    /// Build the full `FisheyeExportConfig` for one (source, output) pair
    /// from explicit clip facts + a Settings/ExportOptions pair. Pure —
    /// used by both the single export (current clip + live settings) and
    /// the batch (per-item probe + per-item settings snapshot).
    #[allow(clippy::too_many_arguments)]
    fn build_export_cfg(
        source_path: &std::path::Path,
        output_path: &std::path::Path,
        fps: f32,
        source_kind: vr180_pipeline::SourceKind,
        fisheye_eye_w: u32,
        fisheye_eye_h: u32,
        width: u32,
        height: u32,
        settings: &crate::decoder::Settings,
        opts: &ExportOptions,
        segments: Vec<PathBuf>,
    ) -> vr180_pipeline::fisheye_export::FisheyeExportConfig {
        // Pick the backend based on the chosen codec + platform. On
        // Windows/Linux, H.265 defaults to NVENC (hardware) — libx265 is
        // the VR180 export bottleneck — but the user can pick software in
        // the options; either way `open_h265_encoder` falls back to libx265
        // if NVENC can't be opened. macOS always uses VideoToolbox.
        use vr180_pipeline::encode::EncoderBackend;
        let backend = match (opts.codec, cfg!(target_os = "macos")) {
            (ExportCodec::H265, true) => EncoderBackend::VideoToolbox,
            (ExportCodec::H265, false) =>
                if opts.h265_hardware {
                    EncoderBackend::HevcNvenc
                } else {
                    EncoderBackend::Libx265
                },
            (ExportCodec::ProRes, true) => EncoderBackend::ProResVideoToolbox,
            (ExportCodec::ProRes, false) => EncoderBackend::ProResKs,
        };

        // Output resolution depends on the projection target. For
        // half-equirect VR180 we use the source eye dims (or fallback
        // for non-fisheye sources). For fisheye pass-through we use
        // the raw source eye dims since the projection is bypassed.
        let (mut eye_w, mut eye_h) = if fisheye_eye_w > 0 {
            (fisheye_eye_w, fisheye_eye_h)
        } else if source_kind.is_eac() {
            // GoPro EAC: a SQUARE half-equirect per eye at the cross's
            // native resolution (cross_w = 2·tile_w + 1920; 3936 on Max).
            // Full quality vs the preview's ≤2048-clamped square.
            let cross_w = 2 * (width.saturating_sub(1920) / 4) + 1920;
            (cross_w.max(512), cross_w.max(512))
        } else {
            (width.max(512), height.max(512))
        };
        // 8K target: render the projection itself at 4096² per eye
        // (8192×4096 SBS). The projection samples the native source
        // directly into the 4096 grid — full resolution end-to-end,
        // not a last-step upscale.
        if opts.resolution == ExportResolution::R8k {
            eye_w = 4096;
            eye_h = 4096;
        }

        let projection = match settings.fisheye_output_mode {
            crate::decoder::FisheyeOutputMode::HalfEquirect =>
                vr180_pipeline::fisheye_export::FisheyeExportProjection::HalfEquirect,
            crate::decoder::FisheyeOutputMode::Fisheye =>
                vr180_pipeline::fisheye_export::FisheyeExportProjection::Fisheye,
            crate::decoder::FisheyeOutputMode::Reframe => {
                // The exact viewport at the chosen per-eye size and aspect
                // (overrides the VR180 resolution targets above); the
                // still-only sizes never reach a video export.
                let (w, h) = opts.reframe_size
                    .clamped(source_kind.is_still(), settings.reframe_aspect)
                    .eye_dims(settings.reframe_aspect);
                eye_w = w;
                eye_h = h;
                vr180_pipeline::fisheye_export::FisheyeExportProjection::Reframe {
                    hfov_deg: settings.reframe_hfov_deg,
                    defish: settings.reframe_defish,
                }
            }
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
        let mut color_stack = settings.build_color_stack();
        // Automatic eye-exposure matching is an X6 feature (independent
        // per-sensor AE); never let the persisted flag leak onto other sources.
        if !matches!(source_kind, vr180_pipeline::SourceKind::Insta360Insv) {
            color_stack.auto_eye_match = None;
        }
        // A still bound for a linear DNG: only what the DNG carries as
        // metadata. Decided from THIS export's options (a batch may use a
        // different still format than the live preview flag reflects).
        if source_kind.is_still() && opts.still_format == StillFormatChoice::LinearDng {
            crate::decoder::restrict_to_dng_metadata(&mut color_stack);
        }
        // "BeyondVR Hack" — final-stage per-eye scale. Half-equirect SBS
        // output only; the preview plan (built separately) never sets it.
        if opts.beyondvr_hack
            && projection == vr180_pipeline::fisheye_export::FisheyeExportProjection::HalfEquirect
        {
            color_stack.eye_scale = (opts.beyondvr_scale_pct / 100.0).clamp(0.25, 1.0);
        }

        // Bitrate / bit-depth come from the options window. ProRes
        // ignores the bitrate field (the profile picks the rate)
        // and is always 10-bit / 12-bit per profile.
        let (bitrate_kbps, bit_depth) = match opts.codec {
            ExportCodec::H265 => (
                // Reframed exports take the (much lower) reframed rate,
                // recommended per clip from its size / aspect / fps unless
                // the user pinned one; VR180 takes the VR180 slider.
                opts.effective_h265_mbps(settings.reframe_aspect_if_active(), fps)
                    .saturating_mul(1000),
                opts.h265_bit_depth,
            ),
            ExportCodec::ProRes => {
                // The encoder ignores bitrate for ProRes but the config
                // field is non-optional — pass a sane default.
                let pix_is_12bit = matches!(
                    opts.prores_profile,
                    ProResProfile::P4444 | ProResProfile::P4444Xq,
                );
                (0, if pix_is_12bit { 10 } else { 10 })
            }
        };

        vr180_pipeline::fisheye_export::FisheyeExportConfig {
            source_path: source_path.to_path_buf(),
            segments,
            // Reuse the IMU the preview already loaded for this clip (if cached),
            // so exporting a just-previewed file skips the djmd re-read.
            preloaded_imu: crate::decoder::cached_dji_imu(source_path),
            output_path: output_path.to_path_buf(),
            source_kind,
            audio_track: match settings.audio_format {
                crate::decoder::AudioFormat::Stereo    => vr180_pipeline::fisheye_export::AudioTrack::Stereo,
                crate::decoder::AudioFormat::Ambisonic => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
                // No APAC encoder on Windows: a persisted/carried-over Apac
                // becomes Ambisonic here too, so the export does not depend
                // on the Audio panel having been drawn (which is where the
                // picker folds it). The pipeline's own fallback stays as the
                // last net.
                #[cfg(target_os = "macos")]
                crate::decoder::AudioFormat::Apac      => vr180_pipeline::fisheye_export::AudioTrack::Apac,
                #[cfg(not(target_os = "macos"))]
                crate::decoder::AudioFormat::Apac      => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
            },
            eye_w,
            eye_h,
            fps,
            bitrate_kbps,
            encoder: backend,
            bit_depth,
            prores_profile: opts.prores_profile as i32,
            inject_apmp: opts.inject_apmp,
            inject_youtube_vr180: opts.inject_youtube,
            apmp_baseline_mm: opts.apmp_baseline_mm,
            stabilize: settings.stabilize,
            // Camera lock drives the pipeline's frame-0 lock via smooth_ms=0 and
            // drops the correction cap (0 → ∞ at the export sites): fully locked
            // no matter what the camera does. Soft-stab passes the window floored
            // off 0 (the lock sentinel) and the user's cap.
            dji_max_corr_deg: if settings.dji_camera_lock { 0.0 } else { settings.dji_max_corr_deg },
            dji_smooth_ms: if settings.dji_camera_lock { 0.0 } else { settings.dji_smooth_ms.max(1.0) },
            dji_responsiveness: settings.dji_responsiveness,
            denoise_strength: settings.denoise_strength,
            view_adjust: settings.output_view_adjust(),
            fisheye_preset: settings.fisheye_preset.clone(),
            sbs_dewarp_fisheye: settings.sbs_dewarp_fisheye,
            fisheye_override_left: settings.fisheye_override_left,
            fisheye_override_right: settings.fisheye_override_right,
            fisheye_fov_deg_left: settings.fisheye_fov_deg_left,
            fisheye_fov_deg_right: settings.fisheye_fov_deg_right,
            fisheye_k_left: settings.fisheye_k_left,
            fisheye_k_right: settings.fisheye_k_right,
            fisheye_k5_left: settings.fisheye_k5_left,
            fisheye_k5_right: settings.fisheye_k5_right,
            fisheye_p_left: settings.fisheye_p_left,
            fisheye_p_right: settings.fisheye_p_right,
            fisheye_cx_norm_left: settings.fisheye_cx_norm_left,
            fisheye_cy_norm_left: settings.fisheye_cy_norm_left,
            fisheye_cx_norm_right: settings.fisheye_cx_norm_right,
            fisheye_cy_norm_right: settings.fisheye_cy_norm_right,
            // Effective swap: manual toggle XOR upside-down mount.
            fisheye_swap_eyes: settings.effective_swap_eyes(),
            braw_color: settings.braw_color.clone(),
            trim_in_s: settings.trim_in_s,
            trim_out_s: settings.trim_out_s,
            color_stack,
            projection,
        }
    }

    /// Spawn the export worker thread for a prepared config and install
    /// it as the (single) running `export_job`.
    /// `eac_stab`: for a GoPro `.360` (EAC) export, the `(Settings,
    /// n_frames)` needed to compute the per-eye stabilization (the
    /// preview's `build_per_eye_frames`); `None` for fisheye sources,
    /// which compute stab inside `export_fisheye`.
    fn spawn_export_job(
        &mut self,
        cfg: vr180_pipeline::fisheye_export::FisheyeExportConfig,
        eac_stab: Option<(crate::decoder::Settings, usize)>,
    ) {
        let output_path = cfg.output_path.clone();
        let still_format = self.batch_opts.unwrap_or(self.export_opts).still_format;

        // Stop preview playback when it reads the SAME file as the export
        // (they'd race on the decoder / file). Previewing a DIFFERENT clip
        // is fine — that's what lets you keep tuning clips mid-batch.
        // A STILL is the exception: its preview thread read the DNG once
        // and now holds only textures, and the export re-reads the file
        // itself, so there is nothing to race on. Stopping it left the
        // photo frozen on the last render with no transport to restart it
        // (the still has no Play button) — the "UI stopped responding
        // after exporting" report.
        if self.loaded_path.as_ref() == Some(&cfg.source_path) && !cfg.source_kind.is_still() {
            self.stop_playback();
        }

        let (progress_tx, progress_rx) = crossbeam_channel::bounded(256);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        // The export worker runs GPU work + RGB48 readback (`Maintain::Wait`)
        // on a background thread. Lesson #1: doing that on the device eframe
        // presents with can wedge the present queue. Export never hands a
        // texture to egui (it reads back to CPU for the encoder), so on
        // Windows we give it a DEDICATED logical device on the same adapter —
        // fully isolated from eframe's present queue. Falls back to the shared
        // device if the dedicated one can't be created.
        #[cfg(target_os = "windows")]
        let pipeline = match vr180_pipeline::gpu::Device::new_dedicated_from_adapter(
            &self.pipeline.adapter,
        ) {
            Ok(d) => {
                tracing::info!(
                    "export: dedicated wgpu device (isolated from eframe present queue)"
                );
                Arc::new(d)
            }
            Err(e) => {
                tracing::warn!("export: dedicated device unavailable ({e}); sharing eframe device");
                self.pipeline.clone()
            }
        };
        #[cfg(not(target_os = "windows"))]
        let pipeline = self.pipeline.clone();
        let progress_tx2 = progress_tx.clone();
        let handle = std::thread::spawn(move || {
            if let Some((settings, n_frames)) = eac_stab {
                // GoPro EAC: compute the same per-eye stab the preview uses,
                // then run the EAC export. build_per_eye_frames_multi is
                // empty when stabilization + RS are both off → identity.
                // Multi-segment: span ALL segments (gyro aggregated as one).
                let total_n = if cfg.segments.len() > 1 {
                    crate::decoder::segments_total_frames(&cfg.segments)
                } else { n_frames };
                let per_eye = crate::decoder::build_per_eye_frames_multi(
                    &cfg.segments, &settings, cfg.fps, total_n,
                ).unwrap_or_default();
                vr180_pipeline::fisheye_export::export_eac(
                    pipeline, cfg, per_eye,
                    move |p| { let _ = progress_tx.try_send(p); },
                    cancel_for_thread,
                )
            } else if cfg.source_kind.is_still() {
                // Still photo: one frame through the raw stage + projection,
                // written as PNG / JPEG / linear DNG. Progress is reported as a
                // 0..100 "frame" count so the export bar needs no special case.
                let still_cfg = vr180_pipeline::still_export::StillExportConfig {
                    source_path: cfg.source_path.clone(),
                    output_path: cfg.output_path.clone(),
                    format: still_format.to_pipeline(),
                    eye_w: cfg.eye_w, eye_h: cfg.eye_h,
                    view_adjust: cfg.view_adjust,
                    swap_eyes: cfg.fisheye_swap_eyes,
                    develop: vr180_pipeline::raw_develop::DevelopParams::default(),
                    color_stack: cfg.color_stack.clone(),
                    projection: cfg.projection,
                    lens_override: vr180_pipeline::still_export::StillLensOverride::from_export_cfg(&cfg),
                };
                let t0 = std::time::Instant::now();
                vr180_pipeline::still_export::export_still(pipeline, &still_cfg, &|_stage, f| {
                    let done = (f * 100.0).round() as u64;
                    let _ = progress_tx2.try_send(vr180_pipeline::fisheye_export::ExportProgress {
                        frame_idx: done.min(100), total_frames: 100,
                        fps_avg: done as f32 / t0.elapsed().as_secs_f32().max(1e-3),
                    });
                })
            } else {
                vr180_pipeline::fisheye_export::export_fisheye(
                    pipeline, cfg,
                    move |p| { let _ = progress_tx2.try_send(p); },
                    cancel_for_thread,
                )
            }
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
    /// promote `export_job` to None when it has. When a batch is
    /// running, completion records the item's outcome and immediately
    /// starts the next queued item.
    /// Drain updater events + drive the periodic re-check. Mirrors the
    /// reference UX: auto-prompt once per discovered version, never
    /// during an export; manual checks surface "up to date"/errors.
    /// Run a native file picker on a WORKER thread and deliver the result
    /// through `dialog_rx`. Never call a blocking rfd dialog directly from
    /// `update()` — see the `dialog_tx` field comment for the abort that
    /// causes. Re-entrant clicks are ignored while a panel is up.
    fn spawn_dialog<F>(&mut self, ctx: &egui::Context, pick: F)
    where
        F: FnOnce() -> Option<DialogResult> + Send + 'static,
    {
        if self.dialog_open {
            return;
        }
        self.dialog_open = true;
        let tx = self.dialog_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let r = pick().unwrap_or(DialogResult::Cancelled);
            let _ = tx.send(r);
            // The UI keeps painting behind the panel, but the result can
            // land between frames — wake the loop so it is applied now.
            ctx.request_repaint();
        });
    }

    /// Apply whatever the worker-thread pickers came back with. Drained
    /// once per frame from `update()`.
    fn poll_dialogs(&mut self) {
        while let Ok(r) = self.dialog_rx.try_recv() {
            self.dialog_open = false;
            match r {
                DialogResult::Videos(paths) => self.open_paths(paths),
                DialogResult::BatchOutDir(dir) => self.batch_out_dir = Some(dir),
                DialogResult::Lut(p) => {
                    self.settings.lut_path = p.to_string_lossy().to_string();
                }
                DialogResult::GyroflowProfile(path) => {
                    let s = &mut self.settings;
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
                DialogResult::Cancelled => {}
            }
        }
    }

    fn poll_updater(&mut self, ctx: &egui::Context) {
        while let Ok(ev) = self.updater_rx.try_recv() {
            use crate::updater::UpdateEvent as E;
            match ev {
                E::Checked(crate::updater::CheckOutcome::UpdateAvailable(info)) => {
                    let skipped = self.updater_prefs.skip_version.as_deref()
                        == Some(info.version.as_str());
                    let prompted = self.update_prompted_version.as_deref()
                        == Some(info.version.as_str());
                    // Auto-open once per version per session; skipped
                    // versions only open on a manual check. Never
                    // interrupt a running export.
                    if self.update_manual_check
                        || (!skipped && !prompted && self.export_job.is_none())
                    {
                        self.show_update_popover = true;
                        self.update_prompted_version = Some(info.version.clone());
                    }
                    self.update_available = Some(info);
                    self.update_status = None;
                }
                E::Checked(crate::updater::CheckOutcome::UpToDate { current_notes }) => {
                    self.update_available = None;
                    // "What's new in the version you're on" — shown in the
                    // popover's release-notes section.
                    if !current_notes.is_empty() {
                        self.current_version_notes = Some(current_notes);
                    }
                    if self.update_manual_check {
                        self.update_status =
                            Some(tr("You're on the latest version.").to_string());
                        self.show_update_popover = true;
                    }
                }
                E::CheckFailed(e) => {
                    tracing::warn!("updater: check failed: {e}");
                    if self.update_manual_check {
                        // Missing feed = informational, not an error (no
                        // release has shipped a latest.json yet).
                        self.update_status = Some(if e == "no update feed published yet" {
                            tr("No update feed published yet.").to_string()
                        } else {
                            format!("{} {e}", tr("Check failed:"))
                        });
                        self.show_update_popover = true;
                    }
                }
                E::Progress { got, total } => {
                    self.update_progress = Some((got, total));
                }
                E::Installing => {
                    self.update_progress = None;
                    // The process is about to be replaced; nothing to do —
                    // the popover shows "restarting…" until then.
                }
                E::ReadyToInstall(staged) => {
                    // Windows: close gracefully; `on_exit` spawns the
                    // installer once the event loop has unwound (settings
                    // saved, GPU/decode/audio threads torn down). A
                    // process::exit from the worker thread wedged on DLL
                    // teardown — the app stayed open behind the installer.
                    tracing::info!(
                        "updater: v{} staged — closing to install", staged.version);
                    self.pending_update_install = Some(staged);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                E::Failed(e) => {
                    tracing::error!("updater: {e}");
                    self.update_installing = false;
                    self.update_progress = None;
                    self.update_status = Some(format!("{} {e}", tr("Update failed:")));
                    self.show_update_popover = true;
                }
            }
        }
        // Periodic re-check every 4 h (reference cadence), skipped while
        // anything update-related is busy or shown.
        const RECHECK: std::time::Duration = std::time::Duration::from_secs(4 * 60 * 60);
        if self.update_last_check.elapsed() >= RECHECK
            && !self.update_installing
            && !self.show_update_popover
        {
            self.update_last_check = std::time::Instant::now();
            self.update_manual_check = false;
            crate::updater::spawn_check(self.updater_tx.clone());
        }
    }

    /// The update popover — bottom-anchored non-modal window (the app's
    /// accepted popover idiom; no new floating tool windows).
    fn draw_update_popover(&mut self, ctx: &egui::Context) {
        if !self.show_update_popover {
            return;
        }
        let mut open = self.show_update_popover;
        egui::Window::new(tr("Software update"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -150.0])
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.set_max_width(420.0);
                ui.label(format!("{} v{}", tr("Installed:"), env!("CARGO_PKG_VERSION")));
                if let Some(info) = self.update_available.clone() {
                    ui.label(
                        RichText::new(format!("{} v{}", tr("New version:"), info.version))
                            .strong(),
                    );
                    if !info.notes.is_empty() {
                        ui.add_space(4.0);
                        egui::ScrollArea::vertical()
                            .max_height(160.0)
                            .show(ui, |ui| {
                                ui.label(RichText::new(info.notes.clone()).small());
                            });
                    }
                    ui.add_space(6.0);
                    if self.update_installing {
                        match self.update_progress {
                            Some((got, total)) if total > 0 => {
                                let frac = got as f32 / total as f32;
                                ui.add(egui::ProgressBar::new(frac).desired_width(380.0).text(
                                    format!(
                                        "{} {:.1} / {:.1} MB",
                                        tr("Downloading…"),
                                        got as f64 / 1e6,
                                        total as f64 / 1e6
                                    ),
                                ));
                            }
                            Some((got, _)) => {
                                ui.label(format!(
                                    "{} {:.1} MB",
                                    tr("Downloading…"),
                                    got as f64 / 1e6
                                ));
                            }
                            None => {
                                ui.label(tr("Installing — the app will restart…"));
                            }
                        }
                    } else {
                        ui.horizontal(|ui| {
                            if ui
                                .button(RichText::new(tr("Install & Restart")).strong())
                                .clicked()
                            {
                                self.update_installing = true;
                                self.update_status = None;
                                self.update_progress = Some((0, 0));
                                crate::updater::spawn_install(
                                    info.clone(),
                                    self.updater_tx.clone(),
                                );
                            }
                            if ui.button(tr("Skip this version")).clicked() {
                                self.updater_prefs.skip_version = Some(info.version.clone());
                                crate::updater::save_prefs(&self.updater_prefs);
                                self.show_update_popover = false;
                            }
                        });
                    }
                }
                // Up to date: show what changed in the version we're on
                // (the feed's notes for the running release).
                if self.update_available.is_none() {
                    if let Some(notes) = self.current_version_notes.clone() {
                        ui.add_space(6.0);
                        ui.label(RichText::new(format!(
                            "{} v{}", tr("What's new in"), env!("CARGO_PKG_VERSION")
                        )).strong());
                        ui.add_space(2.0);
                        egui::ScrollArea::vertical()
                            .max_height(160.0)
                            .show(ui, |ui| {
                                ui.label(RichText::new(notes).small());
                            });
                    }
                }
                if let Some(status) = &self.update_status {
                    ui.add_space(4.0);
                    ui.label(RichText::new(status.clone()).small().color(Color32::GRAY));
                }
                if !self.update_installing {
                    ui.add_space(4.0);
                    if ui.small_button(tr("Check for updates")).clicked() {
                        self.update_manual_check = true;
                        self.update_status = None;
                        self.update_last_check = std::time::Instant::now();
                        crate::updater::spawn_check(self.updater_tx.clone());
                    }
                }
            });
        // Don't let the X close the window mid-install (the process is
        // about to restart anyway; closing would just hide the state).
        if !self.update_installing {
            self.show_update_popover = open;
        }
    }

    fn poll_export_job(&mut self) {
        let Some(job) = self.export_job.as_mut() else { return; };
        // Drain the channel.
        while let Ok(p) = job.progress_rx.try_recv() {
            job.last_progress = Some(p);
        }
        let has_frames = job.last_progress.as_ref().map_or(false, |p| p.frame_idx > 0);
        // The run's first written frame anchors the ETA rate: everything
        // before it (IMU load, decoder/encoder init, ProRes Vulkan warm-up)
        // is start-up, not encode throughput.
        if has_frames && self.run_rate_anchor.is_none() {
            let done = self.run_frame_tally().map(|t| t.0).unwrap_or(0);
            self.run_rate_anchor = Some((std::time::Instant::now(), done));
        }
        let Some(job) = self.export_job.as_mut() else { return; };
        // Check completion. JoinHandle::is_finished is stable in 1.61+.
        if job.handle.is_finished() {
            let job = self.export_job.take().unwrap();
            let result = job.handle.join();
            match &result {
                Ok(Ok(())) => tracing::info!(
                    "export: done → {} (took {:.2?})",
                    job.output_path.display(), job.started_at.elapsed()
                ),
                Ok(Err(e)) => tracing::error!("export: failed: {e}"),
                Err(_) => tracing::error!("export: worker panicked"),
            }
            // Batch bookkeeping: record this item's outcome, then chain
            // the next queued item. A failed item never stops the batch
            // (that's the unattended-run contract) — it records the error
            // and moves on.
            if let Some(idx) = self.batch_current.take() {
                if let Some(item) = self.batch.get_mut(idx) {
                    item.status = if self.batch_skip_flag {
                        BatchStatus::Skipped
                    } else {
                        match result {
                            Ok(Ok(())) => BatchStatus::Done,
                            Ok(Err(e)) => BatchStatus::Failed(e.to_string()),
                            Err(_) => BatchStatus::Failed("worker panicked".into()),
                        }
                    };
                }
                self.batch_skip_flag = false;
                if self.batch_running {
                    self.start_next_batch_item();
                }
            }
        }
    }

    // ─── Source clip list + batch export ──────────────────────────

    /// Save the live sidebar settings back into the ACTIVE clip's list
    /// entry. Call before switching clips / starting a batch so edits
    /// aren't lost (the sidebar edits `self.settings`, not the entry).
    /// Persisted-settings bucket for a source kind. OSV / EAC / SBS / BRAW
    /// each keep an independent setup; anything else shares `"other"`.
    fn settings_class_key(kind: vr180_pipeline::SourceKind) -> &'static str {
        use vr180_pipeline::SourceKind as K;
        match kind {
            K::DjiOsv         => "osv",
            K::Insta360Insv   => "insv",
            K::GoProEac       => "eac",
            K::SbsFisheye     => "sbs",
            K::BlackmagicRaw  => "braw",
            K::Insta360Dng    => "dng",
            K::KandaoQoocam   => "kandao",
            K::KandaoDng      => "kandao_dng",
            _                 => "other",
        }
    }

    /// Remembered settings template for `kind` (defaults if none saved yet).
    /// A kind with a video/still sibling (X6, QooCam) takes the camera's
    /// shared stereo offset (see `shared_stereo`) over whatever its own
    /// template remembers.
    fn kind_template(&self, kind: vr180_pipeline::SourceKind) -> Settings {
        let mut s = self.kind_settings.get(Self::settings_class_key(kind))
            .cloned().unwrap_or_default();
        self.apply_shared_stereo(kind, &mut s);
        s
    }

    /// The two template keys of a camera that writes video and stills from
    /// the same two lenses (the X6's `.insv` + `.dng`, the QooCam's `.mp4` +
    /// `.dng`): its stereo offset (the rig's own eye alignment) is ONE value
    /// shared by both kinds — whichever was edited last. Video key first.
    fn stereo_shared_keys(kind: vr180_pipeline::SourceKind) -> Option<[&'static str; 2]> {
        let sib = kind.stereo_sibling()?;
        let (video, still) = if kind.is_still() { (sib, kind) } else { (kind, sib) };
        Some([Self::settings_class_key(video), Self::settings_class_key(still)])
    }

    /// The camera's shared stereo offset `(yaw, pitch, roll)`: both templates
    /// carry it (kept equal by `save_active_kind_template`), so read either —
    /// the video's first, for a user who has only ever tuned video.
    fn shared_stereo(&self, kind: vr180_pipeline::SourceKind) -> Option<(f32, f32, f32)> {
        Self::stereo_shared_keys(kind)?.iter().find_map(|k| self.kind_settings.get(*k)
            .map(|s| (s.stereo_yaw_deg, s.stereo_pitch_deg, s.stereo_roll_deg)))
    }

    fn apply_shared_stereo(&self, kind: vr180_pipeline::SourceKind, s: &mut Settings) {
        if let Some((yaw, pitch, roll)) = self.shared_stereo(kind) {
            s.stereo_yaw_deg = yaw;
            s.stereo_pitch_deg = pitch;
            s.stereo_roll_deg = roll;
        }
    }

    /// Fold the active clip's live settings into ITS kind's template, so the
    /// next same-kind clip inherits them (kept independent across kinds —
    /// except a camera's shared stereo offset, written to both its templates).
    fn save_active_kind_template(&mut self) {
        if let Some(kind) = self.clip.as_ref().map(|c| c.source_kind) {
            self.kind_settings.insert(
                Self::settings_class_key(kind).to_string(), self.settings.clone());
            if let Some(keys) = Self::stereo_shared_keys(kind) {
                let (yaw, pitch, roll) = (self.settings.stereo_yaw_deg,
                    self.settings.stereo_pitch_deg, self.settings.stereo_roll_deg);
                for key in keys {
                    let t = self.kind_settings.entry(key.to_string()).or_default();
                    t.stereo_yaw_deg = yaw;
                    t.stereo_pitch_deg = pitch;
                    t.stereo_roll_deg = roll;
                }
            }
        }
    }

    fn sync_active_clip_settings(&mut self) {
        if let Some(idx) = self.active_clip {
            if let Some(item) = self.batch.get_mut(idx) {
                if self.loaded_path.as_ref() == Some(&item.path) {
                    item.settings = self.settings.clone();
                }
            }
        }
        // Also update the active clip's KIND template (so a later same-kind
        // clip inherits these edits, and a different-kind clip does not).
        self.save_active_kind_template();
    }

    /// Make clip `idx` the active one: preview it and point the sidebar
    /// at ITS settings. The outgoing clip's edits are saved first.
    fn select_clip(&mut self, idx: usize) {
        if self.active_clip == Some(idx) { return; }
        let Some(path) = self.batch.get(idx).map(|b| b.path.clone()) else { return; };
        self.sync_active_clip_settings();
        self.settings = self.batch[idx].settings.clone();
        // An X6 clip (video or still) takes the camera's shared stereo
        // offset — the latest edit on either kind — over its own snapshot.
        let kind = self.batch[idx].source_kind;
        let mut restored = std::mem::take(&mut self.settings);
        self.apply_shared_stereo(kind, &mut restored);
        self.settings = restored;
        tracing::info!("RESEED-DBG select_clip {idx}: restored cx_l={:.4} cy_l={:.4} fov_l={:.2} ov_l={}",
            self.settings.fisheye_cx_norm_left, self.settings.fisheye_cy_norm_left,
            self.settings.fisheye_fov_deg_left, self.settings.fisheye_override_left);
        self.active_clip = Some(idx);
        // The restored settings may carry hand-tuned Override values for
        // THIS clip — don't let the auto-reseed overwrite them when the
        // clip's in-file calib arrives.
        self.suppress_reseed_once = true;
        // The entry's settings already hold the right per-clip trim + IMU
        // phase (normalized at add, or saved from a previous edit session)
        // — preserve them through the load.
        self.load_file_inner(path, true);
    }

    /// Probe + add files to the clip list. Each entry starts from a copy
    /// of the CURRENT settings (matching the app's "a new clip keeps your
    /// last setup" behavior), with the per-clip fields normalized: IMU
    /// phase re-seeds to SROT/2 of the file's own fps and trim clears —
    /// unless it IS the currently loaded clip (its live values apply).
    /// If nothing is loaded yet, the first added clip becomes active.
    fn add_batch_files(&mut self, paths: Vec<PathBuf>) {
        let mut first_new: Option<usize> = None;
        for path in paths {
            if self.batch.iter().any(|b| b.path == path) {
                continue; // already in the list
            }
            let source_kind = vr180_pipeline::source_kind::detect(&path)
                .unwrap_or(vr180_pipeline::SourceKind::Unknown);
            if !source_kind.is_exportable() {
                tracing::warn!("batch: skipping non-exportable source {}", path.display());
                continue;
            }
            let probe = if source_kind.is_still() {
                // One Bayer frame, both eyes side by side: no video probe.
                match vr180_fisheye::dng::probe(&path) {
                    Ok((w, h)) => vr180_pipeline::decode::VideoProbe { width: w, height: h, fps: 1.0, duration_sec: 0.0, hlg: false },
                    Err(e) => { tracing::warn!("batch: DNG probe failed for {}: {e}", path.display()); continue; }
                }
            } else { match vr180_pipeline::decode::probe_video(&path) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("batch: probe failed for {}: {e}", path.display());
                    continue;
                }
            } };
            let (fisheye_eye_w, fisheye_eye_h) = match source_kind {
                vr180_pipeline::SourceKind::SbsFisheye
                | vr180_pipeline::SourceKind::Insta360Dng
                | vr180_pipeline::SourceKind::KandaoDng => (probe.width / 2, probe.height),
                vr180_pipeline::SourceKind::KandaoQoocam => (probe.width, probe.height / 2),
                vr180_pipeline::SourceKind::DjiOsv
                | vr180_pipeline::SourceKind::Insta360Insv => (probe.width, probe.height),
                // The probe is the helper's own (`probe_video` routes `.braw`
                // to it): the frame holds both eyes side by side, unless the
                // first frame shows one centred circle (`braw_layout`).
                vr180_pipeline::SourceKind::BlackmagicRaw => braw_eye_dims(&path, probe.width, probe.height),
                _ => (0, 0),
            };
            let is_loaded_clip = self.loaded_path.as_deref() == Some(path.as_path());
            // A NEW clip inherits ITS kind's template (OSV / .360 / … kept
            // independent); the already-loaded clip keeps its live edits.
            let mut settings = if is_loaded_clip {
                self.settings.clone()
            } else {
                self.kind_template(source_kind)
            };
            if !is_loaded_clip {
                // RS mode + readout are per-clip — auto-detect firmware vs
                // no-firmware RS from the GoPro CORI stream (the toggle still
                // overrides). Non-EAC sources fall back to the default.
                settings.rs_mode = crate::decoder::detect_rs_mode_for_path(&path, source_kind);
                settings.rs_readout_ms = crate::decoder::detect_rs_readout_ms_for_path(&path, source_kind);
                // OSMO 360 II ("Osmo OQ002"): streams arrive eye-swapped
                // relative to the I after the VR180 mod — seed the swap
                // (the toggle still overrides per clip).
                if matches!(source_kind, vr180_pipeline::SourceKind::DjiOsv)
                    && crate::decoder::osv_is_osmo_ii(&path)
                {
                    settings.fisheye_swap_eyes = true;
                }
                // Blackmagic with the dual-fisheye preset: the lens crosses
                // over (the LEFT eye is the RIGHT half) — seed the swap from
                // the preset (the toggle still overrides per clip).
                if matches!(source_kind, vr180_pipeline::SourceKind::BlackmagicRaw) {
                    let name = if settings.fisheye_preset.is_empty() || settings.fisheye_preset == "Auto" {
                        vr180_pipeline::braw_imu::BRAW_PRESET_NAME
                    } else { settings.fisheye_preset.as_str() };
                    if let Some(p) = vr180_fisheye::presets::find(name) {
                        settings.fisheye_swap_eyes = p.left_eye_is_right_half;
                    }
                }
                settings.trim_in_s = None;
                settings.trim_out_s = None;
            }
            self.batch.push(BatchItem {
                path,
                fps: probe.fps,
                duration_sec: probe.duration_sec,
                source_kind,
                fisheye_eye_w, fisheye_eye_h,
                width: probe.width, height: probe.height,
                settings,
                status: BatchStatus::Idle,
                selected: false,
            });
            if is_loaded_clip {
                // The loaded clip just joined the list — link it up.
                self.active_clip = Some(self.batch.len() - 1);
            }
            first_new.get_or_insert(self.batch.len() - 1);
        }
        // Nothing on screen yet → activate the first added clip so the
        // preview + sidebar have something to work on.
        if self.loaded_path.is_none() {
            if let Some(idx) = first_new {
                self.select_clip(idx);
            }
        }
    }

    /// Overwrite an item's settings snapshot with the CURRENT settings,
    /// preserving its per-clip fields (trim + RS mode) — "apply my
    /// grade/stab/output choices", not "copy clip A's trim everywhere".
    fn apply_current_settings_to_item(&mut self, idx: usize) {
        let Some(item) = self.batch.get_mut(idx) else { return; };
        if item.status == BatchStatus::Running { return; }
        let keep_trim = (item.settings.trim_in_s, item.settings.trim_out_s);
        let keep_rs = (item.settings.rs_mode, item.settings.rs_readout_ms);
        item.settings = self.settings.clone();
        item.settings.trim_in_s = keep_trim.0;
        item.settings.trim_out_s = keep_trim.1;
        // RS mode + readout are per-clip — keep this item's own, don't copy mine.
        item.settings.rs_mode = keep_rs.0;
        item.settings.rs_readout_ms = keep_rs.1;
        // Re-arm a previously finished item so the change exports on the
        // next Start.
        if !matches!(item.status, BatchStatus::Queued) {
            item.status = BatchStatus::Idle;
        }
    }

    /// Output path for a batch item: `<stem>_SBS.<ext>` in the override
    /// folder (or next to the source), auto-renamed ` (2)`, ` (3)`… on
    /// collision rather than overwriting or silently skipping.
    fn batch_output_for(&self, item: &BatchItem, opts: &ExportOptions) -> PathBuf {
        let mut out = if item.source_kind.is_still() {
            Self::default_still_output_for(&item.path, opts.still_format)
        } else {
            Self::default_output_for(&item.path, opts.codec)
        };
        if let Some(dir) = &self.batch_out_dir {
            if let Some(name) = out.file_name() {
                out = dir.join(name);
            }
        }
        if out.exists() {
            let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("output").to_string();
            let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("mp4").to_string();
            let dir = out.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            for n in 2.. {
                let cand = dir.join(format!("{stem} ({n}).{ext}"));
                if !cand.exists() { out = cand; break; }
            }
        }
        out
    }

    /// Export every not-yet-done clip in the list (Idle / Failed / Skipped).
    /// Done clips stay done — re-arm one with its Re-run button.
    fn start_batch(&mut self) {
        if self.batch_running || self.export_job.is_some() { return; }
        self.sync_active_clip_settings();
        let mut run = Vec::new();
        for (i, item) in self.batch.iter_mut().enumerate() {
            if matches!(item.status,
                BatchStatus::Idle | BatchStatus::Failed(_) | BatchStatus::Skipped)
            {
                item.status = BatchStatus::Queued;
                run.push(i);
            }
        }
        self.begin_run(run);
    }

    fn start_next_batch_item(&mut self) {
        let Some(idx) = self.batch.iter().position(|b| b.status == BatchStatus::Queued) else {
            // Queue drained — run complete. Record the summary (banner +
            // notification) and reset run state.
            let done = self.batch_run_indices.iter()
                .filter(|&&i| self.batch.get(i).map_or(false, |b| b.status == BatchStatus::Done))
                .count();
            let failed = self.batch_run_indices.iter()
                .filter(|&&i| self.batch.get(i).map_or(false,
                    |b| matches!(b.status, BatchStatus::Failed(_))))
                .count();
            let elapsed = self.batch_started_at.map(|t| t.elapsed())
                .unwrap_or_default();
            tracing::info!("export run: complete — {done} done, {failed} failed in {:.1?}", elapsed);
            let n_run = self.batch_run_indices.len();
            self.batch_running = false;
            self.batch_opts = None;
            self.batch_started_at = None;
            self.batch_run_indices.clear();
            if done + failed > 0 {
                self.notify_run_complete(done, failed, elapsed, n_run);
                self.last_batch_summary = Some(BatchSummary { done, failed, elapsed });
            }
            return;
        };
        let opts = self.batch_opts.unwrap_or(self.export_opts);
        let output = self.batch_output_for(&self.batch[idx], &opts);
        let item_path = self.batch[idx].path.clone();
        let segments = self.clip_segments(&item_path);
        let (cfg, eac_stab) = {
            let item = &self.batch[idx];
            // Total across the merged recording (== single-file count when
            // `segments` is one file) so EAC stab spans the whole timeline.
            let n_frames = crate::decoder::segments_total_frames(&segments).max(1);
            let eac_stab = if item.source_kind.is_eac() {
                Some((item.settings.clone(), n_frames))
            } else { None };
            (
                Self::build_export_cfg(
                    &item.path, &output,
                    item.fps, item.source_kind,
                    item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height,
                    &item.settings, &opts,
                    segments,
                ),
                eac_stab,
            )
        };
        self.batch[idx].status = BatchStatus::Running;
        self.batch_current = Some(idx);
        tracing::info!("batch: item {}/{} → {}", idx + 1, self.batch.len(), output.display());
        self.spawn_export_job(cfg, eac_stab);
    }

    /// Cancel the running item but keep the batch going (poll marks it
    /// Skipped and chains the next).
    fn skip_current_batch_item(&mut self) {
        if let (Some(_), Some(job)) = (self.batch_current, self.export_job.as_ref()) {
            self.batch_skip_flag = true;
            job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Stop the whole batch: cancel the running item and don't chain.
    fn stop_batch(&mut self) {
        self.batch_running = false;
        self.batch_opts = None;
        if let (Some(_), Some(job)) = (self.batch_current, self.export_job.as_ref()) {
            self.batch_skip_flag = true; // mark the cut-short item Skipped
            job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
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
        // EAC (.360): the native-res zoom still is produced by the decoder
        // thread (it already holds the native cross) and ferried in via
        // `detail_frame_rx` — not the fisheye main-thread `DetailCache`.
        if self.clip.as_ref().map(|c| c.source_kind.is_eac()).unwrap_or(false) {
            self.poll_full_res_eac(ctx);
            return;
        }
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
        // Defer the full-res native decoder until the user actually zooms in.
        // Opening it eagerly on load adds a second 3840² decode of the same
        // file, contending with the preview worker (a slow first frame off an
        // external drive). The cache still persists on zoom-out (keep_alive),
        // so only the FIRST zoom-in pays the open.
        if self.detail_cache.is_none() && self.wants_full_res_still() {
            let segments = self.clip.as_ref()
                .map(|c| c.segments.clone())
                .unwrap_or_default();
            self.detail_cache = Some(crate::decoder::DetailCache::new(
                path, segments, kind, fps, self.settings.effective_swap_eyes(), ctx.clone()));
            self.full_res_key = 0;
            self.full_res_desired_key = 0;
            self.detail_last_key = 0;
        }

        // Only render a still while zoomed in. When zoomed out we keep the
        // cache + the last rendered still (the 1× view uses the live
        // preview), so returning here makes zoom-in instant when nothing
        // changed — and a cheap re-projection (no decode) if settings moved.
        if !self.wants_full_res_still() {
            return;
        }

        // Key identifies the (frame, settings) we want a still for. Zoom is
        // excluded — one still covers the whole frame; the UV sub-rect
        // samples it.
        let ts = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
        let gen = self.control.as_ref()
            .map(|c| c.settings_generation.load(Ordering::SeqCst)).unwrap_or(0);
        let ts_ms = (ts * 1000.0).round() as i64 as u64;
        // A pending snapshot renders at the export eye size — fold it into
        // the key so the still re-renders at that size, and back at native
        // once the snapshot is written.
        let snap_out = self.snapshot_pending.as_ref().map(|r| r.out);
        let snap_salt = snap_out.map(|(w, h)| ((w as u64) << 32) | h as u64).unwrap_or(0);
        let key = (ts_ms.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(gen) ^ snap_salt).max(1);

        self.full_res_desired_key = key;

        // Settle timer: reset whenever the wanted (frame+settings) changes.
        // The settle ONLY guards a frame change (scrub/seek) — that needs a
        // background re-decode, and the live preview is the right thing to
        // show meanwhile. A settings adjustment on a PAUSED frame is handled
        // separately below (no settle, no low-res drop).
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(150);
        if key != self.detail_last_key {
            self.detail_last_key = key;
            self.detail_key_changed_at = std::time::Instant::now();
        }

        // Already showing this exact key → done.
        if key == self.full_res_key {
            return;
        }

        // Is the wanted still for the SAME frame we're already showing at full
        // res? Then only the SETTINGS changed — an adjustment on a paused
        // frame. The DetailCache already has this frame's native pair decoded,
        // so re-projecting it is cheap (GPU project+compose on this thread, no
        // decode). Render it IN PLACE immediately instead of dropping to the
        // low-res live preview — so dragging a calib/view slider while zoomed
        // updates the full-res still live, with no low-res↔full-res swap.
        let same_frame = self.full_res_display.as_ref()
            .map(|d| (d.timestamp_s * 1000.0).round() as i64 as u64 == ts_ms)
            .unwrap_or(false);
        // Frame changed and not settled yet → leave the live preview up while
        // the background decode runs; wake when it settles.
        if !same_frame && self.detail_key_changed_at.elapsed() < SETTLE {
            ctx.request_repaint_after(SETTLE);
            return;
        }

        // Throttle the same-frame (slider-drag) re-render. It runs
        // synchronously on the UI thread at NATIVE res (project + RS quats +
        // compose), so doing it for every pointer event made the sliders feel
        // laggy — each drag tick paid a full-res render. ~8 Hz keeps the
        // still visibly tracking the drag while the slider stays fluid; the
        // trailing poll (key still != full_res_key once the interval elapses)
        // renders the FINAL value, so nothing is ever lost.
        const DRAG_RENDER_INTERVAL: std::time::Duration =
            std::time::Duration::from_millis(120);
        if same_frame && self.full_res_key != 0 {
            let since = self.full_res_rendered_at.elapsed();
            if since < DRAG_RENDER_INTERVAL {
                ctx.request_repaint_after(DRAG_RENDER_INTERVAL - since);
                return;
            }
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
            .render(&pipeline, ts, &settings, 0, snap_out);
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
            // Plain Unorm pass-through view (`preview_view_format`): egui-wgpu
            // 0.34 wants gamma/raw-byte samples, so NO sRGB decode here (an
            // sRGB view darkens the preview to sRGB⁻¹(v) vs the export).
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.preview_view_format),
                // wgpu 29 derives a view's usages from the texture (which
                // has STORAGE_BINDING from the compose pass) and rejects an
                // sRGB storage view. This view is only sampled for display,
                // so restrict it to TEXTURE_BINDING.
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
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
            self.full_res_rendered_at = std::time::Instant::now();
        }
    }

    /// EAC (.360) zoom magnifier. While paused + zoomed we ask the decoder
    /// thread for a native-resolution still of the current frame (it already
    /// has the assembled native cross, so this is a re-projection, not a
    /// re-decode) and register the freshest one into `full_res_display`. The
    /// central panel's `use_full_res` check then prefers it over the capped
    /// live preview whenever its timestamp matches the shown frame.
    ///
    /// Mirrors the fisheye `DetailCache` UX (debounce/throttle live on the
    /// decoder side) but reuses the live render path as the single source of
    /// truth, so the still is pixel-identical to the preview at native res.
    fn poll_full_res_eac(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        let want = self.decoder_alive && !self.playing && self.wants_full_res_still();
        if let Some(c) = &self.control {
            c.want_detail.store(want, Ordering::SeqCst);
        }
        if !want {
            // Keep the still across a zoom-OUT while still paused (zoom-in
            // reuses it instantly via the timestamp match). Only drop it when
            // playback resumes or the decoder is gone — then it's stale.
            if (self.playing || !self.decoder_alive) && self.full_res_display.is_some() {
                if let Some(prev) = self.full_res_display.take() {
                    self.egui_renderer.write().free_texture(&prev.egui_id);
                }
            }
            return;
        }
        // Pick up the freshest native-res still the decoder shipped (drop
        // any older ones queued behind it).
        if let Some(rx) = &self.detail_frame_rx {
            let mut newest: Option<DecodedFrame> = None;
            while let Ok(f) = rx.try_recv() { newest = Some(f); }
            if let Some(f) = newest {
                let mut rend = self.egui_renderer.write();
                if let Some(prev) = self.full_res_display.take() {
                    rend.free_texture(&prev.egui_id);
                }
                // Plain Unorm pass-through view (`preview_view_format`) — same
                // as the live preview, so no sRGB double-decode vs compose.
                let view = f.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.preview_view_format),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
                let id = rend.register_native_texture(
                    &self.pipeline.device, &view, wgpu::FilterMode::Linear);
                drop(rend);
                self.full_res_display = Some(DisplayFrame {
                    texture: f.texture, egui_id: id,
                    width: f.width, height: f.height,
                    frame_idx: f.frame_idx, timestamp_s: f.timestamp_s,
                });
            }
        }
        // Keep a slow poll alive while zoomed + paused so the TRAILING still
        // (the crisp one the decoder emits ~one throttle window after a slider
        // settles, when egui would otherwise have stopped repainting) is always
        // picked up. Cheap — this is a paused, hold-still inspection mode.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
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
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(f) => {
                    self.fps_stats.frames_in_window += 1;
                    newest = Some(f);
                    drained += 1;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                // The worker is gone. `Empty` and `Disconnected` used to look
                // identical here, so a dead decoder was indistinguishable from
                // a momentarily idle one.
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected && self.decoder_alive {
            tracing::warn!("decoder frame channel disconnected — worker is gone");
        }
        if drained > 0 && self.fps_stats.last_window_start.is_none() {
            tracing::info!("ui drain: first frame received (drained {} on first poll)", drained);
        }
        if let Some(f) = newest {
            self.replace_display(f);
            self.decoder_starting = false; // first frame in → load done, ungate respawns
            self.decoder_retried = false;  // a later start failure gets its one retry again
            // No extra repaint: this drain runs at the top of `update`, so
            // the frame is painted by this very pass. Requesting another
            // doubled the repaint rate to the display's, which parked the UI
            // thread in `nextDrawable` with wgpu's fence lock held and
            // stalled every decoder submit by a refresh (see `wake_ui`).
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
        // Plain Unorm pass-through view (`preview_view_format`): egui-wgpu 0.34
        // samples display textures as gamma/raw bytes, so no sRGB decode here
        // (an sRGB view darkens the preview to sRGB⁻¹(v) vs the export).
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.preview_view_format),
            // wgpu 29: restrict to TEXTURE_BINDING so the sRGB view isn't
            // treated as a storage view (the SBS texture has STORAGE usage).
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
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

    fn pick_and_load_file(&mut self, ctx: &egui::Context) {
        self.spawn_dialog(ctx, || {
            Self::video_file_dialog().pick_files().map(DialogResult::Videos)
        });
    }

    /// Open one or more files: fisheye sources join the clip list (and
    /// the first picked becomes active); legacy non-fisheye sources
    /// (GoPro `.360` EAC) bypass the list and load directly, exactly as
    /// before — they aren't batchable.
    fn open_paths(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() { return; }
        let first = paths[0].clone();
        let groups = self.group_imports(&paths);
        // If anything looks like a multi-file recording, ask the user
        // whether to merge it; otherwise import every file straight away.
        if groups.iter().any(|g| g.mergeable()) {
            self.pending_import = Some(PendingImport { groups, first });
        } else {
            self.apply_import(groups, false, first);
        }
    }

    /// Group the just-picked files into recordings. GoPro `.360` files
    /// expand to their auto-detected chapters; a multi-selection of OSV
    /// files becomes ONE merge candidate (their segments can't be told
    /// from separate clips by name, so the user picks them explicitly).
    fn group_imports(&self, paths: &[PathBuf]) -> Vec<ImportGroup> {
        use vr180_pipeline::SourceKind;
        let kind_of = |p: &std::path::Path|
            vr180_pipeline::source_kind::detect(p).unwrap_or(SourceKind::Unknown);
        let mut groups: Vec<ImportGroup> = Vec::new();
        let mut claimed: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

        // 1) The OSV multi-selection → one merge candidate, ordered by the DJI
        //    file index so the merge concatenates in CAPTURE sequence. A lone
        //    OSV falls through to step 2 as a singleton.
        let mut osv: Vec<PathBuf> = paths.iter()
            .filter(|p| matches!(kind_of(p), SourceKind::DjiOsv))
            .cloned().collect();
        // `CAM_<timestamp>_<index>_D.OSV` → sort by the numeric <index>
        // (…_0040_… < …_0041_… < …_0042_…). The index is the reliable monotonic
        // counter; a plain name/path sort keys on the timestamp first and can
        // disagree if it differs between segments. Tie-break by name for any
        // non-conforming filename so the order stays deterministic.
        osv.sort_by_cached_key(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let idx = name.split('_').nth(2).and_then(|s| s.parse::<u64>().ok()).unwrap_or(u64::MAX);
            (idx, name)
        });
        if osv.len() > 1 {
            for p in &osv { claimed.insert(p.clone()); }
            groups.push(ImportGroup {
                label: format!("OSV · {} files merged", osv.len()),
                picked: osv.clone(),
                members: osv,
            });
        }

        // 2) Remaining picked files, in pick order. GoPro `.360` expands to
        //    its chapters; everything else is a singleton.
        for p in paths {
            if claimed.contains(p) { continue; }
            let kind = kind_of(p);
            if !kind.is_exportable() {
                tracing::warn!("import: skipping non-exportable {}", p.display());
                continue;
            }
            let members = vr180_core::segments::detect_segments(p);
            if members.iter().any(|m| claimed.contains(m)) { continue; } // sibling already grouped
            for m in &members { claimed.insert(m.clone()); }
            let label = if members.len() > 1 {
                format!("GoPro · {} chapters", members.len())
            } else {
                p.file_name().and_then(|s| s.to_str()).unwrap_or("clip").to_string()
            };
            groups.push(ImportGroup { picked: vec![p.clone()], members, label });
        }
        groups
    }

    /// Carry out a (possibly resolved) import. `merge` merges every
    /// multi-file group into one clip; otherwise each PICKED file becomes
    /// its own clip. Records the per-clip segment choice in `merge_groups`
    /// so decode + export both honor it, then activates the first pick.
    fn apply_import(&mut self, groups: Vec<ImportGroup>, merge: bool, first: PathBuf) {
        let mut to_add: Vec<PathBuf> = Vec::new();
        let mut active_rep: Option<PathBuf> = None;
        for g in groups {
            if merge && g.mergeable() {
                let rep = g.members[0].clone();
                self.merge_groups.insert(rep.clone(), g.members.clone());
                if g.members.contains(&first) { active_rep = Some(rep.clone()); }
                to_add.push(rep);
            } else {
                // Individual: pin each picked file's segments to itself so a
                // GoPro chapter doesn't silently re-merge via detect_segments.
                for p in &g.picked {
                    self.merge_groups.insert(p.clone(), vec![p.clone()]);
                    if *p == first { active_rep = Some(p.clone()); }
                    to_add.push(p.clone());
                }
            }
        }
        self.add_batch_files(to_add);
        let active = active_rep.unwrap_or(first);
        if let Some(idx) = self.batch.iter().position(|b| b.path == active) {
            self.select_clip(idx);
        }
    }

    /// The files that make up the clip rooted at `path`: the import-time
    /// choice if one was recorded, else GoPro auto-chapter detection (for
    /// OSV this is just the file — segments aren't detectable by name).
    fn clip_segments(&self, path: &std::path::Path) -> Vec<PathBuf> {
        self.merge_groups.get(path).cloned()
            .unwrap_or_else(|| vr180_core::segments::detect_segments(path))
    }

    /// Modal asking whether to merge a multi-file recording. Shown while
    /// `pending_import` is `Some`; resolves to `apply_import(merge)`.
    fn draw_import_prompt(&mut self, ctx: &egui::Context) {
        if self.pending_import.is_none() { return; }
        let mut decision: Option<bool> = None; // Some(true)=merge, Some(false)=individual
        let mut cancel = false;
        {
            let pending = self.pending_import.as_ref().unwrap();
            let n_files: usize = pending.groups.iter()
                .filter(|g| g.mergeable()).map(|g| g.members.len()).sum();
            egui::Window::new(tr("Import — merge segments?"))
                .resizable(false)
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .default_width(440.0)
                .show(ctx, |ui| {
                    ui.label(RichText::new(format!(
                        "{n_files} {}", tr("files look like one continuous recording:"))).strong());
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                        for g in pending.groups.iter().filter(|g| g.mergeable()) {
                            ui.label(RichText::new(format!("▸ {}", g.label)).strong());
                            for m in &g.members {
                                ui.label(RichText::new(format!("     {}",
                                    m.file_name().and_then(|s| s.to_str()).unwrap_or("?")))
                                    .small().color(Color32::GRAY));
                            }
                        }
                    });
                    ui.add_space(8.0);
                    ui.label(RichText::new(tr(
                        "Merge plays + exports each recording as ONE clip with continuous \
                         stabilization across the join. Individual imports every file \
                         as its own clip.")).small().color(Color32::GRAY));
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button(RichText::new(tr("Merge into one clip")).strong()).clicked() {
                            decision = Some(true);
                        }
                        ui.add_space(6.0);
                        if ui.button(tr("Import individually")).clicked() {
                            decision = Some(false);
                        }
                        ui.add_space(6.0);
                        if ui.button(tr("Cancel")).clicked() { cancel = true; }
                    });
                });
        }
        if cancel {
            self.pending_import = None;
        } else if let Some(merge) = decision {
            let pending = self.pending_import.take().unwrap();
            self.apply_import(pending.groups, merge, pending.first);
        }
    }

    fn load_file(&mut self, path: PathBuf) {
        self.load_file_inner(path, false);
    }

    /// Return the app to the "no clip loaded" state: stop the decoder,
    /// free the preview/zoom textures, drop the caches, cancel any load
    /// still in flight, and clear `clip` + `loaded_path` so the sidebar
    /// and central panel show the empty state. `load_file_inner` runs
    /// this before every load; the clip list runs it when the loaded
    /// clip's entry is removed. Everything here is UI-thread-only state
    /// (egui textures, playback flags). `preserve_clip_settings` keeps
    /// the per-clip trim (same-clip reloads); the rest of `Settings` is
    /// remembered across loads regardless.
    fn unload_clip(&mut self, preserve_clip_settings: bool) {
        // 1. Stop the decoder thread + drop the IPC channels.
        self.stop_playback();
        // 2. Free the egui-registered preview texture from the old clip, so
        //    the previous frame doesn't stay painted.
        if let Some(prev) = self.current_display.take() {
            self.egui_renderer.write().free_texture(&prev.egui_id);
        }
        // 3. Drop the zoom-magnifier caches so they can't serve the PREVIOUS
        //    clip's still later.
        if let Some(prev) = self.full_res_display.take() {
            self.egui_renderer.write().free_texture(&prev.egui_id);
        }
        self.detail_cache = None;
        self.full_res_key = 0;
        self.full_res_desired_key = 0;
        self.detail_last_key = 0;
        // 4. Reset volatile playback state. Everything in `Settings` is
        //    REMEMBERED across loads, except the clip-specific trim range.
        self.fps_stats = FpsStats::default();
        if !preserve_clip_settings {
            self.settings.trim_in_s = None;
            self.settings.trim_out_s = None;
        }
        // 5. Drop ClipInfo + path: the sidebar shows "No file loaded." (or
        //    "Loading…" once a new load is queued behind this).
        self.clip = None;
        self.loaded_path = None;
        // The next clip's in-file calibration must reseed the Override fields.
        self.fisheye_last_seeded = None;
        // Drop any stale "play when ready" intent from a prior load.
        self.play_when_ready = false;
        // 6. Cancel a load still in flight: bump the token so a late worker
        //    result reads as superseded, and drop the pending handle (its
        //    receiver goes with it — the worker's send simply fails).
        self.load_token = self.load_token.wrapping_add(1);
        self.load_pending = None;
    }

    /// `preserve_clip_settings = true` keeps the per-clip fields (trim +
    /// RS mode) currently in `self.settings` instead of resetting them
    /// — used when re-activating a clip whose own settings were just
    /// restored from the clip list (and for same-clip reloads like the
    /// eye-orientation toggle).
    fn load_file_inner(&mut self, path: PathBuf, preserve_clip_settings: bool) {
        // ── Tear down everything tied to the previous clip ──────────
        // (Shared with the clip list's remove-loaded-clip path.) The actual
        // SOURCE I/O — probe, format detect, GoPro GPMF, segment scan,
        // per-segment probes — is NOT done here: it runs on a worker thread
        // (`load_clip_metadata`) and lands in `finish_load`. Doing it inline
        // used to freeze the whole GUI whenever the source was cold/slow —
        // most painfully a spun-down SMB/NAS share, where the first access
        // stalls for seconds while the disks wake (every later access is
        // instant once cached, which is exactly the "first time hangs,
        // second time instant" report).
        self.unload_clip(preserve_clip_settings);

        // Resolve the segment list on the UI thread when it's a cheap
        // import-time choice (a HashMap lookup); otherwise leave it to the
        // worker (`None` → on-disk chapter detection, which scans sibling
        // files and can be slow on a network share).
        let merge_entry = self.merge_groups.get(&path).cloned();

        // Bump the load id so a result from a superseded load is ignored,
        // then hand all blocking source I/O to a worker thread.
        self.load_token = self.load_token.wrapping_add(1);
        let token = self.load_token;
        let (tx, rx) = crossbeam_channel::bounded::<LoadOutcome>(1);
        let path_for_worker = path.clone();
        std::thread::spawn(move || {
            let outcome = Self::load_clip_metadata(
                token, path_for_worker, preserve_clip_settings, merge_entry);
            let _ = tx.send(outcome);
        });
        self.load_pending = Some(LoadPending {
            token, path, rx, started: std::time::Instant::now(),
        });
    }

    /// Read everything `finish_load` needs, OFF the UI thread: format
    /// detect, video probe, GoPro GPMF (CORI/GRAV/SROT/RS), per-eye dims,
    /// segment chain, and the chain's total duration/frame count. All the
    /// potentially-slow source reads live here so the GUI stays responsive
    /// even when the source is on a cold/slow share. Pure function — touches
    /// no `&self` state, so it's safe to run on a detached thread.
    fn load_clip_metadata(
        token: u64,
        path: PathBuf,
        preserve_clip_settings: bool,
        merge_entry: Option<Vec<PathBuf>>,
    ) -> LoadOutcome {
        let t_load = std::time::Instant::now();
        let source_kind = vr180_pipeline::source_kind::detect(&path)
            .unwrap_or(vr180_pipeline::SourceKind::Unknown);
        tracing::info!("load_file: {} → kind={:?}", path.display(), source_kind);

        let probe = if source_kind.is_still() {
            let (w, h) = vr180_fisheye::dng::probe(&path).map_err(|e| format!("DNG probe failed: {e}"))?;
            vr180_pipeline::decode::VideoProbe { width: w, height: h, fps: 1.0, duration_sec: 0.0, hlg: false }
        } else {
            vr180_pipeline::decode::probe_video(&path)
                .map_err(|e| format!("probe failed: {e}"))?
        };
        tracing::info!("load_timing: after probe @ {:?}", t_load.elapsed());

        // GoPro family: parse GPMF + GEOC for metadata display + RS auto-detect.
        let (cori_count, grav_count, srot_ms, detected_rs_mode) =
            if source_kind == vr180_pipeline::SourceKind::GoProEac {
                match crate::decoder::extract_gpmf_cached(&path) {
                    Ok(gpmf) => {
                        let cori = vr180_core::gyro::parse_cori(&gpmf);
                        let raw = vr180_core::gyro::parse_raw_imu(&gpmf);
                        let srot = vr180_core::geoc::find_srot_ms(&gpmf)
                            .or_else(|| {
                                vr180_core::geoc::lookup_srot_s(&path, None)
                                    .ok().flatten().map(|s| s * 1000.0)
                            });
                        // Auto-detect on CHAPTER-1 CORI: an individually
                        // loaded later chapter's own CORI starts mid-flight.
                        let c1 = crate::decoder::first_chapter_of(&path);
                        let rs_mode = if c1 != path {
                            crate::decoder::extract_gpmf_cached(&c1)
                                .map(|g| crate::decoder::detect_rs_mode(
                                    &vr180_core::gyro::parse_cori(&g)))
                                .unwrap_or_else(|_| crate::decoder::detect_rs_mode(&cori))
                        } else {
                            crate::decoder::detect_rs_mode(&cori)
                        };
                        (cori.len(), raw.grav.len(), srot, rs_mode)
                    }
                    Err(_) => (0, 0, None, crate::decoder::default_rs_mode()),
                }
            } else {
                (0, 0, None, crate::decoder::default_rs_mode())
            };

        // Fisheye family: compute one-eye dimensions for FOV slider hints.
        let (fisheye_eye_w, fisheye_eye_h) = match source_kind {
            vr180_pipeline::SourceKind::SbsFisheye
            | vr180_pipeline::SourceKind::Insta360Dng
            | vr180_pipeline::SourceKind::KandaoDng => (probe.width / 2, probe.height),
            vr180_pipeline::SourceKind::KandaoQoocam => (probe.width, probe.height / 2),
            vr180_pipeline::SourceKind::DjiOsv
            | vr180_pipeline::SourceKind::Insta360Insv => (probe.width, probe.height),
            vr180_pipeline::SourceKind::BlackmagicRaw => braw_eye_dims(&path, probe.width, probe.height),
            _ => (0, 0),
        };

        let segments = if source_kind.is_still() { vec![path.clone()] } else {
            merge_entry.unwrap_or_else(|| vr180_core::segments::detect_segments(&path))
        };
        tracing::info!("load_timing: after clip_segments ({} segs) @ {:?}",
            segments.len(), t_load.elapsed());

        // Duration + frame count span the WHOLE segment chain (GS01…, GS02…).
        // Single-segment: just this file's probe.
        let (duration_sec, frame_count) = if segments.len() > 1 {
            // Sum each segment's duration via the FAST moov probe, run
            // concurrently. ffmpeg's find_stream_info (probe_video) does scattered
            // packet reads (~10 s per cold segment on SMB) — for a 5-chapter clip
            // that blocked the load ~40 s. fps is uniform across the chain.
            // VIDEO-track duration (frame-exact) — `total_frames` below feeds
            // frame-indexed stab/RS tables, and the movie (mvhd) duration's
            // audio overhang would add a spurious frame per segment.
            let fps = probe.fps.max(1e-3) as f64;
            let durs: Vec<f64> = std::thread::scope(|sc| {
                let hs: Vec<_> = segments.iter().map(|s| sc.spawn(move || {
                    vr180_pipeline::decode::probe_video_duration_via_moov(s).ok()
                        .or_else(|| vr180_pipeline::decode::probe_duration_via_moov(s).ok())
                        .or_else(|| vr180_pipeline::decode::probe_video(s).ok().map(|p| p.duration_sec))
                        .unwrap_or(0.0)
                })).collect();
                hs.into_iter().map(|h| h.join().unwrap_or(0.0)).collect()
            });
            let total_dur: f64 = durs.iter().sum();
            let total_frames = (total_dur * fps).round() as u32;
            tracing::info!(
                "multi-segment: {} chapters → {:.1}s, {} frames total",
                segments.len(), total_dur, total_frames);
            (total_dur, total_frames)
        } else if source_kind.is_still() {
            (0.0, 1)
        } else {
            (probe.duration_sec, (probe.duration_sec * probe.fps as f64).round() as u32)
        };

        // NOTE: no DJI metadata read here — the decoder loads the lens calib
        // FAST (first sample) for an immediate preview and streams the heavy
        // per-frame quats in the background (see `load_dji_imu_progressive`), so
        // the load worker stays fast and the giant metadata read never blocks.

        // GoPro `.360`: GEOC cal dimension for the ".360 Lens calibration"
        // panel's cx/cy pixel space (cheap 1 MiB file-tail read, off the UI
        // thread here). The decoder re-parses GEOC itself for the warp.
        let eac_cal_dim = if source_kind == vr180_pipeline::SourceKind::GoProEac {
            vr180_core::geoc::parse_geoc(segments.first().unwrap_or(&path))
                .ok().flatten().map(|g| g.cal_dim)
        } else {
            None
        };

        // X6 "360° Audio": a 4-channel AmbiX AAC instead of stereo. Header
        // read only; the worker is off the UI thread.
        let foa_audio = source_kind == vr180_pipeline::SourceKind::Insta360Insv
            && vr180_pipeline::audio::probe_foa_aac(&path).ok().flatten().is_some();
        if foa_audio { tracing::info!("load: X6 360° Audio track (4-ch AmbiX) detected"); }
        // Kandao QooCam: a 4-channel AAC of its microphone capsules (not a
        // sound field) — mixed to stereo by the player and the export.
        let quad_audio = source_kind == vr180_pipeline::SourceKind::KandaoQoocam
            && vr180_pipeline::audio::probe_foa_aac(&path).ok().flatten().is_some();
        if quad_audio { tracing::info!("load: QooCam 4-channel microphone track detected"); }
        if probe.hlg { tracing::info!("load: HLG (ARIB STD-B67) transfer — HDR clip"); }
        // Blackmagic RAW: the clip's own display LUT ("Gen 5 Film to
        // Extended Video"), written out by the helper for the LUT stage.
        let braw_lut = if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw {
            vr180_braw::BrawInfo::probe(&path).ok()
                .filter(|i| !i.embedded_lut.is_empty())
                .and_then(|i| match vr180_braw::extract_embedded_lut(&path, &i.embedded_lut) {
                    Ok(p) => { tracing::info!("load: BRAW embedded LUT '{}' → {}", i.embedded_lut, p.display()); Some(p.to_string_lossy().into_owned()) }
                    Err(e) => { tracing::warn!("load: BRAW embedded LUT: {e}"); None }
                })
        } else { None };
        // …and the SDK's colour processing as shot (probe memoised: no re-read).
        let braw_color = if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw {
            vr180_braw::BrawInfo::probe(&path).ok().and_then(|i| i.color)
        } else { None };
        if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw && braw_color.is_none() {
            tracing::warn!("load: BRAW probe has no colour block — the Blackmagic RAW colour section needs a rebuilt helper");
        }
        Ok(LoadedMeta {
            token, path, preserve_clip_settings, source_kind,
            width: probe.width, height: probe.height, fps: probe.fps,
            duration_sec, frame_count, segments,
            cori_count, grav_count, srot_ms,
            detected_rs_mode, fisheye_eye_w, fisheye_eye_h,
            eac_cal_dim, foa_audio, quad_audio, hlg: probe.hlg, braw_lut, braw_color,
        })
    }

    /// Apply a finished async load on the UI thread: seed the per-clip
    /// settings, publish `ClipInfo`, and spawn the (paused) decoder. By now
    /// the worker has already read the source, so it's warm — `spawn_decoder`
    /// and its audio open don't block.
    fn finish_load(&mut self, meta: LoadedMeta) {
        let source_kind = meta.source_kind;
        // One-time stale-Override guard for `.360` (see the field docs on
        // `Settings::eac_lens_sanitized`): settings lineages from before the
        // ".360 Lens calibration" feature may carry dead Override values —
        // clear them so the lens warp stays a no-op until deliberately
        // enabled. The kind-map loader sanitizes the stored template; this
        // catches any other path by which old settings go live for an EAC
        // clip (e.g. the legacy single settings.json at startup).
        if source_kind == vr180_pipeline::SourceKind::GoProEac
            && !self.settings.eac_lens_sanitized
        {
            if self.settings.fisheye_override_left || self.settings.fisheye_override_right {
                tracing::info!("finish_load: clearing stale pre-feature .360 lens Override");
            }
            self.settings.fisheye_override_left = false;
            self.settings.fisheye_override_right = false;
            self.settings.eac_lens_sanitized = true;
        }
        // Blackmagic templates from before the dual-fisheye preset (2026-10-19)
        // hold "Custom" or the single-camera "Blackmagic Pyxis 12K" that the
        // old BRAW path auto-picked — neither a deliberate choice for this
        // lens. Migrate once to the new auto preset unless an Override is on.
        if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw
            && !self.settings.fisheye_override_left && !self.settings.fisheye_override_right
            && (self.settings.fisheye_preset == "Custom" || self.settings.fisheye_preset == "Blackmagic Pyxis 12K")
        {
            tracing::info!("load: BRAW preset '{}' → '{}' (migrated)", self.settings.fisheye_preset, vr180_pipeline::braw_imu::BRAW_PRESET_NAME);
            self.settings.fisheye_preset.clear();
            // …and the eye order that preset implies (a clip that was in
            // the batch before this preset existed never got it seeded).
            if let Some(p) = vr180_fisheye::presets::find(vr180_pipeline::braw_imu::BRAW_PRESET_NAME) {
                self.settings.fisheye_swap_eyes = p.left_eye_is_right_half;
            }
        }
        // Per-camera default preset — only when none is set yet, so a
        // remembered/explicit choice is preserved.
        if source_kind != vr180_pipeline::SourceKind::GoProEac
            && self.settings.fisheye_preset.is_empty()
        {
            let auto = match source_kind {
                vr180_pipeline::SourceKind::DjiOsv        => "DJI Osmo 360",
                vr180_pipeline::SourceKind::Insta360Insv
                | vr180_pipeline::SourceKind::Insta360Dng => "Insta360 X6",
                vr180_pipeline::SourceKind::KandaoQoocam
                | vr180_pipeline::SourceKind::KandaoDng   => vr180_pipeline::kandao_imu::KANDAO_PRESET_NAME,
                vr180_pipeline::SourceKind::BlackmagicRaw => vr180_pipeline::braw_imu::BRAW_PRESET_NAME,
                _                                         => "Custom",
            };
            self.settings.fisheye_preset = auto.to_string();
        }
        // Autoload the embedded log→Rec.709 LUT matching the source — DJI
        // D-Log M for OSV, GoPro GP-Log for `.360`, Insta360 I-Log for
        // `.insv` — only when no LUT is set (preserve a remembered/cleared
        // choice); swap the OTHER source's builtin on a clip switch since the
        // log curves are camera-specific.
        {
            use crate::decoder::{BUILTIN_OSMO_LUT_PATH, BUILTIN_GPLOG_LUT_PATH, BUILTIN_X6_LUT_PATH, BUILTIN_HLG_LUT_PATH};
            let wanted = match source_kind {
                vr180_pipeline::SourceKind::DjiOsv        => Some(BUILTIN_OSMO_LUT_PATH),
                vr180_pipeline::SourceKind::GoProEac      => Some(BUILTIN_GPLOG_LUT_PATH),
                vr180_pipeline::SourceKind::Insta360Insv  => Some(BUILTIN_X6_LUT_PATH),
                // The QooCam's HDR mode records HLG; its SDR mode is plain
                // Rec.709 and needs nothing.
                vr180_pipeline::SourceKind::KandaoQoocam if meta.hlg => Some(BUILTIN_HLG_LUT_PATH),
                _ => None,
            };
            // A BRAW clip is decoded in the log it was shot in (Blackmagic
            // Design Film Gen 5) and carries Blackmagic's own display
            // transform ("Gen 5 Film to Extended Video"): the helper writes
            // it out as a .cube and it is applied like the other cameras'
            // built-ins. Counts as a built-in for the swap-on-clip-switch rule.
            let wanted: Option<&str> = meta.braw_lut.as_deref().or(wanted);
            let cur_is_builtin = self.settings.lut_path == BUILTIN_OSMO_LUT_PATH
                || self.settings.lut_path == BUILTIN_GPLOG_LUT_PATH
                || self.settings.lut_path == BUILTIN_X6_LUT_PATH
                || self.settings.lut_path == BUILTIN_HLG_LUT_PATH
                || self.settings.lut_path.ends_with(" (embedded).cube");
            match wanted {
                Some(w) if self.settings.lut_path.is_empty() || cur_is_builtin => {
                    if self.settings.lut_path != w {
                        self.settings.lut_path = w.to_string();
                        self.settings.lut_intensity = 1.0;
                    }
                }
                None if cur_is_builtin => self.settings.lut_path.clear(),
                _ => {}
            }
        }
        // Seed the per-clip RS mode from the auto-detect — fresh loads only,
        // so a prior clip's tweak never carries over. The UI toggles still
        // override. Set before the snapshot so the decoder spawns with them
        // and gen isn't bumped on frame 0.
        if !meta.preserve_clip_settings {
            self.settings.rs_mode = meta.detected_rs_mode;
            // Readout from the file's SROT (the loader already found it);
            // the constant is only the no-SROT fallback.
            self.settings.rs_readout_ms = match meta.srot_ms {
                Some(ms) if ms > 0.5 && ms < 100.0 => ms,
                _ => crate::decoder::default_rs_readout_ms(),
            };
        }
        // Blackmagic RAW: the SDK's own development controls (Color ▸
        // Blackmagic RAW) replace the Tone and White balance stages for this
        // camera — those sections are hidden, so keep them neutral here or an
        // invisible setting would shape the image. The embedded Film→video
        // LUT fits the Film gamma only: off while an override picks another.
        if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw {
            let s = &mut self.settings;
            s.lift = 0.0; s.gamma = 1.0; s.gain = 1.0; s.shadow = 0.0; s.highlight = 0.0;
            s.temperature = 0.0; s.tint = 0.0; s.saturation = 1.0;
            if let (Some(g), Some(lut)) = (s.braw_color.gamma.as_deref(), meta.braw_lut.as_deref()) {
                if !vr180_braw::BrawColorInfo::is_film_gamma(g) && s.lut_path == lut { s.lut_path.clear(); }
            }
        }
        self.last_pushed_settings = self.settings.clone();

        let dims = vr180_core::eac::Dims::new(meta.width, meta.height);
        self.clip = Some(ClipInfo {
            width: meta.width, height: meta.height,
            fps: meta.fps, duration_sec: meta.duration_sec,
            frame_count: meta.frame_count,
            segments: meta.segments,
            eac_tile_w: dims.tile_w(),
            cori_count: meta.cori_count, grav_count: meta.grav_count,
            srot_ms: meta.srot_ms,
            source_kind,
            fisheye_eye_w: meta.fisheye_eye_w, fisheye_eye_h: meta.fisheye_eye_h,
            eac_cal_dim: meta.eac_cal_dim,
            foa_audio: meta.foa_audio,
            quad_audio: meta.quad_audio,
            braw_color: meta.braw_color.map(std::sync::Arc::new),
            braw_lut: meta.braw_lut.clone(),
        });
        self.loaded_path = Some(meta.path);

        // Kick off the decoder. Paused by default so the first frame paints
        // without the user pressing Play — unless they DID press Play during the
        // load (`play_when_ready`), in which case start playing so the intent
        // isn't lost.
        let start_paused = !std::mem::take(&mut self.play_when_ready);
        self.spawn_decoder(start_paused);
    }

    /// Poll the in-flight async load (if any) and apply it when ready; also
    /// recover from a decoder that died before delivering its first frame.
    /// Called once per `update`.
    fn poll_load(&mut self, ctx: &egui::Context) {
        // What to do once we've stopped borrowing `self.load_pending`.
        enum LoadStep { None, Clear, Apply(Box<LoadedMeta>) }
        let step = if let Some(pending) = &self.load_pending {
            // Keep repainting while a load is in flight — the worker thread
            // won't wake the UI on its own, so without this the "Loading…"
            // state could sit stale until the next user input.
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
            match pending.rx.try_recv() {
                Ok(Ok(meta)) if meta.token == pending.token => {
                    tracing::info!("load: metadata ready in {:?} — applying",
                        pending.started.elapsed());
                    LoadStep::Apply(Box::new(meta))
                }
                Ok(Ok(_)) => LoadStep::Clear, // superseded by a newer load
                Ok(Err(e)) => { tracing::error!("load failed: {e}"); LoadStep::Clear }
                Err(crossbeam_channel::TryRecvError::Empty) => LoadStep::None,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    tracing::error!("load worker disconnected before sending a result");
                    LoadStep::Clear
                }
            }
        } else {
            LoadStep::None
        };
        match step {
            LoadStep::Apply(meta) => { self.load_pending = None; self.finish_load(*meta); }
            LoadStep::Clear => { self.load_pending = None; }
            LoadStep::None => {}
        }
        // No-wedge: if the decoder thread has EXITED — a failure before frame 0,
        // a mid-stream error, or a clean end-of-clip — while we still believe
        // it's alive, reset the playback flags so Play/seek respawn a fresh
        // decoder instead of toggling a dead one (which would silently do
        // nothing). `control` is set to None by `stop_playback` on an explicit
        // stop, so a normal teardown doesn't trip this; the stale channels of a
        // self-exited decoder are released on the next `spawn_decoder`.
        let exited = self.decoder_alive
            && self.control.as_ref().map_or(false, |ctl| ctl.finished.load(Ordering::SeqCst));
        if exited {
            tracing::info!("decoder thread exited — resetting playback flags so Play respawns");
            let before_first_frame = self.decoder_starting;
            let was_playing = self.playing;
            self.decoder_starting = false;
            self.decoder_alive = false;
            self.playing = false;
            // A decoder that dies BEFORE its first frame leaves the panel on
            // "Click ▶ Play" with nothing to show for the load. Seen on a
            // network share (QooCam clip over SMB, 2026-10-17): a stalled read
            // reported as end-of-stream and the thread "exited cleanly" 240 ms
            // after construction with zero frames. Respawn ONCE on our own; a
            // second failure is left to the user, with the reason in the log.
            let is_still = self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
            if before_first_frame && !self.decoder_retried && !is_still {
                self.decoder_retried = true;
                tracing::warn!("decoder exited before its first frame — respawning once (paused={})", !was_playing);
                self.spawn_decoder(!was_playing);
            }
        }
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
        // Gate against play/seek respawns while a decoder is still coming up.
        // A fresh paused load (`start_paused`) always proceeds; only the
        // unpaused play/seek respawns are debounced. Without this, mashing
        // Play during a slow load spawns several detached decoder threads
        // that all block reading the same file, multiplying contention.
        if !start_paused && self.decoder_starting {
            tracing::info!("spawn_decoder: ignored — a decoder is still starting up");
            return;
        }
        self.stop_playback();
        let is_still = self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
        // A still only ever starts paused (its loop ignores the flag; a
        // "playing" photo would spin the 8 ms repaint cadence).
        let start_paused = start_paused || is_still;
        tracing::info!(
            "spawn_decoder: starting decoder for {} (paused={})",
            path.display(), start_paused
        );
        // Segment chain (GS01…, GS02…) for the loaded clip — played +
        // gyro-aggregated as one. Falls back to the lone path.
        let segments = self.clip.as_ref()
            .map(|c| c.segments.clone())
            .filter(|segs| !segs.is_empty())
            .unwrap_or_else(|| vec![path.clone()]);
        let cfg = DecoderConfig {
            path,
            segments,
            settings: self.settings.clone(),
            eye_w: self.settings.preview_eye_w,
        };
        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        // Native-res zoom stills (.360 EAC) flow on their own bounded(1)
        // channel — drained-to-latest, so a slow UI just drops stale stills.
        let (detail_tx, detail_rx) = crossbeam_channel::bounded(1);
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let control = Arc::new(DecoderControl {
            paused: std::sync::atomic::AtomicBool::new(start_paused),
            ui_ctx: Some(self.egui_ctx.clone()),
            settings: parking_lot::RwLock::new(self.settings.clone()),
            settings_generation: std::sync::atomic::AtomicU64::new(0),
            detected_calib: parking_lot::Mutex::new(None),
            want_detail: std::sync::atomic::AtomicBool::new(false),
            detail_tx: parking_lot::Mutex::new(Some(detail_tx)),
            detail_out: parking_lot::Mutex::new(None),
            finished: std::sync::atomic::AtomicBool::new(false),
            imu_ready: std::sync::atomic::AtomicBool::new(false),
            imu_progress: Default::default(),
            stab_loading: std::sync::atomic::AtomicBool::new(false),
            eye_match_readout: parking_lot::Mutex::new(None),
        });
        let pipeline = self.pipeline.clone();
        let control_for_thread = control.clone();
        std::thread::spawn(move || {
            // Signal the UI that this decoder is gone — from a Drop guard, so
            // it runs even if `start_decoder` PANICS. `finished` is the only
            // thing that clears `decoder_alive`/`decoder_starting`, so
            // unwinding past these stores left Play/Pause toggling a dead
            // worker forever and leaked the background IMU thread + its file
            // handle. (A wgpu error on the decoder thread is a panic by
            // default, which is exactly how we found this.)
            struct SignalFinished(Arc<crate::decoder::DecoderControl>);
            impl Drop for SignalFinished {
                fn drop(&mut self) {
                    use std::sync::atomic::Ordering;
                    // Abort any in-flight/paused background IMU read so it
                    // doesn't leak past the decoder's life.
                    self.0.imu_progress.abort.store(true, Ordering::SeqCst);
                    self.0.finished.store(true, Ordering::SeqCst);
                }
            }
            let _signal = SignalFinished(control_for_thread.clone());

            if let Err(e) = start_decoder(pipeline, cfg, control_for_thread.clone(), frame_tx, cmd_rx) {
                tracing::error!("decoder error: {e}");
            } else {
                tracing::info!("decoder thread exited cleanly");
            }
        });
        self.frame_rx = Some(frame_rx);
        self.detail_frame_rx = Some(detail_rx);
        self.cmd_tx = Some(cmd_tx);
        self.control = Some(control);
        self.playing = !start_paused;
        self.decoder_alive = true;
        self.decoder_starting = true;
        self.no_decoder_warned = false;
        self.last_pushed_settings = self.settings.clone();
        self.fps_stats = FpsStats::default();

        // Spawn the audio player alongside the video decoder. It runs
        // its own thread + cpal output stream and is held by the App
        // for the lifetime of the playback session. Multi-segment: feed
        // it an ffconcat playlist so the chain's audio plays + seeks as
        // one continuous timeline (matches the video chaining).
        // Not for a still: a DNG carries no audio, and the probe is an
        // ffmpeg open of the whole file on the UI thread (a stall on a
        // cold network share).
        if let Some(audio_path) = self.loaded_path.clone().filter(|_| !is_still) {
            let audio_path = self.clip.as_ref()
                .filter(|c| c.segments.len() > 1)
                .and_then(|c| vr180_pipeline::audio::build_segment_playlist(&c.segments, "preview"))
                .unwrap_or(audio_path);
            let t_audio = std::time::Instant::now();
            // 4-channel sources need the caller's mixing matrix: the codec
            // alone cannot tell the X6's AmbiX from the QooCam's microphones.
            let mix = self.clip.as_ref().and_then(|c| {
                if c.foa_audio { Some(vr180_pipeline::audio::StereoMix::Foa) }
                else if c.quad_audio { Some(vr180_pipeline::audio::StereoMix::Quad) }
                else { None }
            });
            self.audio_player = match crate::audio_player::AudioPlayer::open_with_mix(
                audio_path, start_paused, mix,
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
        // Pressed during the async-load window (clip not ready, no decoder yet):
        // remember the intent instead of dropping it — `finish_load` starts the
        // decoder playing once the clip is ready. Toggling lets a second press
        // cancel it. (The on-screen Play button is gated to a loaded clip; this
        // is reached via the Space shortcut.)
        if self.load_pending.is_some() {
            self.play_when_ready = !self.play_when_ready;
            return;
        }
        // A photo has no playback: Space only revives a dead render thread,
        // paused (a "playing" still would spin the 8 ms repaint cadence).
        if self.clip.as_ref().map_or(false, |c| c.source_kind.is_still()) {
            if !self.decoder_alive { self.spawn_decoder(true); }
            return;
        }
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
        self.detail_frame_rx = None;
        self.cmd_tx = None;
        self.control = None;
        self.playing = false;
        self.decoder_alive = false;
        // No live decoder → nothing is "starting". Left set, a Stop (or an
        // export of the loaded clip) before the first frame arrived kept
        // every later Play / Space / seek respawn gated off until a reload.
        self.decoder_starting = false;
        // Drop the audio player: cpal stream stops, worker thread
        // shuts down (Stop command).
        self.audio_player = None;
    }

    /// Re-seed the per-eye Override calibration whenever a NEW clip's in-file
    /// calibration arrives. Without this, loading a different clip while
    /// Override is on would keep the previous clip's factory values. Only
    /// re-seeds eyes currently in Override (Override-off eyes read the in-file
    /// calib directly via the resolver's auto path). The off→on toggle is
    /// handled separately in `draw_eye_lens`.
    fn maybe_reseed_fisheye_override(&mut self) {
        let detected = self.control.as_ref().and_then(|c| *c.detected_calib.lock());
        // No control (stopped) or not seeded yet: KEEP the last seed. A
        // same-clip respawn (Restart preview, Play after Stop) passes through
        // `None` for ~100 ms and must not read as a new clip — that
        // overwrote hand-tuned Override values with the factory ones. A new
        // clip resets `fisheye_last_seeded` in `unload_clip`.
        let Some(d) = detected else { return; };
        if Some(d) == self.fisheye_last_seeded {
            return; // same clip's calibration — nothing to do
        }
        self.fisheye_last_seeded = Some(d);
        tracing::info!("RESEED-DBG: detected changed; suppress={} ov_l={} cx_l {:.4}->{:.4} fov_l {:.2}->{:.2}",
            self.suppress_reseed_once, self.settings.fisheye_override_left,
            self.settings.fisheye_cx_norm_left, d.left.cx_norm,
            self.settings.fisheye_fov_deg_left, d.left.fov_deg);
        // Clip-list restore in flight: the settings carry hand-tuned
        // per-clip Override values — keep them instead of re-seeding
        // from the file. One-shot (the next clip change reseeds again).
        if std::mem::take(&mut self.suppress_reseed_once) {
            return;
        }
        if self.settings.fisheye_override_left {
            self.settings.fisheye_fov_deg_left = d.left.fov_deg;
            self.settings.fisheye_cx_norm_left = d.left.cx_norm;
            self.settings.fisheye_cy_norm_left = d.left.cy_norm;
            self.settings.fisheye_k_left = d.left.k;
            self.settings.fisheye_k5_left = d.left.k5;
            self.settings.fisheye_p_left = d.left.p;
        }
        if self.settings.fisheye_override_right {
            self.settings.fisheye_fov_deg_right = d.right.fov_deg;
            self.settings.fisheye_cx_norm_right = d.right.cx_norm;
            self.settings.fisheye_cy_norm_right = d.right.cy_norm;
            self.settings.fisheye_k_right = d.right.k;
            self.settings.fisheye_k5_right = d.right.k5;
            self.settings.fisheye_p_right = d.right.p;
        }
    }

    /// Whether a paused frame should be shown as the native-resolution still:
    /// when the alignment magnifier is zoomed in, and always in the reframed
    /// mode (its viewport shows a small part of the lens, so the live
    /// working-res preview is soft and the magnifier is inert there).
    fn wants_full_res_still(&self) -> bool {
        self.preview_zoom > 1.001
            || matches!(self.settings.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Reframe)
            || self.snapshot_pending.is_some()
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
        } else if !self.no_decoder_warned && self.loaded_path.is_some() && self.load_pending.is_none() {
            // Once per stop, for a LOADED clip (a load restores settings
            // before its decoder exists — not a lost preview): makes a
            // "preview stopped following the sliders" report diagnosable.
            self.no_decoder_warned = true;
            tracing::warn!("settings changed with no decoder alive — the preview will not follow until it is restarted");
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
            // Per-kind map is the source of truth; the legacy single
            // `settings.json` is kept in sync for the no-clip startup state.
            self.save_active_kind_template();
            Settings::save_kind_map(&self.kind_settings);
            self.settings.save_persisted();
            self.last_saved_settings = self.settings.clone();
            self.last_settings_save_at = std::time::Instant::now();
        }
    }

    /// Draw the floating export-options window when `export_opts_visible`
    /// is set. The window is dismissable; "Start Export…" hands off to
    /// `commit_export()` which pops the save-file dialog and spawns the
    /// encode worker.
    /// Persistent bottom export bar — the single home for output settings +
    /// running the queue (replaces the old Batch window AND the Export-options
    /// window). Idle: Output / Format dropdowns + Export selected / Export all.
    /// Running: overall progress bar + ETA + Skip / Stop. A finished run leaves
    /// a dismissable summary chip.
    fn draw_export_bar(&mut self, ctx: &egui::Context) {
        // Snapshot everything the panel needs up front so the closure only
        // mutably borrows `self.export_opts` (egui disjoint-field capture).
        let running = self.batch_running;
        let run_progress = self.overall_run_progress();
        let cur_nr = self.batch_current
            .and_then(|i| self.batch.get(i))
            .map_or(false, |b| b.settings.denoise_strength > 0.0);
        let n_all = self.batch.iter().filter(|b| matches!(b.status,
            BatchStatus::Idle | BatchStatus::Queued
            | BatchStatus::Failed(_) | BatchStatus::Skipped)).count();
        let n_sel = self.batch.iter().filter(|b| b.selected).count();
        let can_export = self.export_job.is_none() && !running;
        let has_active = self.active_clip.is_some();
        let out_label = match &self.batch_out_dir {
            Some(d) => d.file_name().map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| d.to_string_lossy().to_string()),
            None => tr("next to source").to_string(),
        };
        let has_out_dir = self.batch_out_dir.is_some();
        let clip_fps = self.clip.as_ref().map(|c| c.fps).unwrap_or(30.0);
        let is_still = self.clip.as_ref().map(|c| c.source_kind.is_still()).unwrap_or(false);
        let res_label = match self.settings.reframe_aspect_if_active() {
            Some(aspect) => self.export_opts.reframe_size.label(aspect),
            None if is_still && self.export_opts.resolution == ExportResolution::Native => {
                let (ew, eh) = self.export_eye_dims();
                format!("{} × {}", ew * 2, eh)
            }
            None => self.export_opts.resolution.label().to_string(),
        };
        let h265_mbps = self.export_opts.effective_h265_mbps(
            self.settings.reframe_aspect_if_active(), clip_fps);
        let fmt_label = if is_still {
            // A still has no codec / bitrate: container + size only.
            format!("{} · {}", self.export_opts.still_format.label(), res_label)
        } else { format!("{} · {}{}{}",
            self.export_opts.codec.label(),
            res_label,
            match self.export_opts.codec {
                ExportCodec::H265 => format!(" · {} Mbps · {}-bit",
                    h265_mbps, self.export_opts.h265_bit_depth),
                ExportCodec::ProRes => format!(" · {}", self.export_opts.prores_profile.label()),
            },
            if self.export_opts.beyondvr_hack && self.settings.reframe_aspect_if_active().is_none() {
                format!(" · BeyondVR {:.0}%", self.export_opts.beyondvr_scale_pct)
            } else { String::new() }) };
        let summary = self.last_batch_summary.as_ref().map(|s| {
            let col = if s.failed > 0 { Color32::from_rgb(230, 160, 90) }
                      else { Color32::from_rgb(120, 200, 120) };
            let txt = if s.failed > 0 {
                format!("✓ {} done · {} failed · {}", s.done, s.failed, format_eta(s.elapsed.as_secs_f64()))
            } else {
                format!("✓ {} exported · {}", s.done, format_eta(s.elapsed.as_secs_f64()))
            };
            (txt, col)
        });

        let mut do_export_selected = false;
        let mut do_export_all = false;
        let mut do_skip = false;
        let mut do_stop = false;
        let mut do_choose_dir = false;
        let mut do_clear_dir = false;
        let mut dismiss_summary = false;
        let mut dismiss_gpu_error = false;
        let mut toggle_settings = false;

        egui::TopBottomPanel::bottom("export_bar").exact_height(40.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                if running {
                    if let Some((done, total, eta, pos, n)) = run_progress {
                        let frac = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        ui.add(egui::ProgressBar::new(frac).desired_width(240.0)
                            .text(format!("{:.0}%", frac * 100.0)));
                        let eta_txt = eta.map(|e| if cur_lang() == Lang::Zh {
                            format!(" · 约剩 {}", format_eta(e))
                        } else {
                            format!(" · ~{} left", format_eta(e))
                        }).unwrap_or_default();
                        let nr_txt = if cur_nr {
                            if cur_lang() == Lang::Zh { " · 降噪" } else { " · NR" }
                        } else { "" };
                        let item_txt = if cur_lang() == Lang::Zh {
                            format!("第 {pos}/{n} 项{eta_txt}{nr_txt}")
                        } else {
                            format!("item {pos}/{n}{eta_txt}{nr_txt}")
                        };
                        ui.label(RichText::new(item_txt).color(Color32::from_rgb(120, 180, 255)));
                    } else {
                        ui.label(RichText::new(tr("starting…")).color(Color32::GRAY));
                    }
                    // Slow-path note from the pipeline (portable CPU loop, or
                    // a CPU encoder fallback) — users shouldn't need the log
                    // to know why an export is slow.
                    if let Some(note) = vr180_pipeline::fisheye_export::export_path_note() {
                        ui.label(RichText::new(format!("⚠ {note}"))
                            .small().color(Color32::from_rgb(230, 160, 90)));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(tr("Stop")).clicked() { do_stop = true; }
                        if ui.button(tr("Skip current")).clicked() { do_skip = true; }
                    });
                } else {
                    ui.menu_button(format!("📁  {out_label}"), |ui| {
                        if ui.button(tr("Choose folder…")).clicked() { do_choose_dir = true; ui.close_menu(); }
                        ui.add_enabled_ui(has_out_dir, |ui| {
                            if ui.button(tr("Next to each source")).clicked() { do_clear_dir = true; ui.close_menu(); }
                        });
                    }).response.on_hover_text(tr("Output folder for exports"));
                    // A toggle (not a menu) — a settings menu closes the moment
                    // you touch a ComboBox/slider inside it. Opens a small
                    // non-modal window instead (drawn after this panel).
                    if ui.selectable_label(self.show_export_settings, format!("⚙  {fmt_label}"))
                        .on_hover_text(tr("Output format, resolution & metadata")).clicked()
                    {
                        toggle_settings = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_enabled_ui(can_export && n_all > 0, |ui| {
                            if ui.button(RichText::new(format!("{} ({n_all})", tr("Export all"))).strong()).clicked() {
                                do_export_all = true;
                            }
                        });
                        let sel_label = if n_sel > 0 {
                            format!("{} ({n_sel})", tr("Export selected"))
                        } else {
                            tr("Export selected").to_string()
                        };
                        ui.add_enabled_ui(can_export && (n_sel > 0 || has_active), |ui| {
                            if ui.button(sel_label).on_hover_text(tr(if n_sel > 0 {
                                "Export the ticked clips"
                            } else {
                                "Export the active clip"
                            })).clicked() { do_export_selected = true; }
                        });
                        if let Some((txt, col)) = &summary {
                            if ui.add(egui::Label::new(RichText::new(txt).color(*col))
                                .sense(egui::Sense::click()))
                                .on_hover_text(tr("Click to dismiss")).clicked() {
                                dismiss_summary = true;
                            }
                        }
                        // An uncaptured GPU error used to be a panic; now it
                        // is recorded and shown here, so a GPU fault is
                        // visible instead of a silent wedge. Click to dismiss.
                        if let Some(err) = vr180_pipeline::gpu::last_gpu_error() {
                            if ui.add(egui::Label::new(RichText::new(
                                    format!("⚠ {}", tr("GPU error — see the log")))
                                    .small().color(Color32::from_rgb(230, 120, 120)))
                                .sense(egui::Sense::click()))
                                .on_hover_text(format!("{err}

{}", tr("Click to dismiss")))
                                .clicked() {
                                dismiss_gpu_error = true;
                            }
                        }
                    });
                }
            });
        });

        if do_choose_dir {
            self.spawn_dialog(ctx, || {
                rfd::FileDialog::new().pick_folder().map(DialogResult::BatchOutDir)
            });
        }
        if do_clear_dir { self.batch_out_dir = None; }
        if dismiss_summary { self.last_batch_summary = None; }
        if dismiss_gpu_error { vr180_pipeline::gpu::clear_gpu_error(); }
        if toggle_settings { self.show_export_settings = !self.show_export_settings; }
        if do_export_selected { self.export_selected(); }
        if do_export_all { self.start_batch(); }
        if do_skip { self.skip_current_batch_item(); }
        if do_stop { self.stop_batch(); }

        // Format settings popover — a real (non-modal) window so ComboBoxes
        // and sliders stay open while you edit them.
        if self.show_export_settings {
            let mut open = true;
            let clip_fps = self.clip.as_ref().map(|c| c.fps).unwrap_or(30.0);
            let is_still = self.clip.as_ref().map(|c| c.source_kind.is_still()).unwrap_or(false);
            let still_dims = self.clip.as_ref().filter(|c| c.source_kind.is_still()).map(|c| (c.width, c.height));
            egui::Window::new(tr("Export format"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -150.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(360.0);
                    Self::export_options_ui(ui, &mut self.export_opts, "export",
                        self.settings.reframe_aspect_if_active(), clip_fps, is_still, still_dims);
                });
            self.show_export_settings = open;
        }
    }

    /// The export-output controls (resolution / codec / codec-specific /
    /// VR180 metadata) shown in the export bar's Format dropdown. `id_prefix`
    /// keeps the ComboBox ids distinct.
    fn export_options_ui(
        ui: &mut egui::Ui, opts: &mut ExportOptions, id_prefix: &str,
        reframe: Option<crate::decoder::ReframeAspect>,
        fps: f32,
        is_still: bool,
        still_dims: Option<(u32, u32)>,
    ) {
        if let Some(aspect) = reframe {
            // Reframed view: the exact viewport at a per-eye size.
            ui.label(RichText::new(tr("Per-eye size")).strong());
            opts.reframe_size = opts.reframe_size.clamped(is_still, aspect);
            egui::ComboBox::from_id_source(format!("{id_prefix}_reframe_size"))
                .selected_text(opts.reframe_size.label(aspect))
                .width(260.0)
                .show_ui(ui, |ui| {
                    for &sz in ReframeSize::choices(is_still, aspect) {
                        ui.selectable_value(&mut opts.reframe_size, sz, sz.label(aspect));
                    }
                });
            ui.label(RichText::new(tr("Reframed output is not VR180 — no projection metadata is written."))
                .small().color(Color32::GRAY));
        } else {
            ui.label(RichText::new(tr("Resolution")).strong());
            let native_label: String = if is_still {
                still_dims.map(|(w, h)| format!("{w} × {h} (full)")).unwrap_or_else(|| "native (full)".to_string())
            } else { ExportResolution::Native.label().to_string() };
            let sel_label: &str = if opts.resolution == ExportResolution::Native { &native_label } else { ExportResolution::R8k.label() };
            egui::ComboBox::from_id_source(format!("{id_prefix}_resolution"))
                .selected_text(sel_label)
                .width(220.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut opts.resolution,
                        ExportResolution::Native, native_label.clone());
                    ui.selectable_value(&mut opts.resolution,
                        ExportResolution::R8k, ExportResolution::R8k.label());
                });
        }
        ui.add_space(8.0);

        if is_still {
            ui.label(RichText::new(tr("Still format")).strong());
            egui::ComboBox::from_id_source(format!("{id_prefix}_still_format"))
                .selected_text(opts.still_format.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for f in [StillFormatChoice::Png16, StillFormatChoice::Jpeg, StillFormatChoice::LinearDng] {
                        ui.selectable_value(&mut opts.still_format, f, f.label());
                    }
                });
            ui.label(RichText::new(tr("DNG keeps the camera-native linear data with the raw colour tags — grade it in a raw developer."))
                .small().color(Color32::GRAY));
            if opts.still_format == StillFormatChoice::LinearDng {
                ui.label(RichText::new(tr("Your grade travels as metadata: white balance → AsShotNeutral, tone (lift/gamma/gain/shadow/highlight) → profile tone curve, saturation → profile look table (Adobe); per-eye trims are baked. Not carried: 3D LUT, sharpen."))
                    .small().color(Color32::GRAY));
            }
            ui.add_space(8.0);
            return;
        }

        ui.label(RichText::new(tr("Codec")).strong());
        egui::ComboBox::from_id_source(format!("{id_prefix}_codec"))
            .selected_text(opts.codec.label())
            .width(220.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut opts.codec, ExportCodec::H265,   ExportCodec::H265.label());
                ui.selectable_value(&mut opts.codec, ExportCodec::ProRes, ExportCodec::ProRes.label());
            });
        ui.add_space(8.0);

        match opts.codec {
            ExportCodec::H265 => {
                ui.label(RichText::new(tr("H.265 bitrate")).strong());
                if let Some(aspect) = reframe {
                    // Reframed: its own rate, seeded from the size / aspect /
                    // frame-rate recommendation until the user pins a value
                    // (dragging back onto the recommendation un-pins it).
                    let rec = opts.reframe_size.recommended_bitrate_mbps(aspect, fps);
                    if !opts.reframe_bitrate_custom {
                        opts.reframe_bitrate_mbps = rec;
                    }
                    if ui.add(egui::Slider::new(&mut opts.reframe_bitrate_mbps, 10..=300)
                        .text("Mbps")).changed()
                    {
                        opts.reframe_bitrate_custom = opts.reframe_bitrate_mbps != rec;
                    }
                    let (w, h) = opts.reframe_size.eye_dims(aspect);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!(
                            "{}: {rec} Mbps · {} × {h} @ {fps:.0} fps", tr("Recommended"), 2 * w,
                        )).small().color(Color32::GRAY));
                        if opts.reframe_bitrate_custom
                            && ui.small_button(tr("Use recommended")).clicked()
                        {
                            opts.reframe_bitrate_mbps = rec;
                            opts.reframe_bitrate_custom = false;
                        }
                    });
                } else {
                    ui.add(egui::Slider::new(&mut opts.h265_bitrate_mbps, 20..=500)
                        .text("Mbps"));
                }
                ui.add_space(4.0);
                // 10-bit (Main10) is the default and stays 10-bit
                // end-to-end; 8-bit (Main) is offered for users who
                // need the wider-compatibility profile.
                ui.label(RichText::new(tr("Bit depth")).strong());
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut opts.h265_bit_depth, 10u8, tr("10-bit (Main10)"));
                    ui.selectable_value(&mut opts.h265_bit_depth, 8u8,  tr("8-bit (Main)"));
                });
                // Encoder choice — hardware vs software. macOS always
                // uses VideoToolbox, so only surface this elsewhere.
                if !cfg!(target_os = "macos") {
                    ui.add_space(4.0);
                    ui.label(RichText::new(tr("Encoder")).strong());
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut opts.h265_hardware, true,  tr("Hardware (NVENC)"));
                        ui.selectable_value(&mut opts.h265_hardware, false, tr("Software (libx265)"));
                    });
                    ui.label(RichText::new(if opts.h265_hardware {
                        "GPU HEVC — fast (NVIDIA); auto-falls back to libx265 if unavailable."
                    } else {
                        "CPU HEVC — slow at VR180 resolutions, max software quality."
                    }).weak().size(11.0));
                }
            }
            ExportCodec::ProRes => {
                ui.label(RichText::new(tr("ProRes profile")).strong());
                egui::ComboBox::from_id_source(format!("{id_prefix}_prores_profile"))
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

        if reframe.is_none() {
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label(RichText::new(tr("VR180 metadata")).strong());
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
            tr("Apple Vision Pro (APMP — vexu + hfov)"));
        if matches!(target, MetaTarget::Apmp) {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(tr("Camera baseline"));
                ui.add(egui::Slider::new(&mut opts.apmp_baseline_mm, 30.0..=120.0)
                    .suffix(" mm").fixed_decimals(1));
            });
        }
        ui.radio_value(&mut target, MetaTarget::Youtube,
            tr("YouTube (Spherical V2 — st3d + sv3d)"));
        ui.radio_value(&mut target, MetaTarget::None,
            tr("None (no VR180 metadata)"));
        opts.inject_apmp    = matches!(target, MetaTarget::Apmp);
        opts.inject_youtube = matches!(target, MetaTarget::Youtube);
        ui.label(RichText::new(tr(
            "APMP and Spatial V2 conflict in the same atoms, so \
             only one can be active per file. Re-run the export \
             with the other target if you need a second copy."
        )).small().color(Color32::GRAY));
        }

        // The BeyondVR eye-scale hack is a half-equirect (VR180) fix-up —
        // a reframed view neither shows nor applies it.
        if reframe.is_some() {
            return;
        }
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(4.0);
        ui.checkbox(&mut opts.beyondvr_hack, tr("BeyondVR Hack"));
        if opts.beyondvr_hack {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(tr("Eye scale"));
                // step_by(1.0): whole percents only — a fractional value
                // (e.g. 99.7) would DISPLAY as "100 %" yet still engage
                // the scale stage and leave a thin black outline.
                ui.add(egui::Slider::new(&mut opts.beyondvr_scale_pct, 50.0..=100.0)
                    .suffix(" %").fixed_decimals(0).step_by(1.0));
                opts.beyondvr_scale_pct = opts.beyondvr_scale_pct.round();
            });
            ui.label(RichText::new(tr(
                "Scales each eye about its own center at the final \
                 output stage, padding with black. Half-equirect \
                 (VR180) output only."
            )).small().color(Color32::GRAY));
        }
    }

}

impl eframe::App for App {
    // eframe 0.34 made `ui` the required method and deprecated `update`.
    // Our UI builds multiple panels (top/side/central) directly from
    // `ctx`, so we keep overriding `update` (eframe's runner still calls
    // it) and leave `ui` empty.
    fn ui(&mut self, _ui: &mut egui::Ui, _frame: &mut eframe::Frame) {}

    /// Auto-update hand-off (Windows): the event loop has unwound — the
    /// window is gone, settings flushed, decode/audio threads down. Spawn
    /// the verified silent installer now; it replaces our files and
    /// relaunches the app (windows.iss `[Run] Check: WizardSilent`).
    fn on_exit(&mut self) {
        #[cfg(target_os = "windows")]
        if let Some(staged) = self.pending_update_install.take() {
            match crate::updater::spawn_installer(&staged) {
                Ok(()) => tracing::info!(
                    "updater: installer spawned on exit (v{})", staged.version),
                Err(e) => tracing::error!("updater: {e}"),
            }
        }
    }

    #[allow(deprecated)]
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Mirror the chosen language into the process global that `tr` reads.
        set_cur_lang(self.lang);
        if !self.pending_open.is_empty() {
            let paths = std::mem::take(&mut self.pending_open);
            self.open_paths(paths);
        }
        // Still + DNG output: the preview and the sidebar follow what the
        // DNG can carry (the flag rides on Settings so the decoder sees it).
        let still_dng = self.clip.as_ref().map_or(false, |c| c.source_kind.is_still())
            && self.export_opts.still_format == StillFormatChoice::LinearDng;
        if self.settings.still_dng_mode != still_dng {
            self.settings.still_dng_mode = still_dng;
        }
        let still_native = self.still_native_wanted
            && self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
        if self.settings.still_native_zoom != still_native {
            self.settings.still_native_zoom = still_native;
        }
        // Drain decoder frames first so the most recent texture is
        // ready by the time we render the central panel.
        self.drain_frames(ctx);
        // Apply a finished async clip load (and recover a decoder that died
        // before its first frame). Kept right after frame drain so a clip
        // that just finished loading spawns its decoder this same frame.
        self.poll_load(ctx);
        // Re-seed Override calib when a new clip's in-file calibration
        // arrives (so Override tracks the new clip, not the old one). Runs
        // before maybe_push_settings so the re-seeded values reach the decoder.
        self.maybe_reseed_fisheye_override();
        // Eye-orientation toggles (Swap L↔R / upside-down) bind at decoder
        // START (the dual-stream iterators take swap at open), so a live
        // toggle needs a clip reload to take effect everywhere — decode
        // order, per-eye calib, detail worker. Detect the change and
        // reload the current clip automatically.
        let eye_orient = (self.settings.fisheye_swap_eyes, self.settings.camera_upside_down);
        if eye_orient != self.eye_orientation_applied {
            self.eye_orientation_applied = eye_orient;
            if let Some(p) = self.loaded_path.clone() {
                tracing::info!("eye orientation changed (swap={}, upside_down={}) — reloading clip",
                    eye_orient.0, eye_orient.1);
                // Same clip — keep its trim + RS mode across the reload.
                self.load_file_inner(p, true);
            }
        }
        // Drain export-job progress + check for completion.
        self.poll_export_job();
        // Auto-update: drain events + periodic re-check + popover.
        self.poll_updater(ctx);
        self.poll_dialogs();
        self.poll_auto_align(ctx);
        self.draw_update_popover(ctx);
        if self.update_installing {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Drive the full-res still for the zoom magnifier (and snapshots).
        self.poll_full_res(ctx);
        self.poll_snapshot(ctx);
        // Import merge/individual prompt (shown when a multi-file recording
        // is picked). Modal-ish: it sits on top until resolved.
        self.draw_import_prompt(ctx);
        // Keep repainting while a run is mid-flight so completion chaining +
        // the export-bar progress update without waiting for mouse input.
        if self.batch_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        // Force a continuous repaint while an export is in flight so
        // the progress bar updates without needing mouse input.
        if self.export_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Handle dropped files (drag-and-drop into the window). All
        // dropped fisheye files join the clip list; the first becomes
        // active.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
        });
        if !dropped.is_empty() {
            self.open_paths(dropped);
        }

        // Keyboard shortcuts. Only fire when no text widget owns the
        // focus — egui's `wants_keyboard_input` handles that for us.
        // We capture trim-in/out actions inside the closure (borrows
        // `&i.modifiers` immutably) and apply them after.
        let mut pending_trim_in:  Option<Option<f64>> = None;
        let mut pending_trim_out: Option<Option<f64>> = None;
        let mut pending_snapshot = false;
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                // S — snapshot the frame on screen. (Stop has no hotkey: the
                // old plain-S stop tore the decoder down mid-session by
                // accident too easily; the ■ Stop button remains.)
                if i.key_pressed(egui::Key::S) && i.modifiers.is_none() {
                    pending_snapshot = true;
                }
                if i.key_pressed(egui::Key::Space) {
                    self.toggle_play_pause();
                }
                if i.key_pressed(egui::Key::O)
                    && (i.modifiers.command || i.modifiers.ctrl)
                {
                    self.pick_and_load_file(ctx);
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
        if pending_snapshot { self.request_snapshot(); }

        // ─── Top toolbar (file + preview-size only; transport is bottom) ─
        egui::TopBottomPanel::top("toolbar")
            .exact_height(40.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Brand + version, top-left — THE version display (shown
                    // nowhere else; the window title carries no version).
                    // Clickable: opens the update popover / checks for
                    // updates. Gains an orange "⬆ vNEW" badge when a newer
                    // version is known.
                    let brand = format!(
                        "▶ VR180 Silver Bullet  v{}", env!("CARGO_PKG_VERSION"));
                    let brand_resp = ui.add(
                        egui::Button::new(
                            RichText::new(brand)
                                .strong()
                                .color(Color32::from_rgb(90, 166, 255)),
                        )
                        .frame(false),
                    )
                    .on_hover_text(tr("Check for updates"));
                    if let Some(info) = &self.update_available {
                        ui.add_space(-4.0);
                        if ui.add(
                            egui::Button::new(
                                RichText::new(format!("⬆ v{} {}", info.version, tr("update")))
                                    .small()
                                    .color(Color32::from_rgb(255, 190, 60)),
                            )
                            .frame(false),
                        ).clicked() {
                            self.show_update_popover = true;
                        }
                    }
                    if brand_resp.clicked() {
                        self.show_update_popover = true;
                        if self.update_available.is_none() && !self.update_installing {
                            self.update_manual_check = true;
                            self.update_status = None;
                            self.update_last_check = std::time::Instant::now();
                            crate::updater::spawn_check(self.updater_tx.clone());
                        }
                    }
                    ui.separator();
                    if ui.button(tr("Load video…")).clicked() {
                        self.pick_and_load_file(ctx);
                    }
                    // Language toggle (EN / 中文), persisted.
                    ui.separator();
                    if ui.selectable_label(cur_lang() == Lang::En, "EN").clicked() {
                        self.lang = Lang::En;
                        self.save_lang();
                    }
                    if ui.selectable_label(cur_lang() == Lang::Zh, "中文").clicked() {
                        self.lang = Lang::Zh;
                        self.save_lang();
                    }

                    // (Export controls now live in the persistent bottom
                    // export bar — see `draw_export_bar`.)

                    // A still renders at its working resolution once the
                    // settings rest (the size here only shapes the quick
                    // renders during a drag) — not offered for photos.
                    let preview_size_applies = !self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
                    if preview_size_applies {
                        ui.separator();
                        ui.label(RichText::new(tr("Preview size")).color(Color32::GRAY));
                        egui::ComboBox::from_id_source("preview_w")
                            .selected_text(format!("{}", self.settings.preview_eye_w))
                            .show_ui(ui, |ui| {
                                for &w in &[512_u32, 768, 1024, 1280, 1536, 1920] {
                                    ui.selectable_value(&mut self.settings.preview_eye_w, w,
                                        format!("{w}"));
                                }
                            });
                    }

                    ui.separator();
                    ui.label(RichText::new(tr("View")).color(Color32::GRAY));
                    let three_d_on = self.three_d.enabled;
                    ui.add_enabled_ui(!three_d_on, |ui| {
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
                            ui.selectable_value(&mut self.settings.preview_mode,
                                M::SingleEye, M::SingleEye.as_str());
                        });
                    }).response.on_disabled_hover_text(tr("SBS while the 3D display is on"));

                    // Single-eye: a toggle to switch which eye. Flipping it
                    // only changes which eye renders — the zoom/pan viewpoint
                    // is left untouched, so the view stays put.
                    if self.settings.preview_mode == crate::decoder::PreviewMode::SingleEye {
                        let eye = if self.settings.preview_eye_right { "Right" } else { "Left" };
                        if ui.button(format!("Eye: {eye}  ⇄"))
                            .on_hover_text(tr("Switch eye (keeps zoom/pan)"))
                            .clicked()
                        {
                            self.settings.preview_eye_right = !self.settings.preview_eye_right;
                        }
                    }

                    // External 3D display output (side-by-side monitor / AR
                    // glasses). Toggle: opens / closes the output viewport.
                    ui.separator();
                    let mut three_d = self.three_d.enabled;
                    if ui.toggle_value(&mut three_d, tr("3D display"))
                        .on_hover_text(tr("Show the stereo pair on a side-by-side 3D monitor or AR glasses that appear as one wide screen (3840×1080): left eye in the left half, right eye in the right. Goes fullscreen on that screen automatically when one is connected; otherwise opens a window to drag onto it (F = fullscreen). Esc closes it."))
                        .changed()
                    {
                        if three_d { self.enable_3d_display(); } else { self.disable_3d_display(); }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // (Version lives ONLY in the top-left brand label,
                        // which is the clickable updater entry point.)

                        // Export progress badge in the right corner —
                        // only when a job is in flight.
                        if let Some(job) = &self.export_job {
                            ui.separator();
                            if ui.small_button(tr("Cancel")).clicked() {
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
                                ui.label(RichText::new(tr("starting…"))
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
            // Cap so a stale persisted width (or any future content-width
            // feedback) can never squeeze the preview out of view.
            .max_width(420.0)
            .show(ctx, |ui| {
              // Scroll the controls vertically so every section stays
              // reachable even when many are expanded / the window is short.
              egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                // Longer slider tracks (egui default is 100). FIXED width —
                // sizing from available_width() feeds back: wider sliders
                // grow the panel's content width, which grows the panel,
                // which widens the sliders… until the preview is squeezed
                // out. 150 + value box + label fits the 300 default panel.
                ui.spacing_mut().slider_width = 150.0;
                ui.add_space(8.0);
                egui::CollapsingHeader::new(RichText::new(tr("Source")).strong())
                    .default_open(true)
                    .show(ui, |ui| { self.draw_source_panel(ui); });

                // Fisheye sources (OSV / SBS / BRAW): Stabilization +
                // Output are top-level sections; the camera-preset /
                // per-eye FOV / center / KB panel is its own (collapsed)
                // section. GoPro EAC keeps the existing stab + RS panels.
                let is_still = matches!(kind, Some(k) if k.is_still());
                if is_still {
                    // Still photo: no timeline, no gyro, no rolling shutter,
                    // no audio — only the geometry and colour sections apply.
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("View adjustment")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_view_adjust_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Output")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_output_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Fisheye lens")).strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_fisheye_panel(ui); });
                } else if matches!(kind, Some(k) if k.is_fisheye()) {
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("View adjustment")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_view_adjust_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Stabilization")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_stab_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Output")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_output_panel(ui); });

                    // X6 recorded in 360° Audio: its 4-channel AmbiX track
                    // serves the same Stereo / Ambisonic / APAC choice as .360.
                    if self.clip.as_ref().map_or(false, |c| c.foa_audio || c.quad_audio) {
                        ui.add_space(4.0);
                        egui::CollapsingHeader::new(
                            RichText::new(tr("Audio")).strong()
                        )
                        .default_open(false)
                        .show(ui, |ui| { self.draw_audio_panel(ui); });
                    }

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Fisheye lens")).strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_fisheye_panel(ui); });
                } else {
                    // GoPro EAC: the same View adjustment (global pano-map +
                    // per-eye stereo offset) the OSV path has, then the
                    // GoPro stab + RS panels.
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("View adjustment")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_view_adjust_panel(ui); });

                    // Per-eye lens Override (FOV / center / KB) — the .360
                    // re-dewarp analog of the OSV Fisheye-lens panel, seeded
                    // from the file's GEOC factory calibration.
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Lens calibration")).strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_eac_lens_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Stabilization")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_stab_panel(ui); });

                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Rolling shutter")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_rs_panel(ui); });

                    // Output projection (half-equirect VR180 / fisheye) —
                    // same panel + setting the OSV path uses.
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Output")).strong()
                    )
                    .default_open(true)
                    .show(ui, |ui| { self.draw_fisheye_output_panel(ui); });

                    // Audio track (.360 carries stereo AAC + 4-ch ambisonic).
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Audio")).strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_audio_panel(ui); });
                }

                ui.add_space(4.0);
                egui::CollapsingHeader::new(
                    RichText::new(tr("Color")).strong()
                )
                .default_open(false)
                .show(ui, |ui| { self.draw_color_panel(ui); });

                // Noise Reduction — its own section for emphasis. Only shown
                // on macOS with a capable VideoToolbox temporal filter and an
                // exportable clip loaded; auto-hidden on Windows / older macOS
                // / incapable hardware (see `denoise_supported`).
                let nr_show = Self::denoise_supported()
                    && self.clip.as_ref().map_or(false, |c| c.source_kind.is_exportable())
                    && !is_still; // temporal filter — meaningless for one frame
                if nr_show {
                    ui.add_space(4.0);
                    egui::CollapsingHeader::new(
                        RichText::new(tr("Noise Reduction")).strong()
                    )
                    .default_open(false)
                    .show(ui, |ui| { self.draw_denoise_panel(ui); });
                }
                }); // ScrollArea
            });

        // ─── Bottom: status bar + transport ─────────────────────
        egui::TopBottomPanel::bottom("status").exact_height(24.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let status = if self.playing { tr("▶ playing") } else { tr("paused") };
                ui.label(RichText::new(status).small());
                ui.separator();
                ui.label(RichText::new(format!("{:.1} fps", self.fps_stats.last_fps))
                    .small().color(Color32::from_rgb(76, 208, 125)));
                if let Some((msg, at)) = &self.snapshot_status {
                    if at.elapsed().as_secs_f32() < 5.0 {
                        ui.separator();
                        ui.label(RichText::new(msg.clone()).small().color(Color32::from_rgb(255, 200, 120)));
                        ctx.request_repaint_after(std::time::Duration::from_millis(500));
                    }
                }
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
                if self.clip.as_ref().map_or(false, |c| c.source_kind.is_still()) {
                    // A still has no transport, trim or playhead.
                    ui.label(RichText::new(tr("Still photo — no timeline")).small().color(Color32::GRAY));
                    // The still's render thread normally lives as long as the
                    // clip; if it is gone (a decode error, a stop from another
                    // path) the photo would sit frozen — offer the restart the
                    // hidden Play button used to provide.
                    if !self.decoder_alive && self.load_pending.is_none() {
                        ui.separator();
                        if ui.small_button(tr("Restart preview")).clicked() {
                            self.spawn_decoder(true);
                        }
                    }
                    return;
                }
                let can_play = self.loaded_path.is_some();
                ui.add_enabled_ui(can_play, |ui| {
                    let label = tr(if self.playing { "Pause" } else { "▶ Play" });
                    if ui.button(label).clicked() { self.toggle_play_pause(); }
                    if ui.button(tr("■ Stop")).clicked() { self.stop_playback(); }
                    let has_frame = self.current_display.is_some();
                    if ui.add_enabled(has_frame, egui::Button::new(tr("📷 Snapshot")))
                        .on_hover_text(tr("Save this frame as a JPEG in the export folder (S)"))
                        .clicked()
                    {
                        self.request_snapshot();
                    }
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
                    if ui.button(tr("[ Mark In")).on_hover_text(tr("Set trim-in to current playhead (I)")).clicked() {
                        self.set_trim_in(Some(cur));
                    }
                    if ui.button(tr("Mark Out ]")).on_hover_text(tr("Set trim-out to current playhead (O)")).clicked() {
                        self.set_trim_out(Some(cur));
                    }
                    if ui.button(tr("Clear trim")).on_hover_text(tr("Remove both in/out points")).clicked() {
                        self.set_trim_in(None);
                        self.set_trim_out(None);
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Only show the hint when there's room. A right-to-left
                    // label in this shared row overflows LEFT when the window
                    // is narrow, drawing over the trim buttons — guard on the
                    // remaining width so it just hides instead of overlapping.
                    let open = if cfg!(target_os = "macos") { "⌘O" } else { "Ctrl+O" };
                    let hint = format!("Space play · I in · O out · {open} open");
                    if ui.available_width() > 320.0 {
                        ui.label(RichText::new(hint).small().color(Color32::DARK_GRAY));
                    }
                });
            });
        });

        // ─── Timeline scrubber ──────────────────────────────────
        // A still has no timeline — skip the scrubber strip entirely.
        let timeline_applies = !self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
        if timeline_applies {
            egui::TopBottomPanel::bottom("timeline").exact_height(34.0).show(ctx, |ui| {
                self.draw_timeline(ui);
            });
        }

        // ─── Export bar (output settings + run the queue) ───────
        self.draw_export_bar(ctx);

        // ─── Main preview area ──────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            // Copy out the display handle so we can borrow `self` mutably for
            // the zoom/pan state below. When zoomed + paused we show the
            // `DetailCache` still whenever it is for the CURRENT FRAME
            // (timestamp match). A stale-SETTINGS still (mid slider-drag,
            // waiting out the 120 ms re-render throttle) stays on screen —
            // dropping to the low-res live preview during adjustments was
            // exactly the flicker the user asked to remove; the throttled
            // re-render refreshes it in place. Only a FRAME change (scrub)
            // falls back to the live preview while the new still decodes.
            let cur_ts = self.current_display.as_ref()
                .map(|d| d.timestamp_s).unwrap_or(0.0);
            let use_full_res = self.wants_full_res_still() && !self.playing
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

                // Still photo: does the magnifier (last frame's zoom) need
                // more pixels per eye than the binned working size offers?
                // zoom × on-screen eye size in physical pixels vs the binned
                // half width (7760 / 2 on the X6). The reframed view aims
                // with the FOV instead and always gets the native source.
                if let Some(c) = self.clip.as_ref().filter(|c| c.source_kind.is_still()) {
                    let eye_disp = if self.settings.preview_mode == crate::decoder::PreviewMode::Sbs { w * 0.5 } else { w };
                    let eye_phys = eye_disp * ui.ctx().pixels_per_point();
                    let binned_eye = (c.fisheye_eye_w / 2).max(1) as f32;
                    self.still_native_wanted = self.preview_zoom * eye_phys > binned_eye * 0.95;
                }

                let zoom = &mut self.preview_zoom;
                let center = &mut self.preview_center;
                let reframe = matches!(self.settings.fisheye_output_mode,
                    crate::decoder::FisheyeOutputMode::Reframe);

                if reframe {
                    // Reframed view: the gestures aim the viewport itself —
                    // drag pans, scroll / pinch zooms (the FOV), double-click
                    // recenters. The alignment magnifier stays at fit.
                    *zoom = 1.0;
                    *center = egui::vec2(0.5, 0.5);
                    let s = &mut self.settings;
                    if resp.hovered() {
                        let zd = ui.input(|i| i.zoom_delta());
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        let factor = if (zd - 1.0).abs() > 1e-4 { zd } else { 1.0 + scroll * 0.002 };
                        if (factor - 1.0).abs() > 1e-6 {
                            s.reframe_hfov_deg = (s.reframe_hfov_deg / factor).clamp(30.0, 150.0);
                        }
                    }
                    if resp.dragged() {
                        let d = resp.drag_delta();
                        // Displayed width of ONE eye: the SBS preview shows two
                        // side by side, the other modes one eye full-width.
                        let eye_disp_w = if s.preview_mode == crate::decoder::PreviewMode::Sbs {
                            w * 0.5
                        } else {
                            w
                        };
                        // Angular size of one displayed pixel at the view centre
                        // (dθ/dr = 1 there for every k), so the content follows
                        // the cursor.
                        let k = vr180_pipeline::gpu::reframe_k(s.reframe_defish);
                        let edge_x = vr180_pipeline::gpu::reframe_edge_x(s.reframe_hfov_deg, k);
                        let deg_per_px = (2.0 * edge_x / eye_disp_w.max(1.0)).to_degrees();
                        let mut yaw = s.reframe_yaw_deg - d.x * deg_per_px;
                        if yaw > 180.0 { yaw -= 360.0; } else if yaw < -180.0 { yaw += 360.0; }
                        s.reframe_yaw_deg = yaw;
                        s.reframe_pitch_deg = (s.reframe_pitch_deg + d.y * deg_per_px).clamp(-90.0, 90.0);
                    }
                    if resp.double_clicked() {
                        s.reframe_yaw_deg = 0.0;
                        s.reframe_pitch_deg = 0.0;
                    }
                }

                // Scroll → zoom toward the cursor.
                if resp.hovered() && !reframe {
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
                if resp.dragged() && *zoom > 1.0 && !reframe {
                    let d = resp.drag_delta();
                    center.x -= d.x / (w * *zoom);
                    center.y -= d.y / (h * *zoom);
                }
                // Double-click → reset to fit.
                if resp.double_clicked() && !reframe {
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
                    if reframe {
                        let s = &self.settings;
                        format!("{} {:.1}°  ·  {} {:+.1}°  ·  {} {:+.1}°   ({})",
                            tr("FOV"), s.reframe_hfov_deg, tr("pan"), s.reframe_yaw_deg,
                            tr("tilt"), s.reframe_pitch_deg,
                            tr("drag = pan, scroll / pinch = zoom, dbl-click = recenter"))
                    } else {
                        format!("{:.0}%  (scroll = zoom, drag = pan, dbl-click = reset)",
                            *zoom * 100.0)
                    },
                    egui::FontId::proportional(12.0),
                    Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                );
            } else if let Some(pending) = &self.load_pending {
                // Async load in flight — keep the GUI responsive while the
                // worker reads the source (slow/cold network shares).
                let name = pending.path.file_name()
                    .and_then(|s| s.to_str()).unwrap_or("clip").to_string();
                let secs = pending.started.elapsed().as_secs();
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        ui.add_space(8.0);
                        ui.label(RichText::new(format!("{} {}…", tr("Loading"), name))
                            .size(14.0).color(Color32::GRAY));
                        if secs >= 3 {
                            ui.add_space(4.0);
                            ui.label(RichText::new(tr(
                                "Reading from the source — network drives can take a moment."))
                                .size(12.0).color(Color32::DARK_GRAY));
                        }
                    });
                });
            } else if self.loaded_path.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new(tr("Click ▶ Play to start preview."))
                        .size(14.0).color(Color32::GRAY));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("🎬").size(48.0));
                        ui.add_space(8.0);
                        ui.label(RichText::new(tr("Drop a video file here, or click Load video."))
                            .size(13.0).color(Color32::GRAY));
                    });
                });
            }
        });

        // External 3D display (side-by-side monitor / AR glasses) — an
        // immediate viewport, drawn every frame while enabled.
        self.draw_3d_display(ctx);

        // Push slider edits to the running decoder.
        self.maybe_push_settings();
        // Remember settings across launches (debounced; flushes on close).
        self.persist_settings(ctx);

        // Keep the UI awake whenever a decoder is running so the next
        // frame draws without waiting for a user input event. The worker
        // repaints the window itself when it hands a frame over
        // (`DecoderControl::wake_ui`), so this is only a fallback — and it
        // must stay BELOW the display rate: a UI that repaints faster than
        // vsync waits in `nextDrawable` holding wgpu's fence lock, and
        // every worker `queue.submit` then stalls for a whole refresh.
        // - playing: 10 Hz fallback (frames arrive at the preview rate)
        // - paused but alive: 20 Hz, enough to display slider-driven
        //   re-renders the decoder thread pushes into frame_rx.
        if self.decoder_alive {
            let delay_ms = if self.playing { 100 } else { 50 };
            ctx.request_repaint_after(std::time::Duration::from_millis(delay_ms));
        }
    }
}

impl App {
    /// Source panel = the CLIP LIST (click to select → preview + sidebar
    /// edit that clip's settings) + the active clip's metadata below.
    /// Turn the external 3D display output on: pick the screen, force the
    /// plain SBS compose, and render each eye at half that screen's width.
    fn enable_3d_display(&mut self) {
        use crate::decoder::PreviewMode;
        let displays = crate::displays::all_displays();
        let target = crate::displays::pick_3d_display(&displays);
        let d = &mut self.three_d;
        d.displays = displays;
        d.target = target;
        d.native_fullscreen = target.is_some() && crate::displays::displays_have_separate_spaces();
        d.fullscreen_sent = false;
        d.fullscreen = false;
        d.restore_preview_mode = None;
        d.restore_preview_w = None;
        // Stereo needs the plain L|R compose — anaglyph / overlay / single
        // eye are alignment aids for the main window only.
        if self.settings.preview_mode != PreviewMode::Sbs {
            d.restore_preview_mode = Some(self.settings.preview_mode);
            self.settings.preview_mode = PreviewMode::Sbs;
        }
        // Render each eye at half the screen width so the output is 1:1
        // (a 768-px preview eye stretched to 1920 px looks soft). The eye
        // size binds at decoder start, so this reloads the clip.
        let mut reload = false;
        if let Some(t) = target {
            let want = (d.displays[t].pixel_w / 2).clamp(512, 2048);
            if self.settings.preview_eye_w < want {
                d.restore_preview_w = Some(self.settings.preview_eye_w);
                self.settings.preview_eye_w = want;
                reload = true;
            }
        }
        d.enabled = true;
        match target {
            Some(t) => tracing::info!("3D display: on → {} ({}) at {:?}, {}×{} pt, {}",
                d.displays[t].label(), d.displays[t].name, d.displays[t].origin,
                d.displays[t].size.x, d.displays[t].size.y,
                if d.native_fullscreen { "native fullscreen once placed (separate Spaces)" } else { "borderless at bounds" }),
            None => tracing::info!("3D display: on — no 3840×1080-class screen connected, windowed fallback"),
        }
        if reload {
            if let Some(p) = self.loaded_path.clone() { self.load_file_inner(p, true); }
        }
    }

    fn disable_3d_display(&mut self) {
        let d = &mut self.three_d;
        d.enabled = false;
        if let Some(m) = d.restore_preview_mode.take() { self.settings.preview_mode = m; }
        let mut reload = false;
        if let Some(w) = d.restore_preview_w.take() {
            self.settings.preview_eye_w = w;
            reload = true;
        }
        tracing::info!("3D display: off");
        if reload {
            if let Some(p) = self.loaded_path.clone() { self.load_file_inner(p, true); }
        }
    }

    /// The external 3D display viewport: a chrome-less window on the chosen
    /// screen (or a movable fallback) painting the SBS preview texture with
    /// the left eye fitted into the left half and the right eye into the
    /// right half. Immediate viewport — it renders as part of this frame, so
    /// it follows every preview update; not calling it closes the window.
    fn draw_3d_display(&mut self, ctx: &egui::Context) {
        if !self.three_d.enabled { return; }
        let id = egui::ViewportId::from_hash_of("vr180_3d_display");
        let target = self.three_d.target.and_then(|t| self.three_d.displays.get(t).cloned());
        let mut builder = egui::ViewportBuilder::default()
            .with_title(format!("VR180 Silver Bullet — {}", tr("3D display")));
        let want_native_fs = self.three_d.native_fullscreen && !self.three_d.fullscreen_sent;
        builder = match &target {
            // Always created as a borderless window on the screen's exact
            // bounds. macOS with "Displays have separate Spaces" ON (the
            // default): that display carries its own menu bar and the window
            // gets pushed under it, so once it is confirmed on the target
            // screen we ask for native fullscreen (below) — which macOS
            // applies to the screen the window is on. Requesting fullscreen
            // at creation fullscreened the MAIN screen instead. With the
            // setting OFF the borderless window is the final state: native
            // fullscreen there makes its own Space and the desktop flips.
            Some(d) => builder
                .with_position(d.origin)
                .with_inner_size(d.size)
                .with_decorations(false)
                .with_resizable(false),
            None => builder.with_inner_size([1280.0, 360.0]),
        };
        let frame = self.current_display.as_ref().map(|d| (d.egui_id, d.width, d.height));
        let windowed = target.is_none();
        let fullscreen_now = self.three_d.fullscreen;
        let mut close = false;
        let mut toggle_fullscreen = false;
        let mut go_native_fs = false;
        ctx.show_viewport_immediate(id, builder, |ctx, _class| {
            // Step two of the separate-Spaces path: the window exists and
            // its centre is on the target screen → native fullscreen there.
            if want_native_fs {
                if let Some(d) = &target {
                    let on_target = ctx.input(|i| i.viewport().outer_rect)
                        .map(|r| egui::Rect::from_min_size(d.origin, d.size).contains(r.center()))
                        .unwrap_or(false);
                    if on_target { go_native_fs = true; }
                }
            }
            // Windows, mixed per-monitor DPI: the builder's logical position
            // is resolved against the wrong monitor's scale at creation (the
            // window landed on the MAIN screen when the primary ran at 200%),
            // so correct the placement afterwards against the window's OWN
            // reported scale, in physical pixels. Converges in ≤2 frames:
            // the first move lands the window on the target screen (its
            // scale then becomes the target's), the next pass fixes the
            // size under that scale and then matches exactly. No-op once
            // the physical rect is right; macOS keeps the builder placement
            // (CG points are the native space there — never entered).
            #[cfg(target_os = "windows")]
            if let Some(d) = &target {
                let (cur, ppp) = ctx.input(|i| {
                    let vp = i.viewport();
                    let ppp = vp.native_pixels_per_point.unwrap_or_else(|| i.pixels_per_point());
                    (vp.outer_rect.zip(vp.inner_rect), ppp)
                });
                if let Some((outer, inner)) = cur {
                    let pos_ok = (outer.min.x * ppp - d.origin_px.x).abs() <= 1.5
                        && (outer.min.y * ppp - d.origin_px.y).abs() <= 1.5;
                    let size_ok = (inner.width() * ppp - d.pixel_w as f32).abs() <= 1.5
                        && (inner.height() * ppp - d.pixel_h as f32).abs() <= 1.5;
                    if !pos_ok || !size_ok {
                        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(
                            egui::pos2(d.origin_px.x / ppp, d.origin_px.y / ppp)));
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                            egui::vec2(d.pixel_w as f32 / ppp, d.pixel_h as f32 / ppp)));
                        ctx.request_repaint();
                    }
                }
            }
            if ctx.input(|i| i.viewport().close_requested()) { close = true; }
            ctx.input(|i| {
                if i.key_pressed(egui::Key::Escape) { close = true; }
                // F = native fullscreen, windowed fallback only (the
                // auto-placed output already covers its screen).
                if windowed && i.key_pressed(egui::Key::F) && i.modifiers.is_none() { toggle_fullscreen = true; }
            });
            if fullscreen_now || !windowed { ctx.set_cursor_icon(egui::CursorIcon::None); }
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(Color32::BLACK))
                .show(ctx, |ui| {
                    let full = ui.max_rect();
                    let half = egui::vec2(full.width() * 0.5, full.height());
                    let halves = [
                        egui::Rect::from_min_size(full.min, half),
                        egui::Rect::from_min_size(egui::pos2(full.min.x + half.x, full.min.y), half),
                    ];
                    match frame {
                        Some((tex, w, h)) if w > 0 && h > 0 => {
                            // The SBS texture is L|R: each eye is half its
                            // width. Fit each eye into its half of the screen
                            // (exact fill for a 16:9 reframed eye on a
                            // 3840×1080 screen; letterboxed otherwise).
                            let eye_aspect = (w as f32 * 0.5) / h as f32;
                            for (i, r) in halves.iter().enumerate() {
                                let fit = fit_aspect(*r, eye_aspect);
                                let u0 = i as f32 * 0.5;
                                let uv = egui::Rect::from_min_max(
                                    egui::pos2(u0, 0.0), egui::pos2(u0 + 0.5, 1.0));
                                let sized = egui::load::SizedTexture::new(tex, fit.size());
                                egui::Image::new(sized).uv(uv).paint_at(ui, fit);
                            }
                        }
                        _ => {
                            ui.centered_and_justified(|ui| {
                                ui.label(RichText::new(tr("3D display — play a clip to see it here"))
                                    .size(16.0).color(Color32::GRAY));
                            });
                        }
                    }
                    if windowed && !fullscreen_now {
                        ui.painter().text(
                            full.left_top() + egui::vec2(10.0, 8.0), egui::Align2::LEFT_TOP,
                            tr("Drag this window onto the 3D screen · F = fullscreen · Esc = close"),
                            egui::FontId::proportional(13.0), Color32::from_rgb(200, 200, 200));
                    }
                });
        });
        if go_native_fs {
            self.three_d.fullscreen_sent = true;
            self.three_d.fullscreen = true;
            tracing::info!("3D display: window placed on the target screen — going native fullscreen there");
            ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Fullscreen(true));
        }
        if toggle_fullscreen {
            self.three_d.fullscreen = !self.three_d.fullscreen;
            ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Fullscreen(self.three_d.fullscreen));
        }
        if close { self.disable_3d_display(); }
    }

    /// Per-eye output size the export would use right now — the same
    /// derivation as the export config builder: source eye dims (EAC: the
    /// native cross square), 4096² for the 8K target, or the reframed
    /// per-eye size. Snapshots render the paused frame at exactly this.
    fn export_eye_dims(&self) -> (u32, u32) {
        let Some(clip) = self.clip.as_ref() else { return (0, 0); };
        let kind = clip.source_kind;
        let (mut ew, mut eh) = if clip.fisheye_eye_w > 0 {
            (clip.fisheye_eye_w, clip.fisheye_eye_h)
        } else if kind.is_eac() {
            let cross_w = 2 * (clip.width.saturating_sub(1920) / 4) + 1920;
            (cross_w.max(512), cross_w.max(512))
        } else {
            (clip.width.max(512), clip.height.max(512))
        };
        if self.export_opts.resolution == ExportResolution::R8k {
            ew = 4096;
            eh = 4096;
        }
        match self.settings.fisheye_output_mode {
            crate::decoder::FisheyeOutputMode::Reframe =>
                self.export_opts.reframe_size.eye_dims(self.settings.reframe_aspect),
            crate::decoder::FisheyeOutputMode::Fisheye => { let side = ew.min(eh); (side, side) }
            crate::decoder::FisheyeOutputMode::HalfEquirect => (ew, eh),
        }
    }

    /// Snapshot (📷 / `S`): save the frame on screen at the EXPORT's
    /// resolution — native / 8K for VR180, the chosen per-eye size for
    /// Reframed — as a 92 % JPEG in the export output folder, named
    /// `<source stem>_frame<N>.jpg`. The full-size render comes from the
    /// zoom-still path, which needs the decoder paused on the frame, so a
    /// snapshot mid-playback pauses there first; `poll_snapshot` writes the
    /// file when the still lands (or falls back to the preview texture if it
    /// never does).
    fn request_snapshot(&mut self) {
        if self.current_display.is_none() || self.loaded_path.is_none() {
            self.snapshot_status = Some((tr("Nothing to snapshot yet").to_string(), std::time::Instant::now()));
            return;
        }
        // A still's full-size render IS the export; the zoom-still path that
        // serves snapshots is video-only (no DetailCache for a photo), so a
        // request here would wait out the 15 s timeout and save the
        // working-res preview.
        if self.clip.as_ref().map_or(false, |c| c.source_kind.is_still()) {
            self.snapshot_status = Some((tr("Still photo — use Export for the full-size file").to_string(), std::time::Instant::now()));
            return;
        }
        if self.playing { self.toggle_play_pause(); }
        let out = self.export_eye_dims();
        if let Some(c) = &self.control {
            // EAC: the decoder renders the still — ask for the export size and
            // bump the generation so it re-renders even on an unchanged frame.
            *c.detail_out.lock() = Some(out);
            c.settings_generation.fetch_add(1, Ordering::SeqCst);
        }
        self.snapshot_pending = Some(SnapshotReq { out, since: std::time::Instant::now() });
        self.snapshot_status = Some((format!("{}…", tr("Rendering snapshot")), std::time::Instant::now()));
    }

    /// Write the pending snapshot once the export-size still of the paused
    /// frame is showing; after 15 s without one, save the preview texture
    /// instead and say so.
    fn poll_snapshot(&mut self, ctx: &egui::Context) {
        let Some((out, since)) = self.snapshot_pending.as_ref().map(|r| (r.out, r.since)) else { return; };
        let cur = self.current_display.as_ref().map(|d| (d.timestamp_s, d.frame_idx));
        let ready = match (&self.full_res_display, cur) {
            (Some(f), Some((ts, _))) =>
                (f.timestamp_s - ts).abs() < 0.02 && f.width == out.0 * 2 && f.height == out.1,
            _ => false,
        };
        let timed_out = since.elapsed().as_secs_f32() > 15.0;
        if !ready && !timed_out {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            return;
        }
        self.snapshot_pending = None;
        if let Some(c) = &self.control {
            *c.detail_out.lock() = None;
            c.settings_generation.fetch_add(1, Ordering::SeqCst);
        }
        let frame = if ready { self.full_res_display.as_ref() } else { self.current_display.as_ref() }
            .map(|d| (d.texture.clone(), d.width, d.height));
        let frame_idx = cur.map(|c| c.1).unwrap_or(0);
        if let Some((tex, w, h)) = frame {
            self.write_snapshot(tex, w, h, frame_idx, !ready);
        }
    }

    fn write_snapshot(&mut self, tex: Arc<wgpu::Texture>, w: u32, h: u32, frame_idx: u32, preview_fallback: bool) {
        let Some(src) = self.loaded_path.clone() else { return; };
        let dir = self.batch_out_dir.clone()
            .or_else(|| src.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let stem = src.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "frame".into());
        let mut path = dir.join(format!("{stem}_frame{frame_idx}.jpg"));
        let mut n = 2;
        while path.exists() {
            path = dir.join(format!("{stem}_frame{frame_idx}-{n}.jpg"));
            n += 1;
        }
        let result = (|| -> anyhow::Result<()> {
            let rgb = self.pipeline.read_texture_rgb8(&tex, w, h)?;
            std::fs::create_dir_all(&dir)?;
            let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 92);
            enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
            Ok(())
        })();
        let msg = match result {
            Ok(()) => {
                tracing::info!("snapshot: {} ({w}×{h}, preview_fallback={preview_fallback})", path.display());
                let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                if preview_fallback {
                    format!("{} {name} ({w}×{h}, {})", tr("Saved"), tr("preview size — full-size still never arrived"))
                } else {
                    format!("{} {name} ({w}×{h})", tr("Saved"))
                }
            }
            Err(e) => {
                tracing::warn!("snapshot failed: {e:#}");
                format!("{}: {e}", tr("Snapshot failed"))
            }
        };
        self.snapshot_status = Some((msg, std::time::Instant::now()));
    }

    fn draw_source_panel(&mut self, ui: &mut egui::Ui) {
        let mut select_idx: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let mut requeue_idx: Option<usize> = None;
        let mut toggle_select: Option<usize> = None;
        let mut do_apply_all = false;

        if !self.batch.is_empty() {
            for (i, item) in self.batch.iter().enumerate() {
                ui.horizontal(|ui| {
                    // Multi-select checkbox → drives "Export selected".
                    let mut sel = item.selected;
                    if ui.checkbox(&mut sel, "")
                        .on_hover_text(tr("Tick to include in \"Export selected\""))
                        .changed()
                    {
                        toggle_select = Some(i);
                    }
                    let active = self.active_clip == Some(i);
                    let name = item.path.file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let label = if active {
                        RichText::new(format!("▶ {name}")).strong()
                            .color(Color32::from_rgb(255, 200, 120))
                    } else {
                        RichText::new(format!("   {name}"))
                    };
                    if ui.add(egui::Label::new(label).sense(egui::Sense::click()))
                        .on_hover_text(tr("Click to make this the active clip"))
                        .clicked() && !active
                    {
                        select_idx = Some(i);
                    }
                    // Status chip (only interesting once exports ran).
                    if item.status != BatchStatus::Idle {
                        let col = match &item.status {
                            BatchStatus::Done      => Color32::from_rgb(120, 200, 120),
                            BatchStatus::Failed(_) => Color32::from_rgb(230, 110, 110),
                            BatchStatus::Running   => Color32::from_rgb(120, 180, 255),
                            _ => Color32::GRAY,
                        };
                        ui.label(RichText::new(item.status.label()).small().color(col));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let removable = item.status != BatchStatus::Running;
                        ui.add_enabled_ui(removable, |ui| {
                            if ui.small_button("X").on_hover_text(tr("Remove from list")).clicked() {
                                remove_idx = Some(i);
                            }
                        });
                        // Re-arm a finished clip so the next export includes it.
                        if matches!(item.status,
                            BatchStatus::Done | BatchStatus::Failed(_) | BatchStatus::Skipped)
                            && ui.small_button("↻").on_hover_text(tr("Re-arm for export")).clicked()
                        {
                            requeue_idx = Some(i);
                        }
                    });
                });
            }
            ui.add_space(2.0);
        }
        // Deferred: the picker must not run inside the egui frame (see
        // the `dialog_tx` field comment).
        let mut add_clips = false;
        ui.horizontal(|ui| {
            if ui.small_button(tr("+ Add clips…")).clicked() {
                add_clips = true;
            }
            let n_sel = self.batch.iter().filter(|b| b.selected).count();
            if self.batch.len() > 1 {
                ui.label(RichText::new(if n_sel > 0 {
                    format!("{} clips · {n_sel} selected", self.batch.len())
                } else {
                    format!("{} clips", self.batch.len())
                }).small().color(Color32::GRAY));
                if ui.small_button(tr("Apply settings to all"))
                    .on_hover_text(tr("Copy the active clip's settings to every clip OF THE SAME TYPE \
                                    (OSV→OSV, .360→.360, …) — other camera types keep their own \
                                    settings. Per-clip trim & RS mode are kept."))
                    .clicked()
                {
                    do_apply_all = true;
                }
            }
        });
        if add_clips {
            let ctx = ui.ctx().clone();
            self.spawn_dialog(&ctx, || {
                Self::video_file_dialog().pick_files().map(DialogResult::Videos)
            });
        }

        if let Some(i) = select_idx { self.select_clip(i); }
        if let Some(i) = remove_idx { self.remove_clip(i); }
        if let Some(i) = toggle_select {
            if let Some(item) = self.batch.get_mut(i) { item.selected = !item.selected; }
        }
        if let Some(i) = requeue_idx {
            if let Some(item) = self.batch.get_mut(i) {
                // Mid-run a re-armed clip joins the current run; otherwise it
                // waits for the next Export.
                item.status = if self.batch_running {
                    BatchStatus::Queued
                } else {
                    BatchStatus::Idle
                };
            }
        }
        if do_apply_all {
            // SAME-KIND ONLY — never overwrite a different camera type's
            // settings (OSV stab/calib means nothing to a .360 clip, etc.).
            self.sync_active_clip_settings();
            if let Some(kind) = self.active_clip.and_then(|a| self.batch.get(a)).map(|b| b.source_kind) {
                for i in 0..self.batch.len() {
                    if self.batch[i].source_kind == kind {
                        self.apply_current_settings_to_item(i);
                    }
                }
            }
        }

        if !self.batch.is_empty() || self.loaded_path.is_some() {
            ui.separator();
        }
        self.draw_source_info(ui);

        // Generic side-by-side .mp4/.mov: raw dual fisheye (needs the lens
        // dewarp) or an already-dewarped VR180 half-equirect? Off by default
        // — most SBS files brought in are finished VR180 exports, and
        // dewarping those a second time distorts them (worst in Reframed).
        // .360 / .osv / .insv never show this: their lens model is in-file.
        if matches!(self.clip.as_ref().map(|c| c.source_kind),
                    Some(vr180_pipeline::SourceKind::SbsFisheye))
        {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(tr("Input"));
                ui.checkbox(&mut self.settings.sbs_dewarp_fisheye, tr("Dewarp fisheye input"))
                    .on_hover_text(tr("On: each half is a raw fisheye image — dewarp it with the Fisheye lens settings. Off (default): the file is already VR180 side-by-side half-equirect and is used as-is."));
            });
        }
    }

    /// Remove a clip from the list, keeping `active_clip` /
    /// `batch_current` indices coherent. If the removed entry is the
    /// clip currently loaded (or still loading), the preview is unloaded
    /// with it: the neighbouring entry becomes active when one remains,
    /// otherwise the app returns to the empty "no clip loaded" state.
    fn remove_clip(&mut self, idx: usize) {
        if idx >= self.batch.len() { return; }
        if self.batch[idx].status == BatchStatus::Running { return; }
        let removed_path = self.batch[idx].path.clone();
        let was_loaded = self.loaded_path.as_deref() == Some(removed_path.as_path())
            || self.load_pending.as_ref().map_or(false, |p| p.path == removed_path);
        self.batch.remove(idx);
        self.active_clip = match self.active_clip {
            Some(a) if a == idx => None,
            Some(a) if a > idx  => Some(a - 1),
            other => other,
        };
        self.batch_current = match self.batch_current {
            Some(c) if c > idx => Some(c - 1),
            // c == idx impossible (Running rows can't be removed).
            other => other,
        };
        if was_loaded {
            self.unload_clip(false);
            self.active_clip = None;
            if !self.batch.is_empty() {
                // Keep something on screen: activate the entry that slid
                // into the removed slot (or the new last one).
                let next = idx.min(self.batch.len() - 1);
                self.select_clip(next);
            }
        }
    }

    /// The shared open-dialog filter for every "add video" entry point.
    fn video_file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new()
            .add_filter(
                "All supported video",
                &["360", "osv", "OSV", "insv", "INSV", "braw", "BRAW",
                  "mp4", "MP4", "mov", "MOV", "dng", "DNG"],
            )
            .add_filter("Camera still (.dng: Insta360 X6, Kandao QooCam 3 Ultra)", &["dng", "DNG"])
            .add_filter("DJI Osmo 360 (.osv)", &["osv", "OSV"])
            .add_filter("Insta360 (.insv)", &["insv", "INSV"])
            .add_filter("Kandao QooCam 3 Ultra (.mp4)", &["mp4", "MP4"])
            .add_filter("Blackmagic RAW (.braw)", &["braw", "BRAW"])
            .add_filter("Side-by-side fisheye (.mp4 / .mov)",
                &["mp4", "MP4", "mov", "MOV"])
    }

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
                    ui.label(RichText::new(tr("Stream")).color(Color32::GRAY).small());
                    ui.label(format!("{} × {}", c.width, c.height));
                });
                if c.source_kind.is_still() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("Still")).color(Color32::GRAY).small());
                        ui.label(if c.source_kind.is_kandao() { tr("single photo, 10-bit raw") } else { tr("single photo, 14-bit raw") });
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("FPS")).color(Color32::GRAY).small());
                        ui.label(format!("{:.3}", c.fps));
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("Duration")).color(Color32::GRAY).small());
                        ui.label(format!("{:.2}s ({} frames)", c.duration_sec, c.frame_count));
                    });
                }
                if c.source_kind.is_eac() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("EAC tile")).color(Color32::GRAY).small());
                        ui.label(format!("{} px", c.eac_tile_w));
                    });
                } else if c.source_kind.is_fisheye() && c.fisheye_eye_w > 0 {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("Eye dims")).color(Color32::GRAY).small());
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
                ui.label(RichText::new(tr("No file loaded.")).color(Color32::GRAY));
            }
        }
    }

    fn draw_stab_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.checkbox(&mut s.stabilize, tr("Enable stabilization"));
        ui.add_enabled_ui(s.stabilize, |ui| {
            ui.horizontal(|ui| {
                ui.label(tr("CORI source"));
                egui::ComboBox::from_id_source("cori_src")
                    .selected_text(s.cori_source.as_str())
                    .show_ui(ui, |ui| {
                        use crate::decoder::CoriSource as C;
                        ui.selectable_value(&mut s.cori_source, C::Direct, tr("direct"));
                        ui.selectable_value(&mut s.cori_source, C::Vqf, tr("vqf"));
                        ui.selectable_value(&mut s.cori_source, C::Auto, tr("auto"));
                    });
            });
            // Camera lock owns the sharp frame-0 lock (full counter-rotation,
            // no cap); Smooth / Max corr / Response are the soft-stab controls
            // and gray out under it (mirrors the OSV/INSV panel).
            ui.checkbox(&mut s.camera_lock, tr("Camera lock"))
                .on_hover_text(tr("Lock the view to the first frame; smoothing and max correction are ignored."));
            let cam_lock = s.camera_lock;
            ui.add_enabled_ui(!cam_lock, |ui| {
                ui.add(egui::Slider::new(&mut s.smooth_ms, 0.0..=3000.0).text(tr("Smooth (ms)")));
            });
            ui.add_enabled_ui(!cam_lock, |ui| {
                ui.add(egui::Slider::new(&mut s.max_corr_deg, 0.0..=45.0).text(tr("Max corr (°)")));
            });
            ui.add_enabled_ui(!cam_lock, |ui| {
                ui.add(egui::Slider::new(&mut s.gyro_responsiveness, 0.2..=3.0)
                    .text(tr("Response")))
                    .on_hover_text(tr("Velocity response curve: <1 follows motion early, \
                        1 linear, >1 holds longer then catches up (cinematic lag)."));
            });
        });
        ui.add_space(6.0);
        ui.label(RichText::new(tr(
            "Settings apply live during playback — the decoder rebuilds \
             per-eye bundles on the next iteration (small stutter ~300 ms \
             on long clips)."
        )).small().color(Color32::GRAY));
    }

    /// Audio-track choice for GoPro `.360` (it carries both a stereo AAC
    /// and a 4-channel ambisonic track). Lives in its own controls section
    /// (not the export window) so it's visible while reviewing the clip.
    /// Applies to export only — the live preview always plays stereo.
    fn draw_audio_panel(&mut self, ui: &mut egui::Ui) {
        use crate::decoder::AudioFormat as AF;
        let x6 = self.clip.as_ref().map_or(false, |c| c.foa_audio);
        let quad = self.clip.as_ref().map_or(false, |c| c.quad_audio);
        if x6 {
            ui.label(RichText::new(tr(
                "Recorded in 360° Audio: first-order ambisonics (AmbiX). A plain copy plays \
                 lopsided, so Stereo decodes a front-facing pair from it."
            )).small().color(Color32::GRAY));
        } else if quad {
            ui.label(RichText::new(tr(
                "Four microphone capsules (front, left, back, right). \
                 The app encodes them to first-order ambisonics; Stereo is a front-facing pair from that."
            )).small().color(Color32::GRAY));
        }
        let s = &mut self.settings;
        // APAC is Apple's Positional Audio Codec, encoded by the macOS
        // `apac_encode` helper over AVFoundation — there is no Windows encoder,
        // and the export would only log "APAC is macOS-only" and fall back to
        // stereo. So the choice is not offered here, and a value persisted
        // before it was hidden (or carried over from a mac) is folded into
        // Ambisonic, the nearest spatial option Windows can actually write.
        #[cfg(not(target_os = "macos"))]
        if s.audio_format == AF::Apac {
            s.audio_format = AF::Ambisonic;
        }
        ui.horizontal(|ui| {
            ui.label(tr("Export track"));
            egui::ComboBox::from_id_source("audio_format")
                .selected_text(s.audio_format.as_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.audio_format, AF::Stereo, AF::Stereo.as_str());
                    ui.selectable_value(&mut s.audio_format, AF::Ambisonic, AF::Ambisonic.as_str());
                    #[cfg(target_os = "macos")]
                    ui.selectable_value(&mut s.audio_format, AF::Apac, AF::Apac.as_str());
                });
        });
        let hint = match s.audio_format {
            AF::Stereo if x6 => tr("Stereo AAC decoded from the ambisonics (virtual cardioids ±45°, camera front) — universal."),
            AF::Stereo if quad => tr("Stereo AAC from the capsule array (virtual cardioids ±45°, camera front) — universal."),
            AF::Stereo    => tr("Stereo AAC — universal compatibility."),
            AF::Ambisonic => tr("4-channel 1st-order ambisonic (AmbiX) + SA3D — \
                              YouTube VR / Quest head-track it."),
            AF::Apac      => tr("Apple Positional Audio — Vision Pro head-tracked \
                              spatial (macOS only; re-encodes the ambisonic)."),
        };
        ui.label(RichText::new(hint).small().color(Color32::GRAY));
    }

    fn draw_rs_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.checkbox(&mut s.rs_correct, tr("Enable RS correction"));
        ui.add_enabled_ui(s.rs_correct, |ui| {
            ui.horizontal(|ui| {
                ui.label(tr("Mode"));
                egui::ComboBox::from_id_source("rs_mode")
                    .selected_text(s.rs_mode.as_str())
                    .show_ui(ui, |ui| {
                        use crate::decoder::RsMode as R;
                        ui.selectable_value(&mut s.rs_mode, R::Firmware, tr("firmware"));
                        ui.selectable_value(&mut s.rs_mode, R::NoFirmware, tr("no-firmware"));
                    });
            });
            // SROT readout is fixed at the GoPro firmware value (15.224 ms),
            // re-seeded per clip — no manual override needed, slider hidden.
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
    /// Whether temporal noise reduction is available — a cached VideoToolbox
    /// capability probe (always false off macOS). Gates the NR slider so it
    /// never appears on hardware that can't run `VTTemporalNoiseFilter`.
    fn denoise_supported() -> bool {
        #[cfg(target_os = "macos")]
        {
            use std::sync::OnceLock;
            static CAP: OnceLock<bool> = OnceLock::new();
            *CAP.get_or_init(vr180_pipeline::vt_denoise::is_supported)
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn draw_color_panel(&mut self, ui: &mut egui::Ui) {
        // Source kind drives which built-in log→709 LUT is offered
        // (DJI D-LogM for OSV, GoPro GP-Log for `.360`) — never show the
        // wrong camera's curve.
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        // Deferred: the picker must not run inside the egui frame (see
        // the `dialog_tx` field comment). Also `s` borrows self below.
        let mut browse_lut = false;
        // Latest automatic eye-exposure measurement (X6 preview), for the readout.
        // Only the preview worker feeds it — an export stops playback, so the
        // panel then says the offset is applied rather than "measuring".
        let preview_running = self.decoder_alive;
        let eye_match_readout = if self.decoder_alive {
            self.control.as_ref().and_then(|c| *c.eye_match_readout.lock())
        } else { None };
        let is_still = matches!(kind, Some(k) if k.is_still());
        // Still + DNG: only what the DNG carries as metadata is offered.
        let dng_mode = is_still && self.export_opts.still_format == StillFormatChoice::LinearDng;
        // Blackmagic RAW: the SDK develops the sensor data — its colour
        // science, gamma/gamut, white balance, exposure/ISO and tone curve
        // replace the Tone and White balance stages for this camera (kept
        // neutral in `finish_load`), so those sections are not shown.
        let is_braw = matches!(kind, Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        let braw_info = if is_braw { self.clip.as_ref().and_then(|c| c.braw_color.clone()) } else { None };
        let braw_lut = if is_braw { self.clip.as_ref().and_then(|c| c.braw_lut.clone()) } else { None };
        let s = &mut self.settings;

        if let Some(info) = braw_info.as_deref() {
            draw_braw_color_section(ui, s, info, braw_lut.as_deref());
        } else if is_braw {
            ui.label(RichText::new(tr(
                "Blackmagic RAW colour controls need the clip probe's colour block — rebuild the helper (helpers/braw/build_braw.sh)."
            )).small().color(Color32::GRAY));
        } else {
            // CDL: lift / gamma / gain / shadow / highlight.
            ui.label(RichText::new(tr("Tone (CDL)")).strong());
            ui.add(egui::Slider::new(&mut s.lift, -1.0..=1.0)
                .text(tr("Lift")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.gamma, 0.2..=3.0)
                .text(tr("Gamma")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.gain, 0.2..=3.0)
                .text(tr("Gain")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.shadow, -1.0..=1.0)
                .text(tr("Shadow")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.highlight, -1.0..=1.0)
                .text(tr("Highlight")).fixed_decimals(2));
            if ui.button(tr("Reset tone")).clicked() {
                s.lift = 0.0; s.gamma = 1.0; s.gain = 1.0;
                s.shadow = 0.0; s.highlight = 0.0;
            }

            ui.separator();

            // White balance + saturation.
            ui.label(RichText::new(tr("White balance")).strong());
            ui.add(egui::Slider::new(&mut s.temperature, -1.0..=1.0)
                .text(tr("Temperature")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.tint, -1.0..=1.0)
                .text(tr("Tint")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.saturation, 0.0..=2.0)
                .text(tr("Saturation")).fixed_decimals(2));
            if ui.button(tr("Reset color")).clicked() {
                s.temperature = 0.0; s.tint = 0.0; s.saturation = 1.0;
            }
        }

        ui.separator();

        // ── Matching Eyes — inter-eye trim. Every knob here applies
        //    OPPOSITELY to each eye (left +, right −) to correct an inter-lens
        //    discrepancy without shifting the overall look: CT/Tint for color,
        //    Exposure (in stops) for brightness. ↑/↓ = 0.01. ──
        ui.label(RichText::new(tr("Matching Eyes")).strong());
        ui.label(RichText::new(tr(
            "Fixes a difference between the two lenses — CT, Tint and Exposure apply oppositely to each eye."
        )).small().color(Color32::GRAY));
        fine_slider(ui, 1.0, &mut s.eye_match_ct, -1.0..=1.0, tr("Eye CT (±)"), 2, 1.0, 0.01);
        fine_slider(ui, 1.0, &mut s.eye_match_tint, -1.0..=1.0, tr("Eye Tint (±)"), 2, 1.0, 0.01);
        fine_slider(ui, 1.0, &mut s.eye_match_exposure, -0.5..=0.5, tr("Eye Exposure (±EV)"), 2, 1.0, 0.01);
        if ui.button(tr("Reset matching")).clicked() {
            s.eye_match_ct = 0.0; s.eye_match_tint = 0.0; s.eye_match_exposure = 0.0;
        }
        // ── Automatic per-frame eye-exposure matching — X6 only. Its two
        //    sensors auto-expose independently (shutter and gain), so the eyes
        //    differ by up to ~1 stop and the difference moves with the AE.
        //    Measured on the GPU every frame, cancelled as a log-domain code
        //    offset before the LUT; the static trim above still composes. ──
        if matches!(kind, Some(vr180_pipeline::SourceKind::Insta360Insv)) {
            use vr180_pipeline::gpu::EyeMatchMode;
            ui.add_space(4.0);
            ui.checkbox(&mut s.auto_eye_match, tr("Auto-match eye exposure (per frame)"))
                .on_hover_text(tr("The X6's two sensors auto-expose independently; this measures the brightness difference between the eyes every frame and cancels it before the LUT."));
            if s.auto_eye_match {
                let mode_label = |m: EyeMatchMode| match m {
                    EyeMatchMode::Average => tr("Meet in the middle"),
                    EyeMatchMode::ToLeft  => tr("Match to left eye"),
                    EyeMatchMode::ToRight => tr("Match to right eye"),
                };
                ui.horizontal(|ui| {
                    ui.label(tr("Reference"));
                    egui::ComboBox::from_id_source("auto_eye_match_mode")
                        .selected_text(mode_label(s.auto_eye_match_mode))
                        .show_ui(ui, |ui| {
                            for m in [EyeMatchMode::Average, EyeMatchMode::ToLeft, EyeMatchMode::ToRight] {
                                ui.selectable_value(&mut s.auto_eye_match_mode, m, mode_label(m));
                            }
                        });
                });
                match eye_match_readout {
                    Some(r) if r.valid != 0 => {
                        let ev = vr180_pipeline::gpu::ILOG_CODE_PER_STOP;
                        ui.label(RichText::new(format!(
                            "{}: {:+.2} EV · {}: L {:+.2} / R {:+.2} EV",
                            tr("Measured L−R"), r.d / ev, tr("applied"), r.off_l / ev, r.off_r / ev,
                        )).small().color(Color32::GRAY));
                    }
                    Some(r) if r.reason != 0 => {
                        ui.label(RichText::new(tr(
                            "No usable content in the frame (clipped or too dark) — no correction applied."
                        )).small().color(Color32::GRAY));
                    }
                    _ if !preview_running => {
                        ui.label(RichText::new(tr(
                            "Applied per frame — the live readout shows while previewing."
                        )).small().color(Color32::GRAY));
                    }
                    _ => {
                        ui.label(RichText::new(tr("Measuring…")).small().color(Color32::GRAY));
                    }
                }
            }
        }

        ui.separator();

        if dng_mode {
            // Linear DNG output carries the grade as metadata; a 3D LUT,
            // sharpening and automatic eye matching have no DNG equivalent
            // (`still_export::grade_metadata`), and the preview plan drops
            // them too (`Settings::still_dng_mode`) — so they are not offered.
            ui.label(RichText::new(tr(
                "DNG output: the 3D LUT, sharpening and automatic eye matching cannot travel as DNG metadata — hidden, and not applied to the preview."
            )).small().color(Color32::GRAY));
        } else {
            // 3D LUT.
            ui.label(RichText::new(tr("3D LUT (.cube)")).strong());
            ui.horizontal(|ui| {
                let mut label = if s.lut_path.is_empty() {
                    "(none)".to_string()
                } else if s.lut_path == crate::decoder::BUILTIN_OSMO_LUT_PATH {
                    crate::decoder::BUILTIN_OSMO_LUT_NAME.to_string()
                } else if s.lut_path == crate::decoder::BUILTIN_GPLOG_LUT_PATH {
                    crate::decoder::BUILTIN_GPLOG_LUT_NAME.to_string()
                } else if s.lut_path == crate::decoder::BUILTIN_X6_LUT_PATH {
                    crate::decoder::BUILTIN_X6_LUT_NAME.to_string()
                } else if s.lut_path == crate::decoder::BUILTIN_HLG_LUT_PATH {
                    crate::decoder::BUILTIN_HLG_LUT_NAME.to_string()
                } else {
                    std::path::Path::new(&s.lut_path)
                        .file_name().and_then(|f| f.to_str()).unwrap_or("(invalid)")
                        .to_string()
                };
                ui.add(egui::TextEdit::singleline(&mut label).interactive(false).desired_width(160.0));
                if ui.button(tr("Browse…")).clicked() {
                    browse_lut = true;
                }
                if ui.button(tr("Clear")).clicked() {
                    s.lut_path.clear();
                }
            });
            if let Some(info) = braw_info.as_deref() {
                if !vr180_braw::BrawColorInfo::is_film_gamma(s.braw_color.effective_gamma(info)) {
                    ui.label(RichText::new(tr(
                        "The clip's Film→video LUT fits the Film gamma only — it is switched off while another gamma is selected above."
                    )).small().color(Color32::GRAY));
                }
            }
            // Quick re-apply of the embedded log→709 LUT for THIS source
            // (autoloaded on open; offered here in case it was cleared). Only
            // the source-matching builtin is shown — no DJI LUT on a `.360`,
            // no GoPro LUT on an `.osv`.
            match kind {
                Some(vr180_pipeline::SourceKind::DjiOsv)
                    if s.lut_path != crate::decoder::BUILTIN_OSMO_LUT_PATH =>
                {
                    if ui.button(tr("Use built-in Osmo 360 D-LogM→709")).clicked() {
                        s.lut_path = crate::decoder::BUILTIN_OSMO_LUT_PATH.to_string();
                        s.lut_intensity = 1.0;
                    }
                }
                Some(vr180_pipeline::SourceKind::GoProEac)
                    if s.lut_path != crate::decoder::BUILTIN_GPLOG_LUT_PATH =>
                {
                    if ui.button(tr("Use built-in GoPro GP-Log→709")).clicked() {
                        s.lut_path = crate::decoder::BUILTIN_GPLOG_LUT_PATH.to_string();
                        s.lut_intensity = 1.0;
                    }
                }
                Some(vr180_pipeline::SourceKind::Insta360Insv)
                    if s.lut_path != crate::decoder::BUILTIN_X6_LUT_PATH =>
                {
                    if ui.button(tr("Use built-in Insta360 X6 I-Log→709")).clicked() {
                        s.lut_path = crate::decoder::BUILTIN_X6_LUT_PATH.to_string();
                        s.lut_intensity = 1.0;
                    }
                }
                Some(vr180_pipeline::SourceKind::KandaoQoocam)
                    if s.lut_path != crate::decoder::BUILTIN_HLG_LUT_PATH =>
                {
                    if ui.button(tr("Use built-in HLG→709 (QooCam HDR)")).clicked() {
                        s.lut_path = crate::decoder::BUILTIN_HLG_LUT_PATH.to_string();
                        s.lut_intensity = 1.0;
                    }
                }
                Some(vr180_pipeline::SourceKind::BlackmagicRaw)
                    if braw_lut.as_deref().map_or(false, |l| s.lut_path != l) =>
                {
                    if ui.button(tr("Use the clip's embedded Film→video LUT")).clicked() {
                        s.lut_path = braw_lut.clone().unwrap_or_default();
                        s.lut_intensity = 1.0;
                    }
                }
                _ => {}
            }
            ui.add_enabled_ui(!s.lut_path.is_empty(), |ui| {
                ui.add(egui::Slider::new(&mut s.lut_intensity, 0.0..=1.0)
                    .text(tr("Intensity")).fixed_decimals(2));
            });

            ui.separator();

            // Detail: equirect-aware sharpening (ported from the Python app;
            // runs on every exportable source). Noise reduction has its own
            // section.
            ui.label(RichText::new(tr("Detail")).strong());
            ui.add(egui::Slider::new(&mut s.sharpen_amount, 0.0..=2.0)
                .text(tr("Sharpen")).fixed_decimals(2))
                .on_hover_text(tr("Equirect-aware unsharp mask (cos-latitude weighted). \
                    0.5 subtle, 1.0 moderate, 2.0 strong."));
            ui.add_enabled_ui(s.sharpen_amount > 0.0, |ui| {
                ui.add(egui::Slider::new(&mut s.sharpen_radius, 0.5..=3.0)
                    .text(tr("Radius")).fixed_decimals(2));
            });
            if ui.button(tr("Reset detail")).clicked() {
                s.sharpen_amount = 0.0; s.sharpen_radius = 1.5;
            }
            // (Noise reduction has moved to its own "Noise Reduction" section.)
        }

        if browse_lut {
            let ctx = ui.ctx().clone();
            self.spawn_dialog(&ctx, || {
                rfd::FileDialog::new()
                    .add_filter("Cube LUT", &["cube", "CUBE"])
                    .pick_file()
                    .map(DialogResult::Lut)
            });
        }
    }

    /// Dedicated Noise Reduction section. The caller only shows this when
    /// [`denoise_supported`](Self::denoise_supported) is true (macOS 26+ on
    /// capable Apple Silicon) and an exportable clip is loaded.
    fn draw_denoise_panel(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut s.denoise_strength, 0.0..=1.0)
                .text(tr("Strength")).fixed_decimals(2));
            if ui.button(tr("Reset")).clicked() {
                s.denoise_strength = 0.0;
            }
        });
        ui.add_space(6.0);
        ui.label(RichText::new(tr(
            "Temporal noise reduction (Apple VideoToolbox). Averages sensor \
             noise across neighbouring frames before projection — most \
             effective on low-light / high-ISO footage. Runs fully in 10-bit."
        )).small().color(Color32::GRAY));
        ui.add_space(4.0);
        ui.label(RichText::new(tr("⚠ Applied on export only — not shown in the preview."))
            .small().color(Color32::from_rgb(220, 180, 90)));
        ui.label(RichText::new(tr("⚠ Slows the export down (Apple's temporal filter runs at full source resolution)."))
            .small().color(Color32::from_rgb(220, 180, 90)));
    }

    fn draw_view_adjust_panel(&mut self, ui: &mut egui::Ui) {
        let zoom = self.preview_zoom;
        // Snapshot Auto-align state before the &mut settings borrow below.
        let align_ok = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
                | Some(vr180_pipeline::SourceKind::Insta360Insv)
                | Some(vr180_pipeline::SourceKind::KandaoQoocam)
                | Some(vr180_pipeline::SourceKind::SbsFisheye)
                | Some(vr180_pipeline::SourceKind::BlackmagicRaw)
                | Some(vr180_pipeline::SourceKind::GoProEac)
        );
        let align_running = self.align_rx.is_some();
        let align_status = self.align_status.clone();
        let is_still = self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
        let stereo_shared = self.clip.as_ref().map_or(false, |c| c.source_kind.stereo_sibling().is_some());
        let s = &mut self.settings;
        ui.label(RichText::new(tr("Global (both eyes)")).small().color(Color32::GRAY));
        if matches!(s.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Reframe) {
            ui.label(RichText::new(tr(
                "Reframed mode aims the view with the Reframed view pan; these angles apply to VR180 output only."
            )).small().color(Color32::from_rgb(130, 150, 175)));
        }
        // Rotational alignment: extra-fine (0.15×) on top of the zoom
        // scaling, since these need sub-degree precision.
        let rot_fine = 0.15;
        // Global pano-map: Up/Down arrows step 0.1° per press.
        let pano_key_step = 0.1;
        fine_slider(ui, zoom, &mut s.pano_yaw_deg, -180.0..=180.0, tr("Yaw (°)"), 3, rot_fine, pano_key_step);
        fine_slider(ui, zoom, &mut s.pano_pitch_deg, -90.0..=90.0, tr("Pitch (°)"), 3, rot_fine, pano_key_step);
        fine_slider(ui, zoom, &mut s.pano_roll_deg, -45.0..=45.0, tr("Roll (°)"), 3, rot_fine, pano_key_step);
        ui.add_space(4.0);
        ui.label(RichText::new(tr("Stereo offset (right = +, left = −)"))
            .small().color(Color32::GRAY));
        if stereo_shared {
            ui.label(RichText::new(tr("Shared by this camera's video and stills — one camera, one eye alignment."))
                .small().color(Color32::from_rgb(130, 150, 175)));
        }
        // Stereo offset: Up/Down arrows step 0.01° per press.
        let stereo_key_step = 0.01;
        fine_slider(ui, zoom, &mut s.stereo_yaw_deg, -10.0..=10.0, tr("Yaw (°)"), 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_pitch_deg, -10.0..=10.0, tr("Pitch (°)"), 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_roll_deg, -10.0..=10.0, tr("Roll (°)"), 3, rot_fine, stereo_key_step);
        ui.label(RichText::new(tr("Tip: select a field and press ↑ / ↓ to step its value precisely."))
            .small().color(Color32::from_rgb(130, 150, 175)));
        ui.add_space(4.0);
        let mut start_align = false;
        ui.horizontal(|ui| {
            if ui.button(tr("Reset to 0")).clicked() {
                s.pano_yaw_deg = 0.0;
                s.pano_pitch_deg = 0.0;
                s.pano_roll_deg = 0.0;
                s.stereo_yaw_deg = 0.0;
                s.stereo_pitch_deg = 0.0;
                s.stereo_roll_deg = 0.0;
            }
            // Auto align: measure the inter-eye disparity field on a few
            // frames and ADD the fitted pitch/roll (and gated yaw)
            // correction to the stereo sliders. Fisheye-family sources.
            if is_still {
                // Auto align measures disparity over several frames — not for a still.
            } else if align_running {
                ui.spinner();
                ui.label(RichText::new(tr("Aligning…")).small());
            } else if ui.add_enabled(align_ok, egui::Button::new(tr("Auto align")))
                .on_hover_text(tr(
                    "Measures the disparity between the eyes on a few frames \
                     and auto-fills the stereo pitch/roll/yaw. Adds to the \
                     current values — tweak after, or run again to refine."))
                .clicked()
            {
                start_align = true;
            }
        });
        if let Some(status) = align_status {
            ui.label(RichText::new(status).small().color(Color32::from_rgb(130, 170, 130)));
        }
        if start_align {
            self.start_auto_align();
        }
    }

    /// Spawn the one-shot Auto align worker for the active clip.
    fn start_auto_align(&mut self) {
        let Some(idx) = self.active_clip else { return };
        let Some(item) = self.batch.get(idx) else { return };
        let segments = self.clip_segments(&item.path);
        let item = &self.batch[idx];
        let mut cfg = Self::build_export_cfg(
            &item.path,
            std::path::Path::new("/tmp/vr180-auto-align-unused.mp4"),
            item.fps,
            item.source_kind,
            item.fisheye_eye_w,
            item.fisheye_eye_h,
            item.width,
            item.height,
            &self.settings,
            &self.export_opts,
            segments,
        );
        cfg.stabilize = false; // common rotation cancels inter-eye; skip IMU load
        let (tx, rx) = crossbeam_channel::bounded(1);
        self.align_rx = Some(rx);
        self.align_status = None;
        let pipeline = self.pipeline.clone();
        std::thread::spawn(move || {
            let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let res = vr180_pipeline::stereo_align::measure_stereo_align(pipeline, &cfg, cancel)
                .map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
    }

    /// Poll the Auto align worker; apply the correction to the stereo
    /// sliders when it lands.
    fn poll_auto_align(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.align_rx else { return };
        match rx.try_recv() {
            Ok(Ok(r)) => {
                self.settings.stereo_pitch_deg += r.d_stereo_pitch_deg;
                self.settings.stereo_roll_deg += r.d_stereo_roll_deg;
                if let Some(y) = r.d_stereo_yaw_deg {
                    self.settings.stereo_yaw_deg += y;
                }
                // No summary text: the user sees the three stereo sliders
                // move. (Failures below still surface a message.)
                self.align_status = None;
                self.align_rx = None;
                ctx.request_repaint();
            }
            Ok(Err(e)) => {
                self.align_status = Some(format!("{}: {e}", tr("Auto align failed")));
                self.align_rx = None;
                ctx.request_repaint();
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.align_status = Some(tr("Auto align failed").to_string());
                self.align_rx = None;
            }
        }
    }

    /// Stabilization panel for fisheye sources (own sidebar section).
    fn draw_fisheye_stab_panel(&mut self, ui: &mut egui::Ui) {
        use std::sync::atomic::Ordering;
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        if matches!(kind, Some(k) if k.is_still()) {
            ui.label(RichText::new(tr("Still photo — no stabilization. Use the View adjust / stereo controls to level and align."))
                .small().color(Color32::GRAY));
            return;
        }
        let is_braw = matches!(kind, Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        let is_osv  = matches!(kind, Some(vr180_pipeline::SourceKind::DjiOsv));
        // Insta360: raw gyro fused at load (VQF) — same stab controls as OSV,
        // no background quat load to gate on.
        let is_insv = matches!(kind, Some(vr180_pipeline::SourceKind::Insta360Insv));
        // Kandao QooCam: same raw-gyro fusion as the X6.
        let is_kandao = matches!(kind, Some(vr180_pipeline::SourceKind::KandaoQoocam));
        // DJI OSV loads the lens calib first (instant preview) and streams the
        // per-frame quaternions — needed for stabilization — in the background.
        // While they load, gray the controls; the decoder auto-applies stab the
        // moment they're in (if the box is checked).
        // OSV streams the per-frame quats in the background, but ONLY while
        // stabilization is on. `stab_loading` = a read is in flight (spinner +
        // Cancel); `imu_ready` = the quats are loaded (sliders usable).
        let stab_loading = is_osv && self.control.as_ref()
            .map(|c| c.stab_loading.load(Ordering::SeqCst)).unwrap_or(false);
        // Paused = user turned stab off mid-load: the read stays alive under the
        // hood (and resumes on re-enable), but the loading UI is hidden.
        let stab_paused = is_osv && self.control.as_ref()
            .map(|c| c.imu_progress.pause.load(Ordering::SeqCst)).unwrap_or(false);
        let show_loading = stab_loading && !stab_paused;
        // Ready = the decoder says so, OR the full IMU is already cached for
        // this clip. The cache check keeps the sliders live while the decoder
        // is stopped (S / Stop tears `control` down) and right after a
        // restart, instead of graying them until a decoder happens to
        // re-publish the flag.
        let quats_cached = is_osv && self.loaded_path.as_ref()
            .and_then(|p| crate::decoder::cached_dji_imu(p))
            .map(|imu| !imu.frame_quats.is_empty()).unwrap_or(false);
        let quats_ready = !is_osv || quats_cached || self.control.as_ref()
            .map(|c| c.imu_ready.load(Ordering::SeqCst)).unwrap_or(false);
        let imu_pct = if show_loading {
            self.control.as_ref().map(|c| c.imu_progress.percent()).unwrap_or(0)
        } else { 0 };
        let s = &mut self.settings;

        if !(is_braw || is_osv || is_insv || is_kandao) {
            ui.label(RichText::new(
                tr("No gyro stabilization available for this source yet.")
            ).small().color(Color32::GRAY));
            return;
        }
        let stab_label = if is_braw {
            tr("Stabilization (Blackmagic gyro)")
        } else if is_insv {
            tr("Stabilization (Insta360 gyro)")
        } else if is_kandao {
            tr("Stabilization (Kandao gyro)")
        } else {
            tr("Stabilization (DJI camera quats)")
        };
        // The checkbox is ALWAYS enabled: turning it ON starts the quat load,
        // turning it OFF cancels an in-flight load — so it doubles as Cancel.
        ui.checkbox(&mut s.stabilize, stab_label);
        // Stab sliders need the loaded quats — enabled only once ready.
        ui.add_enabled_ui(s.stabilize && quats_ready, |ui| {
            if is_osv || is_insv || is_kandao || is_braw {
                // Camera lock owns the sharp frame-0 lock; the smoothing
                // sliders are the soft-stab controls and gray out under it.
                ui.checkbox(&mut s.dji_camera_lock, tr("Camera lock"))
                    .on_hover_text(tr("Lock the view to the first frame; smoothing and max correction are ignored."));
                let cam_lock = s.dji_camera_lock;
                ui.add_enabled_ui(!cam_lock, |ui| {
                    ui.add(egui::Slider::new(&mut s.dji_smooth_ms, 0.0..=3000.0)
                        .text(tr("Smooth (ms)")));
                });
                // Under camera lock the cap is ignored (fully locked no matter
                // what the camera does), so Max corr grays out with the rest.
                ui.add_enabled_ui(!cam_lock, |ui| {
                    ui.add(egui::Slider::new(&mut s.dji_max_corr_deg, 0.0..=45.0)
                        .text(tr("Max corr (°)")));
                });
                // Velocity→smoothing response curve (Python "Response").
                // <1 follows motion early; >1 holds longer, then catches up.
                ui.add_enabled_ui(!cam_lock, |ui| {
                    ui.add(egui::Slider::new(&mut s.dji_responsiveness, 0.2..=3.0)
                        .fixed_decimals(1)
                        .text(tr("Response")));
                });
            }
        });
        if show_loading {
            ui.horizontal(|ui| {
                ui.spinner();
                let txt = if imu_pct > 0 {
                    format!("{} {}%", tr("Loading stabilization data…"), imu_pct)
                } else {
                    tr("Loading stabilization data…").to_string()
                };
                ui.label(RichText::new(txt)
                    .small().color(Color32::from_rgb(120, 180, 120)));
                // Cancel = turn stabilization off, which aborts the in-flight read.
                if ui.button(tr("Cancel")).clicked() {
                    s.stabilize = false;
                }
            });
            // Keep repainting so the spinner animates + the % updates while loading.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    /// Output panel for fisheye sources — projection target + L↔R swap.
    fn draw_fisheye_output_panel(&mut self, ui: &mut egui::Ui) {
        // Dual-fisheye sources (and the X6 still): 195° lens, eye swap +
        // upside-down mount apply (the still honours both too).
        let is_osv = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
                | Some(vr180_pipeline::SourceKind::Insta360Insv)
                | Some(vr180_pipeline::SourceKind::Insta360Dng)
                | Some(vr180_pipeline::SourceKind::KandaoQoocam)
                | Some(vr180_pipeline::SourceKind::KandaoDng)
                | Some(vr180_pipeline::SourceKind::BlackmagicRaw)
        );
        let mode_before = self.settings.fisheye_output_mode;
        let s = &mut self.settings;

        ui.horizontal(|ui| {
            ui.label(tr("Format"));
            egui::ComboBox::from_id_source("fisheye_output_mode")
                .selected_text(s.fisheye_output_mode.as_str())
                .show_ui(ui, |ui| {
                    use crate::decoder::FisheyeOutputMode as M;
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::HalfEquirect, M::HalfEquirect.as_str());
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::Fisheye, M::Fisheye.as_str());
                    ui.selectable_value(&mut s.fisheye_output_mode,
                        M::Reframe, M::Reframe.as_str());
                });
        });

        if matches!(s.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Reframe) {
            use crate::decoder::ReframeAspect;
            ui.add_space(4.0);
            ui.label(RichText::new(tr("Reframed view")).small().color(Color32::GRAY));
            ui.horizontal(|ui| {
                ui.label(tr("Aspect"));
                ui.selectable_value(&mut s.reframe_aspect, ReframeAspect::Square, ReframeAspect::Square.as_str());
                ui.selectable_value(&mut s.reframe_aspect, ReframeAspect::Wide, ReframeAspect::Wide.as_str());
            });
            ui.add(egui::Slider::new(&mut s.reframe_hfov_deg, 30.0..=150.0)
                .fixed_decimals(1).text(tr("Zoom — horizontal FOV (°)")));
            ui.add(egui::Slider::new(&mut s.reframe_defish, 0.0..=1.0)
                .fixed_decimals(2).text(tr("Defish (0 = rectilinear, 1 = fisheye)")));
            ui.add(egui::Slider::new(&mut s.reframe_yaw_deg, -180.0..=180.0)
                .fixed_decimals(1).text(tr("Pan (°)")));
            ui.add(egui::Slider::new(&mut s.reframe_pitch_deg, -90.0..=90.0)
                .fixed_decimals(1).text(tr("Tilt (°)")));
            ui.add(egui::Slider::new(&mut s.reframe_roll_deg, -45.0..=45.0)
                .fixed_decimals(1).text(tr("Roll (°)")));
            let vfov = vr180_pipeline::gpu::reframe_vfov_deg(
                s.reframe_hfov_deg, s.reframe_defish, s.reframe_aspect.ratio());
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{}: {:.1}° × {:.1}°", tr("Field of view"),
                    s.reframe_hfov_deg, vfov)).small().color(Color32::GRAY));
                if ui.small_button(tr("Reset view")).clicked() {
                    s.reframe_yaw_deg = 0.0;
                    s.reframe_pitch_deg = 0.0;
                    s.reframe_roll_deg = 0.0;
                    s.reframe_hfov_deg = 90.0;
                    s.reframe_defish = 0.0;
                }
            });
        }

        // Show the actual output FOV for the current source — the GoPro
        // Max `.360` captures 185°, the DJI/OSV lens 195°.
        if matches!(s.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Fisheye) {
            let fov = if is_osv { "195°" } else { "185°" };
            ui.label(RichText::new(format!("{fov} equidistant — source lens FOV"))
                .small().color(Color32::GRAY));
        }

        if is_osv {
            ui.add_space(4.0);
            ui.checkbox(&mut s.fisheye_swap_eyes, tr("Swap L↔R eyes"));
            // 180° output rotation + implicit eye swap (flipping the rig
            // mirrors the eye positions). Toggling reloads the clip.
            ui.checkbox(&mut s.camera_upside_down, tr("Upside-down mount (180°)"));
        }

        // A BRAW preview decodes at a size chosen for the output mode when
        // the decoder opens (a reframed view keeps more source): crossing
        // into or out of the reframed view respawns the decoder, at the
        // current position and play state.
        let mode_after = self.settings.fisheye_output_mode;
        let is_braw = matches!(self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        let reframe = |m: crate::decoder::FisheyeOutputMode| matches!(m, crate::decoder::FisheyeOutputMode::Reframe);
        if is_braw && self.decoder_alive && reframe(mode_before) != reframe(mode_after) {
            let cur = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
            let was_playing = self.playing;
            tracing::info!("BRAW output mode {} → {}: respawning the decoder for the matching decode size at {:.2} s",
                mode_before.as_str(), mode_after.as_str(), cur);
            self.spawn_decoder(!was_playing);
            self.seek_to(cur);
        }
    }

    /// ".360 Lens calibration" panel — the GoPro-EAC analog of the OSV
    /// Fisheye-lens panel. Reuses `draw_eye_lens` (same Override fields,
    /// independent per source kind), seeded from the file's GEOC factory
    /// calibration (published by the decoder as `detected_calib`). The
    /// values drive a re-dewarp warp in the EAC projection: with Override
    /// off (or seeded values untouched) the output is unchanged; moving
    /// FOV / cx / cy / k re-maps the geometry as if the firmware had
    /// dewarped with those lens parameters. cx/cy are edited in the GEOC
    /// cal-pixel space (4216² on the GoPro Max — CTRX/CTRY units).
    fn draw_eac_lens_panel(&mut self, ui: &mut egui::Ui) {
        let cal_dim = self.clip.as_ref()
            .and_then(|c| c.eac_cal_dim)
            .unwrap_or(4216) as f32;
        let detected = self.control.as_ref()
            .and_then(|c| *c.detected_calib.lock());
        let zoom = self.preview_zoom;
        if detected.is_none() {
            ui.label(RichText::new(tr(
                "No GEOC factory calibration found in this file — \
                 lens Override unavailable."
            )).small().color(Color32::GRAY));
            return;
        }
        let s = &mut self.settings;

        ui.add_space(4.0);
        draw_eye_lens(
            ui, "L", "Left eye", cal_dim, cal_dim, zoom,
            detected.map(|d| d.left),
            &mut s.fisheye_override_left,
            &mut s.fisheye_fov_deg_left,
            &mut s.fisheye_cx_norm_left,
            &mut s.fisheye_cy_norm_left,
            &mut s.fisheye_k_left,
            &mut s.fisheye_k5_left, &mut s.fisheye_p_left,
            false, // not OSV: no k5 / tangential controls
            180.0, [0.0; 4],
        );
        ui.separator();
        draw_eye_lens(
            ui, "R", "Right eye", cal_dim, cal_dim, zoom,
            detected.map(|d| d.right),
            &mut s.fisheye_override_right,
            &mut s.fisheye_fov_deg_right,
            &mut s.fisheye_cx_norm_right,
            &mut s.fisheye_cy_norm_right,
            &mut s.fisheye_k_right,
            &mut s.fisheye_k5_right, &mut s.fisheye_p_right,
            false,
            180.0, [0.0; 4],
        );

        ui.add_space(6.0);
        ui.label(RichText::new(tr(
            "With Override off, each eye uses the camera's factory (GEOC) \
             calibration as-is. Turn Override on to re-map an eye's FOV, \
             center and distortion — for units whose lenses don't quite \
             match the factory model. Values are in the sensor calibration \
             space (cx/cy in cal pixels)."
        )).small().color(Color32::GRAY));
    }

    /// Per-eye lens calibration panel (camera preset + per-eye FOV /
    /// center / KB + Gyroflow load). Mirrors the Python app's lens
    /// parameters. All sliders apply live during playback.
    fn draw_fisheye_panel(&mut self, ui: &mut egui::Ui) {
        // Deferred: the picker must not run inside the egui frame (see
        // the `dialog_tx` field comment). `s` borrows self below.
        let mut load_gyroflow = false;
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
        // k5 (5th radial coeff) only applies to DJI OSV's 5-coefficient
        // model — gate the override slider on the loaded clip being OSV.
        let is_osv = self.clip.as_ref()
            .map(|c| c.source_kind.has_frame_imu())
            .unwrap_or(false);
        // Blackmagic RAW carries a gyro but NO lens calibration: the preset
        // picker and the Gyroflow loader stay (the lens is entered by hand,
        // seeded from the detected image circles), and so do k5 / p.
        let is_braw_lens = self.clip.as_ref()
            .map(|c| c.source_kind.is_braw())
            .unwrap_or(false);
        let hide_preset = is_osv && !is_braw_lens;
        // Generic SBS with the fisheye dewarp off (Source ▸ Input): none of
        // the lens settings apply — say so and gray the panel out.
        let sbs_no_dewarp = matches!(self.clip.as_ref().map(|c| c.source_kind),
                                     Some(vr180_pipeline::SourceKind::SbsFisheye))
            && !self.settings.sbs_dewarp_fisheye;
        if sbs_no_dewarp {
            ui.label(RichText::new(tr("Fisheye dewarp is off for this side-by-side file (Source ▸ Input) — the lens settings below are not applied."))
                .small().color(Color32::from_rgb(130, 150, 175)));
            ui.disable();
        }
        let s = &mut self.settings;

        // ── Camera preset — hidden for OSV: the .osv file carries the full
        //    per-lens calibration, so a camera/profile picker is redundant.
        if !hide_preset {
        ui.horizontal(|ui| {
            ui.label(tr("Camera"));
            let current = if s.fisheye_preset.is_empty() {
                tr("(Auto)").to_string()
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
                        .filter(|p| p.name == "Custom"
                            || (if is_braw_lens { p.name.starts_with("Blackmagic") } else { p.name == "DJI Osmo 360" }))
                    {
                        let selected = s.fisheye_preset == p.name;
                        if ui.selectable_label(selected, p.name).clicked() {
                            s.fisheye_preset = p.name.to_string();
                        }
                    }
                });
        });
        } // hide camera preset for OSV

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
            &mut s.fisheye_k5_left, &mut s.fisheye_p_left, is_osv,
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
            &mut s.fisheye_k5_right, &mut s.fisheye_p_right, is_osv,
            preset_default_fov, preset_k,
        );

        // ── Load Gyroflow lens profile (applies to BOTH eyes; enables
        //    override on both so the loaded values take effect). Hidden for
        //    OSV — its calibration comes from the file, not a profile. ──
        if !hide_preset {
        ui.add_space(6.0);
        if ui.button("Load Gyroflow lens profile (both eyes)…").clicked() {
            load_gyroflow = true;
        }
        } // hide Gyroflow loader for OSV

        ui.add_space(6.0);
        ui.label(RichText::new(tr(
            "With Override off, each eye uses the in-file (OSV) / preset \
             calibration. Turn Override on for an eye to set its FOV, \
             principal point and distortion by hand."
        )).small().color(Color32::GRAY));

        if load_gyroflow {
            let ctx = ui.ctx().clone();
            self.spawn_dialog(&ctx, || {
                rfd::FileDialog::new()
                    .add_filter("Gyroflow lens profile (.json)", &["json", "JSON"])
                    .pick_file()
                    .map(DialogResult::GyroflowProfile)
            });
        }
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
    k5: &mut f32,
    p: &mut [f32; 2],
    is_osv: bool,
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

        // On the off→on transition, seed from the calibration that was just
        // in effect: the in-file (OSV) per-lens FOV / cx / cy / k when we
        // have it, so Override "freezes" the auto factory values for
        // hand-tweaking. Falls back to preset FOV/k + center when there's no
        // detected calib (non-OSV source).
        if *over && !was_over {
            if let Some(d) = detected {
                *fov = d.fov_deg;
                *cx_norm = d.cx_norm;
                *cy_norm = d.cy_norm;
                *k = d.k;
                *k5 = d.k5;
                *p = d.p;
            } else {
                *fov = if preset_default_fov > 0.0 { preset_default_fov } else { 180.0 };
                *cx_norm = 0.5;
                *cy_norm = 0.5;
                *k = preset_k;
                *k5 = 0.0;
                *p = [0.0, 0.0];
            }
        }

        if !*over {
            // Auto = the per-lens FACTORY calibration read from the OSV file:
            // real FOV (from fx), principal point, and KB k1–k4. Show the
            // actual values so the calibration is visible. Falls back to the
            // preset FOV when there's no detected calib (non-OSV source).
            if let Some(d) = detected {
                ui.label(RichText::new(format!(
                    "In-file FOV {:.1}°  ·  cx {:.1}, cy {:.1} px",
                    d.fov_deg, d.cx_norm * native_w, d.cy_norm * native_h,
                )).small().color(Color32::GRAY));
                ui.label(RichText::new(format!(
                    "in-file k = [{:.4}, {:.4}, {:.4}, {:.4}]{}",
                    d.k[0], d.k[1], d.k[2], d.k[3],
                    if is_osv {
                        format!("  k5={:.5}  p=[{:.5}, {:.5}]", d.k5, d.p[0], d.p[1])
                    } else { String::new() },
                )).small().color(Color32::GRAY));
            } else {
                let fov_disp = if preset_default_fov > 0.0 { preset_default_fov } else { 180.0 };
                ui.label(RichText::new(format!("Preset FOV {:.1}°  ·  center cx/cy", fov_disp))
                    .small().color(Color32::GRAY));
            }
            return;
        }

        // FOV (zoom-scaled fine drag). Up/Down arrows nudge by 0.1°.
        fine_slider(ui, zoom, fov, 90.0..=230.0, "Full FOV (°)", 2, 1.0, 0.1);

        // Principal point — absolute pixels at native res, stored as norm.
        // Slider track + zoom-scaled fine drag; Up/Down arrows nudge by 0.1 px.
        let mut cx_px = *cx_norm * native_w;
        let mut cy_px = *cy_norm * native_h;
        if fine_slider(ui, zoom, &mut cx_px, 0.0..=native_w, "cx (px)", 2, 1.0, 0.1)
            .changed()
        {
            *cx_norm = (cx_px / native_w).clamp(0.0, 1.0);
        }
        if fine_slider(ui, zoom, &mut cy_px, 0.0..=native_h, "cy (px)", 2, 1.0, 0.1)
            .changed()
        {
            *cy_norm = (cy_px / native_h).clamp(0.0, 1.0);
        }
        if ui.small_button("Center").clicked() {
            *cx_norm = 0.5;
            *cy_norm = 0.5;
        }

        // KB distortion (zoom-scaled fine drag; Up/Down arrows nudge by 1e-5).
        // k5 only for OSV (DJI's 5-coefficient model); other cameras use the
        // 4-coeff KB.
        let k_title = "KB parameters";
        ui.collapsing(k_title, |ui| {
            for (i, ki) in k.iter_mut().enumerate() {
                fine_slider(ui, zoom, ki, -0.5..=0.5, &format!("k{}", i + 1), 6, 1.0, 0.00001);
            }
            if is_osv {
                // 5th radial coeff (θ¹¹ term). Keeps the projection monotonic
                // past ~90° out to the full lens FOV.
                fine_slider(ui, zoom, k5, -0.05..=0.05, "k5", 6, 1.0, 0.00001);
                // Brown-Conrady tangential (decentering) — small but visible at
                // the rim. Seeded from the file (field 20) when Override engages.
                fine_slider(ui, zoom, &mut p[0], -0.01..=0.01, "p1 (tangential)", 6, 1.0, 0.00001);
                fine_slider(ui, zoom, &mut p[1], -0.01..=0.01, "p2 (tangential)", 6, 1.0, 0.00001);
            }
            if ui.small_button("Reset k to preset").clicked() {
                *k = preset_k;
                if is_osv { *k5 = 0.0; *p = [0.0, 0.0]; }
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
/// Color panel ▸ Blackmagic RAW: the SDK's colour processing. Every control
/// shows the EFFECTIVE value — the clip's own until the user moves it, then
/// the override (marked with a dot); "Reset to clip" clears every override.
/// The embedded Film→video LUT follows the gamma: on for the Film log, off
/// for any other, since it expects Film code values as its input.
fn draw_braw_color_section(
    ui: &mut egui::Ui,
    s: &mut crate::decoder::Settings,
    info: &vr180_braw::BrawColorInfo,
    braw_lut: Option<&str>,
) {
    use vr180_braw::Ranged;
    fn mark(label: &str, overridden: bool) -> String {
        if overridden { format!("{label} •") } else { label.to_string() }
    }
    // A float slider over the SDK's range, showing the effective value.
    fn f32_slider(ui: &mut egui::Ui, field: &mut Option<f32>, r: Ranged<f32>, label: &str, decimals: usize) {
        let mut v = field.unwrap_or(r.value);
        if ui.add(egui::Slider::new(&mut v, r.min..=r.max).text(mark(label, field.is_some())).fixed_decimals(decimals)).changed() {
            *field = Some(v);
        }
    }
    fn flag(ui: &mut egui::Ui, field: &mut Option<bool>, as_shot: u16, label: &str) {
        let mut v = field.unwrap_or(as_shot != 0);
        if ui.checkbox(&mut v, mark(label, field.is_some())).changed() { *field = Some(v); }
    }

    let p = &mut s.braw_color;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Blackmagic RAW").strong());
        if !p.is_as_shot() && ui.small_button(tr("Reset to clip")).clicked() { *p = Default::default(); }
    });
    ui.label(RichText::new(tr(
        "Developed by the Blackmagic RAW SDK from the sensor data — these replace the Tone and White balance controls for this camera. A dot marks a value changed from the clip's."
    )).small().color(Color32::GRAY));

    // Colour science / gamma / gamut.
    let gamma_before = p.effective_gamma(info).to_string();
    ui.horizontal(|ui| {
        ui.label(tr("Color science"));
        let cur = p.color_science_gen.unwrap_or(info.color_science_gen);
        egui::ComboBox::from_id_source("braw_csgen")
            .selected_text(mark(&format!("Gen {cur}"), p.color_science_gen.is_some()))
            .show_ui(ui, |ui| {
                for g in &info.color_science_gen_list {
                    if ui.selectable_label(*g == cur, format!("Gen {g}")).clicked() { p.color_science_gen = Some(*g); }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(tr("Gamma"));
        let cur = p.gamma.clone().unwrap_or_else(|| info.gamma.clone());
        egui::ComboBox::from_id_source("braw_gamma")
            .selected_text(mark(&cur, p.gamma.is_some())).width(220.0)
            .show_ui(ui, |ui| {
                for g in &info.gamma_list {
                    if ui.selectable_label(*g == cur, g).clicked() { p.gamma = Some(g.clone()); }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(tr("Gamut"));
        let cur = p.gamut.clone().unwrap_or_else(|| info.gamut.clone());
        egui::ComboBox::from_id_source("braw_gamut")
            .selected_text(mark(&cur, p.gamut.is_some())).width(220.0)
            .show_ui(ui, |ui| {
                for g in &info.gamut_list {
                    if ui.selectable_label(*g == cur, g).clicked() { p.gamut = Some(g.clone()); }
                }
            });
    });
    // The embedded display LUT expects the Film log: follow the gamma.
    let gamma_after = p.effective_gamma(info).to_string();
    if gamma_after != gamma_before {
        if let Some(lut) = braw_lut {
            if vr180_braw::BrawColorInfo::is_film_gamma(&gamma_after) {
                if s.lut_path.is_empty() { s.lut_path = lut.to_string(); s.lut_intensity = 1.0; }
            } else if s.lut_path == lut {
                s.lut_path.clear();
            }
        }
    }
    let p = &mut s.braw_color;

    ui.add_space(4.0);
    // Exposure: ISO from the camera's steps, exposure in stops, white balance.
    ui.horizontal(|ui| {
        ui.label("ISO");
        let cur = p.iso.unwrap_or(info.iso);
        egui::ComboBox::from_id_source("braw_iso")
            .selected_text(mark(&cur.to_string(), p.iso.is_some()))
            .show_ui(ui, |ui| {
                for i in &info.iso_list {
                    if ui.selectable_label(*i == cur, i.to_string()).clicked() { p.iso = Some(*i); }
                }
            });
    });
    f32_slider(ui, &mut p.exposure, info.exposure, tr("Exposure (stops)"), 2);
    {
        // Kelvin on a log scale (2000–50000 K); tint as the camera counts it.
        let r = info.wb_kelvin;
        let mut k = p.wb_kelvin.unwrap_or(r.value) as f32;
        if ui.add(egui::Slider::new(&mut k, r.min as f32..=r.max as f32).logarithmic(true)
            .text(mark(tr("White balance"), p.wb_kelvin.is_some())).fixed_decimals(0).suffix(" K")).changed()
        {
            p.wb_kelvin = Some(k.round().clamp(r.min as f32, r.max as f32) as u32);
        }
        let r = info.wb_tint;
        let mut t = p.wb_tint.unwrap_or(r.value);
        if ui.add(egui::Slider::new(&mut t, r.min..=r.max).text(mark(tr("Tint"), p.wb_tint.is_some()))).changed() {
            p.wb_tint = Some(t);
        }
    }
    ui.horizontal(|ui| {
        flag(ui, &mut p.highlight_recovery, info.highlight_recovery, tr("Highlight recovery"));
        flag(ui, &mut p.gamut_compression, info.gamut_compression, tr("Gamut compression"));
    });

    ui.add_space(4.0);
    ui.label(RichText::new(tr("Tone curve")).strong());
    f32_slider(ui, &mut p.saturation, info.saturation, tr("Saturation"), 2);
    f32_slider(ui, &mut p.contrast, info.contrast, tr("Contrast"), 2);
    f32_slider(ui, &mut p.midpoint, info.midpoint, tr("Midpoint"), 2);
    f32_slider(ui, &mut p.highlights, info.highlights, tr("Highlights"), 2);
    f32_slider(ui, &mut p.shadows, info.shadows, tr("Shadows"), 2);
    f32_slider(ui, &mut p.black_level, info.black_level, tr("Black level"), 2);
    f32_slider(ui, &mut p.white_level, info.white_level, tr("White level"), 2);
    flag(ui, &mut p.video_black_level, info.video_black_level, tr("Video black level"));
}

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

/// Human ETA: `1h 5m` / `18m 3s` / `45s`. For the export bar's time-left.
fn format_eta(secs: f64) -> String {
    if !secs.is_finite() {
        return "—".to_string();
    }
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{}s", s)
    }
}
