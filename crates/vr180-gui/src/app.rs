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
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::decoder::{
    DecodedFrame, DecoderCommand, DecoderConfig, DecoderControl, Settings, start_decoder,
};

// The app in parts. A part is a module of `app`, and what it holds goes by
// its name here as it did when the parts were one file.
mod clip_info;
mod clips;
mod dialogs;
mod display_3d;
mod export_options;
mod export_run;
mod joined;
mod lang;
mod load;
mod panel_colour;
mod panel_lens;
mod panels;
mod playback;
mod session;
mod snapshot;
mod take_tools;

use clip_info::*;
use clips::*;
use dialogs::*;
use display_3d::*;
pub(crate) use export_options::*;
use export_run::*;
use joined::*;
use lang::*;
use load::*;
#[cfg(test)]
use panel_colour::*;
use panels::*;
use playback::*;
use session::*;
use snapshot::*;
use take_tools::*;

#[cfg(test)]
mod camera_raw_tests;
#[cfg(test)]
mod ettr_tests;
#[cfg(test)]
mod export_size_tests;
#[cfg(test)]
mod file_info_tests;
#[cfg(test)]
mod join_and_project_tests;
#[cfg(test)]
mod reframe_bitrate_tests;
#[cfg(test)]
mod spatial_export_tests;
#[cfg(test)]
mod take_tests;

/// The app's project file: the list with the export bar's choices.
type Project = crate::project::Project<ExportOptions>;

/// The single egui application.
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
    /// Format of the pass egui draws the window in (the surface's).
    surface_format: wgpu::TextureFormat,
    /// The preview as Apple's players will show the export: a video frame
    /// is read with Apple's Rec.709 video curve instead of as sRGB
    /// (`display_colour`). The user's switch, persisted next to
    /// settings.json; never set where the match is not available.
    match_apple_playback: bool,
    /// Draws the picture while the match is on; built when first needed.
    preview_painter: Option<crate::display_colour::PreviewPainter>,

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
    /// The X6 view choice the running decoder was started with (see the
    /// reload in `update`): it binds at decoder start like the eye order.
    insv_view_applied: vr180_fisheye::X6ViewChoice,
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
    /// Why something the user just opened was not taken (a raw take the
    /// camera never closed, …). Shown in the Source panel until the next open.
    import_notice: Option<String>,
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
    /// The loaded clip's own statistics ([`vr180_pipeline::clip_stats`]) and
    /// what they were sampled for, once they are in: its preview, its stills
    /// and an export of it are all rendered by them.
    clip_stats: Option<(ClipStatsFor, crate::decoder::ClipStatsRef)>,
    /// The sampling in flight: what it is for, how to call it off, where
    /// its outcome arrives.
    clip_stats_job: Option<(ClipStatsFor, Arc<std::sync::atomic::AtomicBool>, ClipStatsRx, Arc<vr180_pipeline::clip_stats::Progress>)>,
    /// What the last sampling failed for: not tried again while it is loaded.
    clip_stats_failed: Option<ClipStatsFor>,
    /// What a sampling is about to start for, and the moment it may: once
    /// what it is for has stood for a while, so a control that changes
    /// what the decoder gives (a Blackmagic RAW clip's colour settings,
    /// dragged) does not start one at every step.
    clip_stats_asked: Option<(ClipStatsFor, std::time::Instant)>,
    /// The handle of them the decoder was last given.
    last_pushed_clip_stats: Option<crate::decoder::ClipStatsRef>,
    /// The device the sampling runs on ([`Self::export_device`]'s: its
    /// readbacks must stay off the one the window presents with), made
    /// when a clip first needs it.
    clip_stats_device: Option<Arc<vr180_pipeline::gpu::Device>>,
    /// True while the export queue is draining (one item exports at a time).
    batch_running: bool,
    /// The system's word that the app is at work, held from a run's first
    /// job to its last ([`Self::logic`]): a run goes on at full speed with
    /// the window covered or another app in front, and the machine stays
    /// awake through it.
    busy: Option<crate::activity::Busy>,
    /// Index of the item the current `export_job` belongs to (only
    /// meaningful while `batch_running`).
    batch_current: Option<usize>,
    /// `export_opts` frozen at run start so mid-run edits can't change
    /// later items.
    batch_opts: Option<ExportOptions>,
    /// The run in flight writes the takes themselves and not their view
    /// ([`TakeRun`], started from the Source panel); `None`: an export.
    take_run: Option<TakeRun>,
    /// The Source panel's choices for such a run.
    take_tools: TakeTools,
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

    // ─── The list as a project, and one file from several clips ────
    /// The clips of an export run are written as ONE file, in the list's
    /// order (see [`JoinRun`]). The user's choice, kept with the project.
    join_clips: bool,
    /// A run that writes one file, while it is in flight.
    join_run: Option<JoinRun>,
    /// The file the last run joined its clips into (for the summary).
    joined_output: Option<PathBuf>,
    /// What that run said of enhanced frames it left out ([`JoinRun::note`]).
    joined_note: Option<String>,
    /// Why a joined export did not start, or did not end in a joined file
    /// (shown in the export bar until dismissed).
    join_notice: Option<String>,
    /// The project file the list was opened from or last saved to.
    project_path: Option<PathBuf>,
    /// The list as that file holds it (to tell what is not saved yet).
    project_saved: Option<Project>,
    /// The session as last written, and when the list was last compared
    /// with it.
    session_saved: Option<Project>,
    session_checked_at: std::time::Instant,
    /// Why the session could not be written the last time it was tried
    /// (it is tried again at the next look; said in the log once).
    session_error: Option<String>,
    /// The last session's list, put back on the first frame.
    pending_session: Option<Project>,
    /// The last launch ended while a clip was loading: the list comes back
    /// without one on screen.
    session_no_autoload: bool,
    /// The mark this launch has on disk (`project::GuardStage`): the list
    /// is being put back, or a clip is loading that has shown no frame yet.
    session_guard: Option<crate::project::GuardStage>,
    /// Frames drawn since the session's list was put back: the mark of
    /// that stage goes once a few have come out whole.
    session_restore_frames: Option<u8>,
    /// Clips of a project being looked for on threads: where the answers
    /// arrive and how many are owed.
    restore_tx: crossbeam_channel::Sender<RestoredClip>,
    restore_rx: crossbeam_channel::Receiver<RestoredClip>,
    restore_owed: usize,
    restore_seq: u64,
    /// The entry that goes on screen when its file is found: the project's
    /// own (if nothing else is on screen by then), or one the user clicked.
    restore_activate: Option<u64>,
    restore_clicked: Option<u64>,
    /// With nothing on screen when the last answer is in, the first clip
    /// that was found goes there.
    restore_autoload: bool,
    /// The list being put back was read from the project file itself (not
    /// from the session, which may hold later changes): once its clips are
    /// found, the list is what that file holds.
    restore_from_file: bool,
    /// New / Open waiting for the user's word: the list has changes that
    /// no project file holds.
    pending_project: Option<PendingProject>,

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
    /// A raw video clip's snapshot also writes the frame as a linear DNG, on
    /// its own thread: the file and where its outcome arrives.
    frame_dng_job: Option<(PathBuf, crossbeam_channel::Receiver<std::result::Result<(), String>>)>,
    /// The frame on screen being written as the sensor recorded it (the
    /// transport row's Original DNG): what it says once it is written.
    original_dng_job: Option<crossbeam_channel::Receiver<std::result::Result<String, String>>>,
    /// The loaded raw take's enhanced frames, as last looked up.
    enhanced_probe: EnhancedProbe,
    /// A raw clip's Camera Raw settings being read from its camera-original
    /// DNGs, on a thread: the take they are for and where the outcome arrives.
    raw_settings_job: Option<(PathBuf, crossbeam_channel::Receiver<std::result::Result<Option<vr180_pipeline::x6_raw::CameraRawSettings>, String>>)>,
    /// What the last such read came to, and the take it was for.
    raw_settings_status: Option<(PathBuf, String)>,
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

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Raw takes: keep the list of a take's pairs between sessions (walking
        // a take on a network volume takes seconds).
        if let Some(dir) = crate::decoder::Settings::config_path().and_then(|p| p.parent().map(|d| d.join("raw-takes"))) {
            vr180_pipeline::x6_raw::set_cache_dir(Some(dir));
        }
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
        let (restore_tx, restore_rx) = crossbeam_channel::unbounded();
        let (pending_session, session_no_autoload) = Self::read_session();

        Self {
            egui_ctx: cc.egui_ctx.clone(),
            pipeline,
            egui_renderer: wgpu_state.renderer.clone(),
            preview_view_format,
            surface_format: wgpu_state.target_format,
            match_apple_playback: Self::load_match_apple_playback(),
            preview_painter: None,
            loaded_path: None,
            // Files, and folders: a raw video take is one (`open_paths`
            // sorts out which folders are takes).
            pending_open: std::env::args_os().skip(1).map(PathBuf::from)
                .filter(|p| p.is_file() || p.is_dir()).collect(),
            eye_orientation_applied: (defaults.fisheye_swap_eyes, defaults.camera_upside_down),
            insv_view_applied: defaults.insv_view,
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
            take_run: None,
            take_tools: Self::load_take_copy(),
            batch: Vec::new(),
            active_clip: None,
            align_rx: None,
            align_status: None,
            clip_stats: None,
            clip_stats_job: None,
            clip_stats_failed: None,
            clip_stats_asked: None,
            last_pushed_clip_stats: None,
            clip_stats_device: None,
            batch_running: false,
            busy: None,
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
            join_clips: false,
            join_run: None,
            joined_output: None,
            joined_note: None,
            join_notice: None,
            project_path: None,
            project_saved: None,
            session_saved: None,
            session_checked_at: std::time::Instant::now(),
            session_error: None,
            pending_session,
            session_no_autoload,
            session_guard: None,
            session_restore_frames: None,
            restore_tx,
            restore_rx,
            restore_owed: 0,
            restore_seq: 0,
            restore_activate: None,
            restore_clicked: None,
            restore_autoload: false,
            restore_from_file: false,
            pending_project: None,
            preview_zoom: 1.0,
            preview_center: egui::vec2(0.5, 0.5),
            full_res_display: None,
            three_d: ThreeDDisplay::default(),
            snapshot_status: None,
            snapshot_pending: None,
            frame_dng_job: None,
            original_dng_job: None,
            enhanced_probe: EnhancedProbe::default(),
            raw_settings_job: None,
            raw_settings_status: None,
            full_res_key: 0,
            full_res_desired_key: 0,
            full_res_rendered_at: std::time::Instant::now(),
            detail_last_key: 0,
            detail_key_changed_at: std::time::Instant::now(),
            detail_cache: None,
            fisheye_last_seeded: None,
            suppress_reseed_once: false,
            load_pending: None,
            import_notice: None,
            load_token: 0,
            play_when_ready: false,
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
        // Closed in order: the list as it stands, and no mark of a load
        // that never showed.
        self.sync_active_clip_settings();
        self.write_session();
        self.clear_session_guard();
        #[cfg(target_os = "windows")]
        if let Some(staged) = self.pending_update_install.take() {
            match crate::updater::spawn_installer(&staged) {
                Ok(()) => tracing::info!(
                    "updater: installer spawned on exit (v{})", staged.version),
                Err(e) => tracing::error!("updater: {e}"),
            }
        }
    }

    /// eframe calls this before every frame, and it is the ONLY thing it
    /// calls while the window is minimised or fully covered by another one:
    /// `update()` is skipped then. A finished export has to hand over to
    /// the next queued one with nobody looking, so that hand-over lives
    /// here, together with the wake-up that brings this back while a run
    /// is in flight. And the system is told the app is at work for as long
    /// as one is: left to itself it rests an app that is not in front.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain export-job progress + check for completion.
        self.poll_export_job();
        let at_work = self.batch_running || self.export_job.is_some();
        if at_work {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        self.tell_the_system(at_work);
    }

    #[allow(deprecated)]
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Mirror the chosen language into the process global that `tr` reads.
        set_cur_lang(self.lang);
        // The last session's list, once: its clip goes on screen unless the
        // last launch died loading one, or files were named to open now.
        self.tend_session_guard();
        if let Some(session) = self.pending_session.take() {
            // Marked while the list goes back and is first drawn: a session
            // that kills the app doing so is not walked into again.
            crate::project::GuardStage::Restoring.leave();
            self.session_guard = Some(crate::project::GuardStage::Restoring);
            self.session_restore_frames = Some(0);
            let file = session.project_file.clone();
            let autoload = !self.session_no_autoload && self.pending_open.is_empty();
            self.apply_project(session, file, autoload, false);
        }
        self.poll_restore(ctx);
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
        // The X6 view choice (an unconfirmed 7.4K60 clip: wide or Crop)
        // binds at decoder start too — the lens calibration and the readout
        // are built from it — so a change rebuilds the clip's cached IMU and
        // reloads. A clip the file identifies ignores the choice: no reload.
        if self.settings.insv_view != self.insv_view_applied {
            self.insv_view_applied = self.settings.insv_view;
            let unconfirmed = self.clip.as_ref()
                .and_then(|c| c.x6_geometry(self.settings.insv_view))
                .map_or(false, |g| g.detected == vr180_fisheye::X6Mode::Unconfirmed7K4);
            if let (true, Some(p)) = (unconfirmed, self.loaded_path.clone()) {
                tracing::info!("X6 view choice → {:?} — rebuilding the lens calibration, reloading clip", self.settings.insv_view);
                crate::decoder::uncache_dji_imu(&p);
                self.load_file_inner(p, true);
            }
        }
        // (Export-job progress + completion are polled in `logic()`, which
        // runs just before this and also while the window is hidden.)
        // Auto-update: drain events + periodic re-check + popover.
        self.poll_updater(ctx);
        self.poll_dialogs();
        self.poll_auto_align(ctx);
        self.drive_clip_stats(ctx);
        self.draw_update_popover(ctx);
        if self.update_installing {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Drive the full-res still for the zoom magnifier (and snapshots).
        self.poll_full_res(ctx);
        self.poll_snapshot(ctx);
        self.poll_frame_dng(ctx);
        self.poll_original_dng(ctx);
        self.poll_raw_settings(ctx);
        // Import merge/individual prompt (shown when a multi-file recording
        // is picked). Modal-ish: it sits on top until resolved.
        self.draw_import_prompt(ctx);
        self.draw_project_prompt(ctx);
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
        let mut pending_original = false;
        let mut pending_project_save = false;
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if i.key_pressed(egui::Key::S) && (i.modifiers.command || i.modifiers.ctrl) {
                    pending_project_save = true;
                }
                // S — snapshot the frame on screen. (Stop has no hotkey: the
                // old plain-S stop tore the decoder down mid-session by
                // accident too easily; the ■ Stop button remains.)
                if i.key_pressed(egui::Key::S) && i.modifiers.is_none() {
                    pending_snapshot = true;
                }
                // Shift+S — the frame on screen as the sensor recorded it
                // (a raw video clip).
                if i.key_pressed(egui::Key::S) && i.modifiers.shift_only() {
                    pending_original = true;
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
        if pending_original { self.request_original_dng(); }
        if pending_project_save && !self.batch.is_empty() && !self.batch_running && self.export_job.is_none() {
            self.save_project(ctx);
        }

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
                    // The list as a project file (not while exporting: the
                    // run reads the list).
                    let project_label = match self.project_path.as_ref().and_then(|p| p.file_stem()) {
                        Some(name) => format!("{}: {}", tr("Project"), name.to_string_lossy()),
                        None => tr("Project").to_string(),
                    };
                    let idle = !self.batch_running && self.export_job.is_none();
                    let has_clips = !self.batch.is_empty();
                    let (mut new, mut open, mut save, mut save_as) = (false, false, false, false);
                    ui.add_enabled_ui(idle, |ui| {
                        ui.menu_button(project_label, |ui| {
                            if ui.button(tr("New project")).clicked() { new = true; ui.close_menu(); }
                            if ui.button(tr("Open project…")).clicked() { open = true; ui.close_menu(); }
                            ui.separator();
                            ui.add_enabled_ui(has_clips, |ui| {
                                if ui.button(tr("Save project")).clicked() { save = true; ui.close_menu(); }
                                if ui.button(tr("Save project as…")).clicked() { save_as = true; ui.close_menu(); }
                            });
                        }).response.on_hover_text(tr(PROJECT_HOVER));
                    });
                    if new { self.request_project(ProjectAction::New); }
                    if open { self.pick_project_open(ctx); }
                    if save { self.save_project(ctx); }
                    if save_as { self.pick_project_save(ctx); }
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

                    // A single-eye clip shows its one eye: no view mode to
                    // pick, no eye to switch, nothing for a 3D display.
                    if !self.clip_single_eye() {
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
                    } // two-eye clips only

                    // The preview as Apple's players will show the export
                    // (macOS): a view choice, kept across launches.
                    if crate::display_colour::available() {
                        ui.separator();
                        let mut on = self.match_apple_playback;
                        if ui.toggle_value(&mut on, tr(MATCH_PLAYBACK_LABEL))
                            .on_hover_text(tr(MATCH_PLAYBACK_HOVER))
                            .changed()
                        {
                            self.match_apple_playback = on;
                            self.save_match_apple_playback();
                            tracing::info!("preview display match: {}", if on { "on" } else { "off" });
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // (Version lives ONLY in the top-left brand label,
                        // which is the clickable updater entry point.)

                        // Export progress badge in the right corner —
                        // only when a job is in flight.
                        let mut stop_run = false;
                        if let Some(job) = &self.export_job {
                            ui.separator();
                            if ui.small_button(tr("Cancel")).clicked() {
                                // A run that writes one file has no file
                                // without this clip: the run stops.
                                if self.join_run.is_some() {
                                    stop_run = true;
                                } else {
                                    job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                                }
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
                        if stop_run {
                            self.stop_batch();
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
                    // An X6 clip in one of the added recording modes leads
                    // with the mode's menu name.
                    let mode = clip.x6_geometry(self.settings.insv_view)
                        .and_then(|g| x6_mode_label(&g))
                        .map(|m| format!("{m} · ")).unwrap_or_default();
                    let rate = bitrate_label(&clip.video_bitrates_bps, clip.file_bitrate_bps, false)
                        .map(|r| format!(" · {r}")).unwrap_or_default();
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!(
                            "{mode}{} × {} @ {:.2} fps · {} frames{rate}",
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
                        .on_hover_text(if self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) {
                            tr("Save this frame as a JPEG and as a linear DNG in the export folder (S)")
                        } else {
                            tr("Save this frame as a JPEG in the export folder (S)")
                        })
                        .clicked()
                    {
                        self.request_snapshot();
                    }
                    // A raw video clip: the frame as the sensor recorded it.
                    if self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video())
                        && ui.add_enabled(has_frame && self.original_dng_job.is_none(), egui::Button::new(tr("Original DNG")))
                            .on_hover_text(tr("Save this frame as the sensor recorded it: one Bayer DNG per eye, where the snapshot goes and under the same name (Shift+S)"))
                            .clicked()
                    {
                        self.request_original_dng();
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
                // With the display match on, the same texture goes through
                // the painter that reads it as Apple's players would.
                let matched = self.matched_picture(use_full_res);
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
                    let eye_disp = if self.preview_shows_sbs() { w * 0.5 } else { w };
                    let eye_phys = eye_disp * ui.ctx().pixels_per_point();
                    let binned_eye = (c.fisheye_eye_w / 2).max(1) as f32;
                    self.still_native_wanted = self.preview_zoom * eye_phys > binned_eye * 0.95;
                }

                let zoom = &mut self.preview_zoom;
                let center = &mut self.preview_center;
                let reframe = matches!(self.settings.fisheye_output_mode,
                    crate::decoder::FisheyeOutputMode::Reframe);
                let shows_sbs = self.settings.preview_mode == crate::decoder::PreviewMode::Sbs
                    && !self.clip.as_ref().is_some_and(|c| c.single_eye());

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
                        let eye_disp_w = if shows_sbs {
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
                if let Some((picture, encoding)) = matched {
                    ui.painter().add(crate::display_colour::PreviewPaint {
                        picture, slot: crate::display_colour::PaintSlot::Main, uv, encoding,
                    }.shape(img_rect));
                } else {
                    let sized = egui::load::SizedTexture::new(egui_id, egui::vec2(w, h));
                    egui::Image::new(sized).uv(uv).paint_at(ui, img_rect);
                }

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

        // The display match rests on the windows being sRGB to the system:
        // made sure of while it is on (nothing is written where they are
        // already, which is the rule), undone when it goes off. After the
        // 3D display so its window, once it exists, is covered too.
        let matching = self.display_match_on();
        let layers = crate::display_colour::set_window_colour_matching(matching);
        if layers > 0 {
            tracing::info!("display match: {layers} window layer(s) {}",
                if matching { "tagged sRGB" } else { "given their own colour space back" });
        }

        // Push slider edits to the running decoder.
        self.maybe_push_settings();
        // Remember settings across launches (debounced; flushes on close).
        self.persist_settings(ctx);
        self.persist_session(ctx);

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
