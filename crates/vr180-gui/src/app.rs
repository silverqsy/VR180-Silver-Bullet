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

#[cfg(test)]
mod file_info_tests;
#[cfg(test)]
mod export_size_tests;
#[cfg(test)]
mod join_and_project_tests;
#[cfg(test)]
mod take_tests;
#[cfg(test)]
mod ettr_tests;
#[cfg(test)]
mod spatial_export_tests;
#[cfg(test)]
mod reframe_bitrate_tests;
#[cfg(test)]
mod camera_raw_tests;
mod lang;
use lang::*;
mod take_tools;
use take_tools::*;
mod export_options;
pub(crate) use export_options::*;
mod clips;
use clips::*;
mod joined;
use joined::*;
mod export_run;
use export_run::*;
mod session;
use session::*;
mod load;
use load::*;

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
    /// A project file to open.
    ProjectOpen(std::path::PathBuf),
    /// Where the project is saved.
    ProjectSave(std::path::PathBuf),
    Cancelled,
}

/// The app's project file: the list with the export bar's choices.
type Project = crate::project::Project<ExportOptions>;

/// What a clip's statistics were sampled for: its path, and what makes its
/// decoder give other pictures (`App::clip_stats_for`).
type ClipStatsFor = (PathBuf, String);
/// Where a sampling's outcome arrives.
type ClipStatsRx = Receiver<std::result::Result<Arc<vr180_pipeline::clip_stats::ClipStats>, String>>;

/// What a raw clip's automatic exposure is doing, as its controls show it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RawAutoShown {
    /// Not in use.
    Off,
    /// The clip is being metered: the part done, 0..1 (`None`: about to start).
    Metering(Option<f32>),
    /// The exposure of the frame on screen, and the least and the most over
    /// the clip, as the Exposure control would read them.
    Now { stops: f32, range: (f32, f32) },
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

/// Where a frame saved from the screen goes and what its files are called
/// before what tells them apart: `<source stem>_frame<N>` in the export
/// folder, beside the source when none is set.
fn saved_frame_place(source: &std::path::Path, out_dir: Option<&std::path::Path>, frame_idx: u32) -> (PathBuf, String) {
    let dir = out_dir.map(|d| d.to_path_buf())
        .or_else(|| source.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = source.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "frame".into());
    (dir, format!("{stem}_frame{frame_idx}"))
}

/// The first of `name`, `name-2`, `name-3` … that is not `taken`: a frame
/// saved again leaves the files of the time before as they are.
fn unused_frame_name(name: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(name) {
        return name.to_owned();
    }
    (2u32..).map(|n| format!("{name}-{n}")).find(|n| !taken(n)).unwrap_or_else(|| name.to_owned())
}

/// The name in `dir` of a frame written as the sensor recorded it: the
/// first of `name`, `name-2` … none of whose files is there.
fn unused_original_name(dir: &std::path::Path, name: &str) -> String {
    unused_frame_name(name, |n| (0..2).any(|channel| vr180_pipeline::sensor_dng::lone_frame_file(dir, n, channel).exists()))
}

/// What the status line says of a frame written as the sensor recorded it
/// under `name`: its first file, and what the other one ends in
/// (`TAKE_frame5181_CAM0.dng + _CAM1.dng`).
fn original_frame_saved(pair: &vr180_pipeline::sensor_dng::WrittenPair, name: &str) -> String {
    let files: Vec<String> = pair.files.iter()
        .map(|f| vr180_pipeline::sensor_dng::lone_frame_file(std::path::Path::new(""), name, f.0).to_string_lossy().into_owned())
        .collect();
    files.iter().enumerate()
        .map(|(i, file)| if i == 0 { file.as_str() } else { file.strip_prefix(name).unwrap_or(file) })
        .collect::<Vec<_>>()
        .join(" + ")
}

struct DisplayFrame {
    texture: Arc<wgpu::Texture>,
    egui_id: egui::TextureId,
    /// The same picture for `display_colour::PreviewPainter`: made the
    /// first time the frame is drawn with the display match on.
    matched_picture: Option<wgpu::BindGroup>,
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

/// The most the ETTR recovery takes back, in stops. Four stops over, a 100 %
/// white already sits at 16 of the 22 the I-Log scale reaches at code 1.0:
/// nothing shot brighter than that on purpose has highlights left.
const ETTR_MAX_STOPS: f32 = 4.0;

/// A clip is taken as Insta360 I-Log when it is an X6 video whose picture
/// profile (`gamma_mode`) does not say otherwise: every recording mode can be
/// shot in another profile, and a file that states none is taken as I-Log.
/// The built-in I-Log LUT and the ETTR recovery both go by this.
fn takes_ilog(kind: vr180_pipeline::SourceKind, x6: Option<&X6ClipInfo>) -> bool {
    kind == vr180_pipeline::SourceKind::Insta360Insv && x6.map_or(true, |x| x.ilog != Some(false))
}

/// What an Insta360 X6 clip's trailer says about how it was recorded.
#[derive(Debug, Clone, Copy, PartialEq)]
struct X6ClipInfo {
    /// The file's `window_crop_info` — the recording mode's signature
    /// (`None`: no record, or no readable trailer).
    window: Option<vr180_fisheye::InsvWindowCrop>,
    /// `gamma_mode` is I-Log (`None`: the file does not say).
    ilog: Option<bool>,
    /// Mean centre scale of the two lenses, native pixels per radian (from
    /// the file's calibration, else the generic profile).
    paraxial_native: f64,
    single_lens: bool,
    single_eye: bool,
}

impl X6ClipInfo {
    /// Read it from a clip's trailer (a few KB from the end of the file).
    /// A clip whose trailer cannot be read still gets an entry: its mode
    /// then follows the frame size alone.
    fn read(path: &std::path::Path) -> (Self, Option<String>) {
        let id = vr180_pipeline::x6_pair::read_identity(path).ok();
        let bare = vr180_fisheye::Insta360Meta::default();
        (
            Self {
                window: id.as_ref().and_then(|m| m.window_crop),
                ilog: id.as_ref().and_then(|m| m.is_ilog()),
                paraxial_native: vr180_pipeline::insv_imu::x6_paraxial_native(id.as_ref().unwrap_or(&bare)),
                single_lens: id.as_ref().is_some_and(|m| m.single_lens),
                single_eye: vr180_pipeline::x6_pair::single_eye(path),
            },
            id.and_then(|m| m.gamma_mode),
        )
    }

    /// Per-eye side of the 180° canvas at the frame's native sharpness —
    /// `Some` only when the frame is a crop of the lens' view under `view`
    /// (see `ExportResolution::Sharp`).
    fn sharp_eye(&self, eye_w: u32, eye_h: u32, view: vr180_fisheye::X6ViewChoice) -> Option<u32> {
        let g = self.geometry(eye_w, eye_h, view);
        g.cropped.then(|| vr180_pipeline::insv_imu::native_sharpness_side(self.paraxial_native * g.scale_x))
    }

    /// The clip's stream geometry under a view choice (which only an
    /// unconfirmed 3712² clip listens to).
    fn geometry(&self, eye_w: u32, eye_h: u32, view: vr180_fisheye::X6ViewChoice) -> vr180_fisheye::InsvStreamGeometry {
        if self.single_lens {
            return vr180_fisheye::InsvStreamGeometry::resolve_single_lens(self.window.as_ref(), eye_w, eye_h);
        }
        vr180_fisheye::InsvStreamGeometry::resolve(self.window.as_ref(), eye_w, eye_h, view)
    }
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
    /// Insta360 X6 video: the trailer's recording-mode signature and
    /// picture profile.
    x6: Option<X6ClipInfo>,
    /// A raw take of one lens ([`raw_take_of_one_lens`]).
    raw_one_lens: bool,
    /// A raw take's frame in its lens's view ([`RawFrame`]).
    raw_frame: Option<RawFrame>,
    /// Bit rate of each video stream of the recording, bits per second, as
    /// the container reports it (two entries = the two eyes of a
    /// dual-stream file; the first chapter's for a chain). Empty when the
    /// container does not say.
    video_bitrates_bps: Vec<u64>,
    /// The clip's size over its duration, bits per second — shown when the
    /// container states no stream rate (Blackmagic RAW). `None` for a still.
    file_bitrate_bps: Option<u64>,
    /// Where frames are missing from the clip's track, in order (empty: none
    /// are); `None` when the track's own frame times could not be read.
    holes: Option<Vec<vr180_pipeline::frame_timeline::Hole>>,
}

impl ClipInfo {
    /// The frames the camera dropped, for a clip of the Insta360 X6 (its
    /// video files and its raw takes leave a hole in the track where the
    /// sensor lost frames): the places, in order. `None` for another camera
    /// and when the track could not be read.
    fn x6_holes(&self) -> Option<&[vr180_pipeline::frame_timeline::Hole]> {
        let x6 = self.source_kind == vr180_pipeline::SourceKind::Insta360Insv || self.source_kind.is_raw_video();
        self.holes.as_deref().filter(|_| x6)
    }

    /// The source holds ONE eye: the preview shows it alone and an export
    /// writes it as the whole frame.
    fn single_eye(&self) -> bool {
        self.x6.is_some_and(|x| x.single_eye) || self.raw_one_lens
    }

    /// The clip is taken as Insta360 I-Log (see [`takes_ilog`]).
    fn ilog(&self) -> bool {
        takes_ilog(self.source_kind, self.x6.as_ref())
    }

    /// Stream geometry of an X6 clip under `view` (`None` for other
    /// sources): a video file's by its recording mode, a raw take's as its
    /// sensor record places it (no view to choose).
    fn x6_geometry(&self, view: vr180_fisheye::X6ViewChoice) -> Option<vr180_fisheye::InsvStreamGeometry> {
        match self.raw_frame {
            Some(frame) => Some(frame.geometry(self.fisheye_eye_w, self.fisheye_eye_h)),
            None => self.x6.map(|x| x.geometry(self.fisheye_eye_w, self.fisheye_eye_h, view)),
        }
    }

    /// Per-eye side of the native-sharpness canvas (`None` unless the clip's
    /// frame is a crop of the lens' view — see `ExportResolution::Sharp`).
    fn sharp_eye(&self, view: vr180_fisheye::X6ViewChoice) -> Option<u32> {
        match self.raw_frame {
            Some(frame) => frame.sharp_eye(self.fisheye_eye_w, self.fisheye_eye_h),
            None => self.x6.and_then(|x| x.sharp_eye(self.fisheye_eye_w, self.fisheye_eye_h, view)),
        }
    }
}

/// A raw video take that holds one lens's frames: a single-eye source (the
/// take's own description says so; read once per take).
/// Where a raw take's frame sits in its lens's view, as the take's sensor
/// record states it: what the crop features of such a clip go by (the mask
/// of a frame that is part of the view, its native-sharpness size).
#[derive(Debug, Clone, Copy, PartialEq)]
struct RawFrame {
    stated: vr180_fisheye::insta360::StatedFrame,
    /// Mean centre scale of the take's lenses, native pixels per radian.
    paraxial_native: f64,
    one_lens: bool,
}

impl RawFrame {
    /// Read it from a take's small files. `None` for a source that is no
    /// raw take, and for a take without a sensor record (its frame is the
    /// camera's own video frame, the whole view).
    fn read(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> Option<Self> {
        if !kind.is_raw_video() {
            return None;
        }
        let take = vr180_pipeline::x6_raw::describe(path).ok()?;
        Some(Self {
            stated: vr180_pipeline::x6_raw::stated_frame(&take)?,
            paraxial_native: vr180_pipeline::insv_imu::x6_paraxial_native(&vr180_pipeline::x6_raw::lens_meta(&take)),
            one_lens: take.layout.single_channel().is_some(),
        })
    }

    /// The frame's geometry, worked at `eye_w × eye_h`.
    fn geometry(&self, eye_w: u32, eye_h: u32) -> vr180_fisheye::InsvStreamGeometry {
        vr180_fisheye::InsvStreamGeometry::stated(&self.stated, eye_w, eye_h, self.one_lens)
    }

    /// Per-eye side of the 180° canvas at the frame's native sharpness,
    /// when the frame is a crop of the lens' view (see
    /// `ExportResolution::Sharp`).
    fn sharp_eye(&self, eye_w: u32, eye_h: u32) -> Option<u32> {
        let g = self.geometry(eye_w, eye_h);
        g.cropped.then(|| vr180_pipeline::insv_imu::native_sharpness_side(self.paraxial_native * g.scale_x))
    }
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

    /// `…/VR180SilverBullet2.0/preview_colour` — the display-match switch.
    fn match_apple_playback_config_path() -> Option<PathBuf> {
        crate::decoder::Settings::config_path()
            .and_then(|p| p.parent().map(|d| d.join("preview_colour")))
    }

    /// Off unless the user turned it on, and off wherever it cannot work.
    fn load_match_apple_playback() -> bool {
        crate::display_colour::available()
            && Self::match_apple_playback_config_path()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .is_some_and(|s| s.trim() == "apple")
    }

    fn save_match_apple_playback(&self) {
        if let Some(p) = Self::match_apple_playback_config_path() {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, if self.match_apple_playback { "apple" } else { "plain" });
        }
    }

    /// The preview is drawn as Apple's players show the export right now.
    fn display_match_on(&self) -> bool {
        self.match_apple_playback && crate::display_colour::available()
    }

    /// What the loaded clip's picture is made of: a developed photograph is
    /// sRGB, everything else is a frame of a video exported as Rec.709.
    fn preview_encoding(&self) -> crate::display_colour::PictureEncoding {
        use crate::display_colour::PictureEncoding;
        if self.clip.as_ref().is_some_and(|c| c.source_kind.is_still()) {
            PictureEncoding::Srgb
        } else {
            PictureEncoding::Rec709Video
        }
    }

    /// The live frame (or the full-resolution still) as the display-matched
    /// painter takes it, with how to read its values. `None` with the match
    /// off or nothing to show: the caller then draws egui's image as before.
    fn matched_picture(&mut self, full_res: bool)
        -> Option<(wgpu::BindGroup, crate::display_colour::PictureEncoding)>
    {
        if !self.display_match_on() { return None; }
        let encoding = self.preview_encoding();
        if self.preview_painter.is_none() {
            let painter = crate::display_colour::PreviewPainter::new(
                &self.pipeline.device, self.surface_format);
            // The paint callback finds it in the renderer's resources.
            self.egui_renderer.write().callback_resources.insert(painter.clone());
            self.preview_painter = Some(painter);
        }
        let painter = self.preview_painter.as_ref()?;
        let frame = if full_res { self.full_res_display.as_mut() } else { self.current_display.as_mut() }?;
        if frame.matched_picture.is_none() {
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.preview_view_format),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
            frame.matched_picture = Some(painter.bind_picture(&self.pipeline.device, &view));
        }
        frame.matched_picture.clone().map(|picture| (picture, encoding))
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
                DialogResult::ProjectOpen(path) => self.request_project(ProjectAction::Open(path)),
                DialogResult::ProjectSave(path) => self.save_project_to(path),
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
                DialogResult::Cancelled => {
                    // A save that was asked for before New / Open: not made,
                    // so the list stays.
                    if self.pending_project.as_ref().is_some_and(|p| p.after_save) {
                        self.pending_project = None;
                    }
                }
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
        let settings = self.worker_settings();
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
                texture: Arc::new(tex), egui_id: id, matched_picture: None,
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
                    texture: f.texture, egui_id: id, matched_picture: None,
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
        // The picture went back to the in point at the trim's out point:
        // the sound goes there too.
        if let Some(t) = self.control.as_ref().and_then(|c| c.take_trim_loop()) {
            if let Some(player) = &self.audio_player {
                player.seek_to(t);
            }
        }
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
            matched_picture: None,
            width: frame.width,
            height: frame.height,
            frame_idx: frame.frame_idx,
            timestamp_s: frame.timestamp_s,
        });
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
        // A jump back at the out point not yet followed is overtaken by this
        // seek: the sound goes where the picture is sent now.
        if let Some(ctl) = &self.control {
            let _ = ctl.take_trim_loop();
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

            // Where the camera dropped frames (an X6 clip): a mark as wide as
            // the hole, two pixels at least, over the track's upper part.
            let holes = self.clip.as_ref().and_then(|c| c.x6_holes()).unwrap_or(&[]);
            let dt = self.clip.as_ref().map(|c| 1.0 / c.fps.max(1e-3) as f64).unwrap_or(0.0);
            for hole in holes {
                let x0 = t_to_x(hole.at_s);
                let x1 = t_to_x(hole.at_s + hole.frames as f64 * dt).max(x0 + 2.0);
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x0, track_y0 - 4.0), egui::pos2(x1, track_y0 + 4.0)),
                    0.0, DROPPED_COLOR);
            }
            // The pointer on a mark says what is missing there.
            if let Some(pos) = resp.hover_pos() {
                let near = holes.iter()
                    .map(|h| (h, (t_to_x(h.at_s) - pos.x).abs().min((t_to_x(h.at_s + h.frames as f64 * dt) - pos.x).abs())))
                    .filter(|(_, d)| *d <= 4.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((hole, _)) = near {
                    resp.clone().on_hover_text_at_pointer(dropped_hole_text(hole, cur_lang() == Lang::Zh));
                }
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

    /// The loaded clip holds ONE eye (see `ClipInfo::single_eye`).
    fn clip_single_eye(&self) -> bool {
        self.clip.as_ref().is_some_and(|c| c.single_eye())
    }

    /// The preview on screen is two eyes side by side (a single-eye clip
    /// always shows its one eye, whatever view mode is remembered).
    fn preview_shows_sbs(&self) -> bool {
        self.settings.preview_mode == crate::decoder::PreviewMode::Sbs && !self.clip_single_eye()
    }

    /// The settings a renderer of the loaded clip works from (the preview
    /// decoder, the zoom still): the sidebar's, taken as a single-eye source
    /// takes them when the clip is one. `self.settings` itself never carries
    /// that — it is what gets saved and handed to the next clip.
    fn worker_settings(&self) -> Settings {
        let mut s = if self.clip_single_eye() { self.settings.for_single_eye() } else { self.settings.clone() };
        s.ilog_source = self.clip.as_ref().is_some_and(|c| c.ilog());
        s.raw_source = self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video());
        s.raw_camera = s.raw_source.then(|| self.loaded_raw_camera());
        s.clip_stats = self.loaded_clip_stats();
        s
    }

    /// The camera the loaded raw clip is developed as: its take's own white
    /// balance where the take records one, the sensor's reference else (and
    /// for a clip that is no raw take).
    fn loaded_raw_camera(&self) -> vr180_pipeline::raw_color::RawCamera {
        match (&self.clip, &self.loaded_path) {
            (Some(clip), Some(path)) if clip.source_kind.is_raw_video() => vr180_pipeline::x6_raw::camera_at(path),
            _ => vr180_pipeline::raw_color::RawCamera::X6,
        }
    }

    /// What the loaded clip's statistics are sampled for: the clip, and
    /// what makes its decoder give other pictures (a Blackmagic RAW clip's
    /// colour settings). `None`: nothing loaded, or a still, which is its
    /// own one frame.
    fn clip_stats_for(&self) -> Option<ClipStatsFor> {
        let clip = self.clip.as_ref().filter(|c| !c.source_kind.is_still())?;
        let decode = if clip.source_kind == vr180_pipeline::SourceKind::BlackmagicRaw { format!("{:?}", self.settings.braw_color) } else { String::new() };
        Some((self.loaded_path.clone()?, decode))
    }

    /// The loaded clip's statistics, where they are in.
    fn loaded_clip_stats(&self) -> Option<crate::decoder::ClipStatsRef> {
        let what = self.clip_stats_for()?;
        self.clip_stats.as_ref().filter(|(of, _)| *of == what).map(|(_, stats)| stats.clone())
    }

    /// The loaded clip is developed with an automatic exposure: it has to
    /// be metered ([`vr180_pipeline::clip_stats::wants_meter`]).
    fn clip_wants_meter(&self) -> bool {
        self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) && self.settings.raw.auto_exposure
    }

    /// What the loaded raw clip's automatic exposure is doing, for its
    /// controls.
    fn raw_auto_shown(&self) -> RawAutoShown {
        if !self.clip_wants_meter() {
            return RawAutoShown::Off;
        }
        if let Some(stats) = self.loaded_clip_stats().filter(|s| s.0.has_meter()) {
            let stage = vr180_pipeline::raw_tone::RawDevelop {
                camera: self.loaded_raw_camera(), settings: self.settings.raw,
                measure: vr180_pipeline::raw_tone::Measure::Wait,
            };
            if let Some(auto) = stats.0.auto_exposure(&stage) {
                let t = self.current_display.as_ref().map_or(0.0, |d| d.timestamp_s as f64);
                let (lo, hi) = auto.range();
                return RawAutoShown::Now {
                    stops: stage.exposure_control(auto.stops_at(t)),
                    range: (stage.exposure_control(lo), stage.exposure_control(hi)),
                };
            }
        }
        RawAutoShown::Metering(self.clip_stats_job.as_ref().map(|job| job.3.fraction()))
    }

    /// The loaded clip is rendered by something that goes by the picture's
    /// statistics: a raw clip always (its development), any other when a
    /// tone control that follows the levels is moved.
    fn clip_wants_stats(&self) -> bool {
        match &self.clip {
            Some(clip) if clip.source_kind.is_raw_video() => true,
            Some(_) => self.settings.tone.needs_stats(),
            None => false,
        }
    }

    /// Keep the loaded clip's statistics coming: take in a sampling that
    /// has finished, call one off that is for a clip no longer loaded, and
    /// start one when the clip needs them and has none. They are sampled
    /// once per clip, on a thread of their own; until they are in, the
    /// preview measures each frame by itself, as a still does.
    fn drive_clip_stats(&mut self, ctx: &egui::Context) {
        let what = self.clip_stats_for();
        if let Some((of, cancel, rx, _)) = &self.clip_stats_job {
            if Some(of) != what.as_ref() {
                cancel.store(true, Ordering::SeqCst);
                self.clip_stats_job = None;
            } else {
                match rx.try_recv() {
                    Ok(Ok(stats)) => {
                        self.clip_stats = Some((of.clone(), crate::decoder::ClipStatsRef(stats)));
                        self.clip_stats_job = None;
                        ctx.request_repaint();
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("clip statistics: {e}: each frame of this clip is measured by itself");
                        self.clip_stats_failed = Some(of.clone());
                        self.clip_stats_job = None;
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        ctx.request_repaint_after(std::time::Duration::from_millis(200));
                    }
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        self.clip_stats_failed = Some(of.clone());
                        self.clip_stats_job = None;
                    }
                }
            }
        }
        let Some(what) = what else {
            self.clip_stats = None;
            return;
        };
        if self.clip_stats.as_ref().is_some_and(|(of, _)| *of != what) {
            self.clip_stats = None;
        }
        // What is there serves unless the clip has to be metered and was
        // not: it is then sampled again, metered (its statistics stay in
        // use meanwhile: they come out the same).
        let metered = self.clip_wants_meter();
        if self.clip_stats.as_ref().is_some_and(|(_, stats)| !metered || stats.0.has_meter()) || self.clip_stats_job.is_some()
            || !self.clip_wants_stats() || self.clip_stats_failed.as_ref() == Some(&what)
        {
            return;
        }
        // Once what it is for has stood for a moment.
        const SETTLED: std::time::Duration = std::time::Duration::from_millis(400);
        let now = std::time::Instant::now();
        match &self.clip_stats_asked {
            Some((of, at)) if *of == what && now >= *at => {}
            Some((of, at)) if *of == what => {
                ctx.request_repaint_after(*at - now);
                return;
            }
            _ => {
                self.clip_stats_asked = Some((what, now + SETTLED));
                ctx.request_repaint_after(SETTLED);
                return;
            }
        }
        let Some(item) = self.batch.iter().find(|b| b.path == what.0) else { return };
        let cfg = Self::build_export_cfg(
            &item.path, std::path::Path::new("/tmp/vr180-clip-stats-unused.mp4"), item.fps, item.source_kind,
            item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height, item.sharp_eye(),
            &self.worker_settings(), &self.export_opts, self.clip_segments(&item.path));
        // An export has the GPU: nothing is sampled while it runs. An
        // export samples the clip it writes by itself, and what it sampled
        // is taken here as it is; else the clip is looked at again shortly.
        if self.batch_running {
            match vr180_pipeline::clip_stats::cached(&cfg).filter(|stats| !metered || stats.has_meter()) {
                Some(stats) => self.clip_stats = Some((what, crate::decoder::ClipStatsRef(stats))),
                None => {
                    const AGAIN: std::time::Duration = std::time::Duration::from_secs(2);
                    self.clip_stats_asked = Some((what, now + AGAIN));
                    ctx.request_repaint_after(AGAIN);
                }
            }
            return;
        }
        let pipeline = match &self.clip_stats_device {
            Some(device) => device.clone(),
            None => {
                let device = self.export_device();
                self.clip_stats_device = Some(device.clone());
                device
            }
        };
        let (tx, rx) = crossbeam_channel::bounded(1);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = Arc::new(vr180_pipeline::clip_stats::Progress::default());
        let (stop, shown) = (cancel.clone(), progress.clone());
        std::thread::spawn(move || {
            let _ = tx.send(vr180_pipeline::clip_stats::for_clip_shown(&pipeline, &cfg, &stop, Some(&shown)).map_err(|e| e.to_string()));
        });
        self.clip_stats_job = Some((what, cancel, rx, progress));
    }

    /// Diff `self.settings` against `last_pushed_settings`; if changed,
    /// push to the shared control and bump the generation counter so
    /// the decoder rebuilds per-eye bundles on its next iteration.
    fn maybe_push_settings(&mut self) {
        // What a raw clip's sharpening counts its radius in: the width of
        // one eye of the frame an export would write now.
        self.settings.export_eye_w = if self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) { self.raw_export_eye_w() } else { 0 };
        // The clip's statistics arriving is a change too: the frame on
        // screen is drawn again by them.
        let clip_stats = self.loaded_clip_stats();
        if self.settings == self.last_pushed_settings && clip_stats == self.last_pushed_clip_stats {
            return;
        }
        self.last_pushed_clip_stats = clip_stats;
        if let Some(ctl) = &self.control {
            *ctl.settings.write() = self.worker_settings();
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
        let matched = self.matched_picture(false);
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
                                if let Some((picture, encoding)) = &matched {
                                    use crate::display_colour::PaintSlot;
                                    ui.painter().add(crate::display_colour::PreviewPaint {
                                        picture: picture.clone(),
                                        slot: if i == 0 { PaintSlot::ThreeDLeft } else { PaintSlot::ThreeDRight },
                                        uv, encoding: *encoding,
                                    }.shape(fit));
                                } else {
                                    let sized = egui::load::SizedTexture::new(tex, fit.size());
                                    egui::Image::new(sized).uv(uv).paint_at(ui, fit);
                                }
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
        // The still is two eyes side by side, or the one eye of a single-eye clip.
        let still_w = if self.clip_single_eye() { out.0 } else { out.0 * 2 };
        let ready = match (&self.full_res_display, cur) {
            (Some(f), Some((ts, _))) =>
                (f.timestamp_s - ts).abs() < 0.02 && f.width == still_w && f.height == out.1,
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
        let (dir, name) = saved_frame_place(&src, self.batch_out_dir.as_deref(), frame_idx);
        let name = unused_frame_name(&name, |n| dir.join(format!("{n}.jpg")).exists());
        let path = dir.join(format!("{name}.jpg"));
        let result = (|| -> anyhow::Result<()> {
            let rgb = self.pipeline.read_texture_rgb8(&tex, w, h)?;
            std::fs::create_dir_all(&dir)?;
            let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 92);
            enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
            Ok(())
        })();
        if result.is_ok() {
            self.spawn_frame_dng(path.with_extension("dng"));
        }
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

    /// A raw video clip's snapshot: the frame on screen also as a linear DNG
    /// at `out` (projected and stabilized as the export would, with the
    /// clip's raw development settings inside), written on its own thread.
    fn spawn_frame_dng(&mut self, out: PathBuf) {
        let Some(src) = self.loaded_path.clone() else { return; };
        let Some(t_s) = self.current_display.as_ref().map(|d| d.timestamp_s) else { return; };
        let Some(item) = self.batch.iter().find(|b| b.path == src).filter(|b| b.source_kind.is_raw_video()) else { return; };
        if self.frame_dng_job.is_some() {
            return;
        }
        let mut cfg = Self::build_export_cfg(
            &item.path, &out, item.fps, item.source_kind,
            item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height, item.sharp_eye(),
            &self.worker_settings(), &self.export_opts, self.clip_segments(&item.path));
        // The snapshot's picture is the preview's, at the size picked without
        // enhanced frames: its DNG keeps that size (an enhanced frame, where
        // there is one, still gives it its pixels).
        cfg.raw_enhanced_eye = None;
        let pipeline = self.export_device();
        let (tx, rx) = crossbeam_channel::bounded(1);
        let path = out.clone();
        std::thread::spawn(move || {
            let result = vr180_pipeline::raw_export::export_frame_dng(pipeline, &cfg, t_s as f64, &path).map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.frame_dng_job = Some((out, rx));
    }

    /// Take over a raw clip's Camera Raw settings once they are read: they
    /// become the clip's raw development when it is still the clip loaded.
    fn poll_raw_settings(&mut self, ctx: &egui::Context) {
        let Some((take, rx)) = self.raw_settings_job.as_ref() else { return; };
        let outcome = match rx.try_recv() {
            Ok(outcome) => outcome,
            Err(crossbeam_channel::TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
                return;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.raw_settings_job = None;
                return;
            }
        };
        let take = take.clone();
        self.raw_settings_job = None;
        let text = match outcome {
            Ok(found) => {
                if let (Some(f), true) = (&found, self.loaded_path.as_ref() == Some(&take)) {
                    tracing::info!("raw development read from {}: {:?}; not carried: {:?}", f.file.display(), f.develop.settings, f.develop.not_carried);
                    self.settings.raw = f.develop.settings;
                }
                camera_raw_status(&found)
            }
            Err(e) => {
                tracing::warn!("raw development: reading Camera Raw settings failed: {e}");
                e
            }
        };
        self.raw_settings_status = Some((take, text));
        ctx.request_repaint();
    }

    /// Original DNG (the transport row's button / Shift+S): the frame on
    /// screen of a raw video clip as the sensor recorded it, one Bayer DNG
    /// per eye, where a snapshot of the frame goes and under its name:
    /// `<source stem>_frame<N>_CAM0.dng` and `…_CAM1.dng` in the export
    /// folder (beside the clip when none is set), `…_frame<N>-2_…` when
    /// the frame was written before. Playback pauses on the frame; the
    /// files are written on their own thread.
    fn request_original_dng(&mut self) {
        if self.original_dng_job.is_some() {
            return;
        }
        let shown = self.current_display.as_ref().map(|d| (d.timestamp_s, d.frame_idx));
        let (Some(src), Some((t_s, frame_idx))) = (self.loaded_path.clone(), shown) else {
            self.snapshot_status = Some((tr("Nothing to snapshot yet").to_string(), std::time::Instant::now()));
            return;
        };
        if !self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) {
            return;
        }
        if self.playing { self.toggle_play_pause(); }
        let out_dir = self.batch_out_dir.clone();
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let (dir, name) = saved_frame_place(&src, out_dir.as_deref(), frame_idx);
            // Looked for on this thread: the folder may be on a slow share.
            let name = unused_original_name(&dir, &name);
            let result = vr180_pipeline::sensor_dng::export_sensor_dng_frame(&src, &dir, &name, t_s)
                .map(|pair| original_frame_saved(&pair, &name))
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.original_dng_job = Some(rx);
        self.snapshot_status = Some((format!("{}…", tr("Writing the camera's frame")), std::time::Instant::now()));
    }

    /// Report the camera's frame once it is written.
    fn poll_original_dng(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.original_dng_job.as_ref() else { return; };
        match rx.try_recv() {
            Ok(Ok(saved)) => {
                tracing::info!("original DNG: {saved}");
                self.snapshot_status = Some((format!("{} {saved}", tr("Saved")), std::time::Instant::now()));
                self.original_dng_job = None;
            }
            Ok(Err(e)) => {
                tracing::warn!("original DNG failed: {e}");
                self.snapshot_status = Some((format!("{}: {e}", tr("Camera original frame failed")), std::time::Instant::now()));
                self.original_dng_job = None;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => self.original_dng_job = None,
        }
    }

    /// Report the frame DNG of a snapshot once it is written.
    fn poll_frame_dng(&mut self, ctx: &egui::Context) {
        let Some((path, rx)) = self.frame_dng_job.as_ref() else { return; };
        let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        match rx.try_recv() {
            Ok(Ok(())) => {
                tracing::info!("snapshot: {}", path.display());
                self.snapshot_status = Some((format!("{} {name}", tr("Saved")), std::time::Instant::now()));
                self.frame_dng_job = None;
            }
            Ok(Err(e)) => {
                tracing::warn!("snapshot: frame DNG failed: {e}");
                self.snapshot_status = Some((format!("{name}: {e}"), std::time::Instant::now()));
                self.frame_dng_job = None;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => self.frame_dng_job = None,
        }
    }

    fn draw_source_panel(&mut self, ui: &mut egui::Ui) {
        let mut select_idx: Option<usize> = None;
        let mut remove_idx: Option<usize> = None;
        let mut requeue_idx: Option<usize> = None;
        let mut toggle_select: Option<usize> = None;
        let mut do_apply_all = false;

        // The order of the list is the order of a joined export: a clip is
        // dragged to another place by its handle.
        let list_id = egui::Id::new("source_clip_list");
        let reorder = self.batch.len() > 1;
        let mut moved: Option<crate::reorder::Move> = None;
        if !self.batch.is_empty() {
            for (i, item) in self.batch.iter().enumerate() {
                let row = ui.horizontal(|ui| {
                    if reorder {
                        crate::reorder::handle(ui, list_id, i).on_hover_text(tr("Drag to change the order"));
                    }
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
                    } else if item.ready() {
                        RichText::new(format!("   {name}"))
                    } else {
                        RichText::new(format!("   {name}")).color(Color32::GRAY)
                    };
                    let hover = match &item.offline {
                        None => tr("Click to make this the active clip").to_string(),
                        Some(off) if off.looking => item.path.display().to_string(),
                        Some(_) => format!("{}\n{}", item.path.display(), tr("The file was not found where the project says it is. Click to look again.")),
                    };
                    if ui.add(egui::Label::new(label).sense(egui::Sense::click()))
                        .on_hover_text(hover)
                        .clicked() && !active
                    {
                        select_idx = Some(i);
                    }
                    // A clip of a project whose file is not at hand.
                    if let Some(off) = &item.offline {
                        let (text, col) = if off.looking {
                            (tr("looking…"), Color32::GRAY)
                        } else {
                            (tr("missing"), Color32::from_rgb(230, 160, 90))
                        };
                        ui.label(RichText::new(text).small().color(col));
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
                }).response.rect;
                if reorder {
                    if let Some(m) = crate::reorder::drop_target(ui, list_id, i, row) {
                        moved = Some(m);
                    }
                }
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

        if let Some(m) = moved { self.move_clip(m); }
        if let Some(i) = select_idx { self.select_clip(i); }
        if let Some(i) = remove_idx { self.remove_clip(i); }
        if let Some(i) = toggle_select {
            if let Some(item) = self.batch.get_mut(i) { item.selected = !item.selected; }
        }
        if let Some(i) = requeue_idx {
            if let Some(item) = self.batch.get_mut(i) {
                // Mid-run a re-armed clip joins the current run; otherwise it
                // waits for the next Export. (A run that writes one file is
                // the clips it started with.)
                item.status = if self.batch_running && self.join_run.is_none() {
                    BatchStatus::Queued
                } else {
                    BatchStatus::Idle
                };
            }
        }
        if do_apply_all {
            // SAME CLASS ONLY — never overwrite a different camera type's
            // settings (OSV stab/calib means nothing to a .360 clip, etc.),
            // nor those of a clip that keeps a setup of its own within the
            // type (`settings_class_key`).
            self.sync_active_clip_settings();
            if let Some(class) = self.active_clip.and_then(|a| self.batch.get(a)).map(BatchItem::settings_class) {
                for i in 0..self.batch.len() {
                    // (A clip whose file is away has no type to go by.)
                    if self.batch[i].ready() && self.batch[i].settings_class() == class {
                        self.apply_current_settings_to_item(i);
                    }
                }
            }
        }

        if !self.batch.is_empty() || self.loaded_path.is_some() {
            ui.separator();
        }
        self.draw_source_info(ui);
        self.draw_take_tools(ui);

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

        // Insta360 X6, 3712² without a usable window record: the wide 7.4K
        // and the 7.4K Crop encode the same size at the same rate, so the
        // file cannot say which it is — the user does. (The clip decodes
        // either way; it renders as the wide mode until told otherwise.)
        // An unrecognized signature only gets the note.
        match self.clip.as_ref().and_then(|c| c.x6_geometry(self.settings.insv_view)).map(|g| g.detected) {
            Some(vr180_fisheye::X6Mode::Unconfirmed7K4) => {
                use vr180_fisheye::X6ViewChoice as V;
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(tr("View"));
                    let v = &mut self.settings.insv_view;
                    let crop = *v == V::Crop;
                    // Auto already renders as wide: only a real change is stored.
                    if ui.selectable_label(!crop, tr("Wide")).clicked() && crop { *v = V::Wide; }
                    if ui.selectable_label(crop, "Crop").clicked() { *v = V::Crop; }
                }).response.on_hover_text(tr("This 3712 × 3712 clip has no usable window record, so the file cannot say whether it is the wide 7.4K or the 7.4K Crop. Pick the one it was shot in: the lens calibration and the rolling-shutter timing follow it."));
            }
            Some(vr180_fisheye::X6Mode::Unknown) => {
                ui.label(RichText::new(tr("The lens geometry of this recording mode is not verified: both eyes are shown through the generic rule so the clip can be inspected."))
                    .small().color(Color32::GRAY));
            }
            _ => {}
        }
    }

    fn draw_source_info(&self, ui: &mut egui::Ui) {
        if let Some(note) = &self.import_notice {
            ui.label(RichText::new(note).small().color(Color32::from_rgb(255, 200, 120)));
        }
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
                // A single-eye clip is no pair of streams: say what it is.
                let kind_name = match (c.single_eye(), c.raw_one_lens) {
                    (true, true) => "Insta360 X6 raw video (single fisheye)",
                    (true, false) => "Insta360 (single fisheye)",
                    _ => c.source_kind.display(),
                };
                ui.label(RichText::new(kind_name).color(kind_color).small());

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
                    // An X6 clip: the frames the camera dropped.
                    if let Some(holes) = c.x6_holes() {
                        let zh = cur_lang() == Lang::Zh;
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(tr("Dropped frames")).color(Color32::GRAY).small());
                            let text = RichText::new(dropped_frames_label(holes, zh));
                            ui.label(if holes.is_empty() { text } else { text.color(DROPPED_COLOR) });
                        }).response.on_hover_text(dropped_frames_hover(holes, zh));
                    }
                    // An archive copy of a raw take: what it was made as.
                    if let Some(note) = Some(p).filter(|_| c.source_kind.is_raw_video()).and_then(|p| vr180_pipeline::x6_raw::archive_note(p)) {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(tr("Archive copy")).color(Color32::GRAY).small());
                            ui.label(archive_note_text(&note, cur_lang() == Lang::Zh));
                        }).response.on_hover_text(tr("This clip is an archive copy of a take, made by this app: not the camera's own recording."));
                    }
                    // The copy is read in the place of its take, which is not at hand.
                    if let Some(stand_in) = self.active_clip.and_then(|i| self.batch.get(i)).filter(|b| b.path == *p).and_then(|b| b.stand_in.as_ref()) {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(tr("In place of")).color(Color32::GRAY).small());
                            ui.label(clip_label(&stand_in.take));
                        }).response.on_hover_text(tr(STAND_IN_HOVER));
                    }
                    if let Some(rate) = bitrate_label(&c.video_bitrates_bps, c.file_bitrate_bps, true) {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(tr("Bitrate")).color(Color32::GRAY).small());
                            ui.label(rate);
                        }).response.on_hover_text(tr("Video bit rate from the file. Two numbers in brackets are the two eyes' streams. ≈ means the whole file's size over its duration."));
                    }
                }
                if c.source_kind.is_eac() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("EAC tile")).color(Color32::GRAY).small());
                        ui.label(format!("{} px", c.eac_tile_w));
                    });
                } else if c.source_kind.is_fisheye() && c.fisheye_eye_w > 0 && !c.single_eye() {
                    // (A single-eye clip's eye is the stream above.)
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tr("Eye dims")).color(Color32::GRAY).small());
                        ui.label(format!("{} × {}", c.fisheye_eye_w, c.fisheye_eye_h));
                    });
                }
                // Insta360 X6 in one of the added recording modes: its menu
                // name (amber while the view is unconfirmed / unrecognized).
                if let Some(g) = c.x6_geometry(self.settings.insv_view) {
                    if let Some(label) = x6_mode_label(&g) {
                        let open = matches!(g.detected, vr180_fisheye::X6Mode::Unconfirmed7K4 | vr180_fisheye::X6Mode::Unknown);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(tr("Mode")).color(Color32::GRAY).small());
                            let text = RichText::new(label);
                            ui.label(if open { text.color(Color32::from_rgb(255, 200, 120)) } else { text });
                        });
                    }
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
        // A single-eye clip has no second eye to match.
        let single_eye = self.clip_single_eye();
        // Still + DNG: only what the DNG carries as metadata is offered.
        let dng_mode = is_still && self.export_opts.still_format == StillFormatChoice::LinearDng;
        // Blackmagic RAW: the SDK develops the sensor data — its colour
        // science, gamma/gamut, white balance, exposure/ISO and tone curve
        // replace the Tone and White balance stages for this camera (kept
        // neutral in `finish_load`), so those sections are not shown.
        let is_braw = matches!(kind, Some(vr180_pipeline::SourceKind::BlackmagicRaw));
        // The ETTR recovery works in the source's log curve: I-Log clips only.
        let ilog = self.clip.as_ref().is_some_and(|c| c.ilog());
        // Raw video: the frames are developed here (white balance, exposure,
        // tone) before every display-referred control below.
        let is_raw = matches!(kind, Some(k) if k.is_raw_video());
        let braw_info = if is_braw { self.clip.as_ref().and_then(|c| c.braw_color.clone()) } else { None };
        let braw_lut = if is_braw { self.clip.as_ref().and_then(|c| c.braw_lut.clone()) } else { None };
        // What the last read of the take's Camera Raw settings came to.
        let raw_status: Option<String> = self.raw_settings_status.as_ref()
            .filter(|(take, _)| Some(take) == self.loaded_path.as_ref()).map(|(_, text)| text.clone());
        let raw_reading = self.raw_settings_job.is_some();
        // What a clip's own bits of interface state are kept under.
        let loaded = self.loaded_path.clone();
        let raw_auto = if is_raw { self.raw_auto_shown() } else { RawAutoShown::Off };
        let raw_camera = self.loaded_raw_camera();
        let s = &mut self.settings;

        if is_raw {
            let read = draw_raw_develop_section(ui, &mut s.raw, raw_camera,
                if raw_reading { Some(tr("Reading…")) } else { raw_status.as_deref() }, !raw_reading, raw_auto);
            // Read the settings a frame was left with in Camera Raw: the
            // frame on screen, from the folder of camera-original DNGs
            // beside the take. On a thread: the folder may be on a share.
            if let (true, Some(take)) = (read, self.loaded_path.clone()) {
                let t_s = self.current_display.as_ref().map_or(0.0, |d| d.timestamp_s as f64);
                let (tx, rx) = crossbeam_channel::bounded(1);
                let (path, repaint) = (take.clone(), ui.ctx().clone());
                std::thread::spawn(move || {
                    let dir = Self::originals_dir_for(&path);
                    let found = vr180_pipeline::x6_raw::camera_raw_settings(&path, &dir, t_s).map_err(|e| e.to_string());
                    let _ = tx.send(found);
                    repaint.request_repaint();
                });
                self.raw_settings_job = Some((take, rx));
            }
            ui.separator();
        }
        if let Some(info) = braw_info.as_deref() {
            draw_braw_color_section(ui, s, info, braw_lut.as_deref());
        } else if is_braw {
            ui.label(RichText::new(tr(
                "Blackmagic RAW colour controls need the clip probe's colour block — rebuild the helper (helpers/braw/build_braw.sh)."
            )).small().color(Color32::GRAY));
        } else {
            // The tone controls: changes of exposure in linear light, made
            // ahead of the LUT. A raw clip has them in its development
            // above, and a linear DNG has no place for them.
            if !is_raw && !dng_mode {
                draw_tone_section(ui, &mut s.tone);
                ui.separator();
            }
            // CDL: lift / gamma / gain, on the codes ahead of the LUT.
            ui.label(RichText::new(tr("Lift / Gamma / Gain")).strong());
            ui.add(egui::Slider::new(&mut s.lift, -1.0..=1.0)
                .text(tr("Lift")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.gamma, 0.2..=3.0)
                .text(tr("Gamma")).fixed_decimals(2));
            ui.add(egui::Slider::new(&mut s.gain, 0.2..=3.0)
                .text(tr("Gain")).fixed_decimals(2));
            // The two zone controls this section had before Shadows and
            // Highlights above took their place: offered while a grade
            // still holds a value in one (and until the section is reset,
            // so a slider does not go while it is being moved through 0).
            let zones_id = egui::Id::new(("cdl_zones_shown", loaded));
            let zones = s.shadow != 0.0 || s.highlight != 0.0
                || ui.data(|d| d.get_temp::<bool>(zones_id)).unwrap_or(false);
            if zones {
                ui.add(egui::Slider::new(&mut s.shadow, -1.0..=1.0)
                    .text(tr("Shadow (earlier)")).fixed_decimals(2));
                ui.add(egui::Slider::new(&mut s.highlight, -1.0..=1.0)
                    .text(tr("Highlight (earlier)")).fixed_decimals(2));
                ui.label(RichText::new(tr(
                    "From a grade made before Shadows and Highlights above existed. They work on the codes; the ones above work in linear light. Reset takes them away."
                )).small().color(Color32::GRAY));
            }
            let reset = ui.button(tr("Reset lift / gamma / gain")).clicked();
            if reset {
                s.lift = 0.0; s.gamma = 1.0; s.gain = 1.0;
                s.shadow = 0.0; s.highlight = 0.0;
            }
            ui.data_mut(|d| d.insert_temp(zones_id, zones && !reset));

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

        if !single_eye {
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
        } // Matching Eyes: two-eye clips only

        ui.separator();

        if dng_mode {
            // Linear DNG output carries the grade as metadata; a 3D LUT,
            // sharpening and automatic eye matching have no DNG equivalent
            // (`still_export::grade_metadata`), and the preview plan drops
            // them too (`Settings::still_dng_mode`) — so they are not offered.
            ui.label(RichText::new(tr(
                "DNG output: the Tone section (Exposure to Blacks), the 3D LUT, sharpening and automatic eye matching cannot travel as DNG metadata — hidden, and not applied to the preview."
            )).small().color(Color32::GRAY));
        } else {
            // ── ETTR recovery: a clip exposed brighter than normal on purpose
            //    ("exposing to the right", for cleaner shadows) has that much
            //    exposure taken back out in the light the sensor recorded,
            //    ahead of the tone controls and the LUT. Which clip was shot
            //    that way, and by how much, only the user knows. ──
            if ilog {
                ui.label(RichText::new(tr("ETTR recovery (I-Log)")).strong());
                ui.horizontal(|ui| {
                    ui.label(tr("Shot over by"));
                    for stops in [0.0_f32, 1.0, 2.0, 3.0] {
                        let text = if stops == 0.0 { "0".to_string() } else { format!("+{stops:.0}") };
                        ui.selectable_value(&mut s.ettr_stops, stops, text);
                    }
                    // The same value, for an amount between the steps.
                    ui.add(egui::DragValue::new(&mut s.ettr_stops)
                        .range(0.0..=ETTR_MAX_STOPS).speed(0.01).fixed_decimals(1));
                    ui.label(tr("stops"));
                });
                ui.label(RichText::new(tr(
                    "For a clip exposed brighter than normal on purpose (exposing to the right). The extra exposure is taken back out in linear light, ahead of the tone controls and the LUT. Clipped highlights stay clipped."
                )).small().color(Color32::GRAY));
                ui.separator();
            }

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
                | Some(vr180_pipeline::SourceKind::X6Raw)
        );
        let align_running = self.align_rx.is_some();
        let align_status = self.align_status.clone();
        let is_still = self.clip.as_ref().map_or(false, |c| c.source_kind.is_still());
        let stereo_shared = self.clip.as_ref().map_or(false, |c| c.source_kind.stereo_sibling().is_some());
        // A single-eye clip: the global angles only — no second eye to
        // offset it against, nothing for Auto align to measure.
        let single_eye = self.clip_single_eye();
        let s = &mut self.settings;
        if !single_eye {
            ui.label(RichText::new(tr("Global (both eyes)")).small().color(Color32::GRAY));
        }
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
        if !single_eye {
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
        } // stereo offset: two-eye clips only
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
            if is_still || single_eye {
                // Auto align measures disparity between the eyes over several
                // frames — not for a still, not for one eye.
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
        if self.clip_single_eye() { return; }
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
            item.sharp_eye(),
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
        let is_insv = matches!(kind, Some(vr180_pipeline::SourceKind::Insta360Insv | vr180_pipeline::SourceKind::X6Raw));
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

    /// The Lens Space part of the Output panel: the lens the picture of a
    /// clip of `kind` is written through — the camera's own (a lens frame)
    /// or an ideal equidistant one, the disc (185° for the cube map of a
    /// GoPro `.360`, 195° for a fisheye) — and what each is for. A clip with
    /// no lens record (a still, a file that is already VR180) has no choice
    /// to make: it is the disc. Nothing in the other views.
    fn lens_space_ui(ui: &mut egui::Ui, s: &mut crate::decoder::Settings, kind: Option<vr180_pipeline::SourceKind>) {
        use crate::decoder::LensSpaceLens;
        if !matches!(s.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Fisheye) {
            return;
        }
        let lens_possible = kind.is_some_and(|k| crate::decoder::lens_frame_possible(k, s));
        if lens_possible {
            ui.horizontal(|ui| {
                ui.label(tr("Lens"));
                ui.selectable_value(&mut s.lens_space, LensSpaceLens::Camera, tr(LensSpaceLens::Camera.as_str()))
                    .on_hover_text(tr(LENS_SPACE_CAMERA_HOVER));
                ui.selectable_value(&mut s.lens_space, LensSpaceLens::Equidistant, tr(LensSpaceLens::Equidistant.as_str()))
                    .on_hover_text(tr(LENS_SPACE_EQUIDISTANT_HOVER));
            });
        }
        let note = if lens_possible && s.lens_space == LensSpaceLens::Camera {
            tr(LENS_SPACE_CAMERA_NOTE).to_string()
        } else {
            let fov = if kind.is_some_and(|k| k.is_eac()) { "185°" } else { "195°" };
            format!("{fov} {}", tr(LENS_SPACE_DISC_NOTE))
        };
        ui.label(RichText::new(note).small().color(Color32::GRAY));
    }

    /// Output panel for fisheye sources — projection target + L↔R swap.
    fn draw_fisheye_output_panel(&mut self, ui: &mut egui::Ui) {
        // Dual-fisheye sources (and the X6 still): 195° lens, eye swap +
        // upside-down mount apply (the still honours both too).
        let is_osv = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
                | Some(vr180_pipeline::SourceKind::Insta360Insv)
                | Some(vr180_pipeline::SourceKind::X6Raw)
                | Some(vr180_pipeline::SourceKind::Insta360Dng)
                | Some(vr180_pipeline::SourceKind::KandaoQoocam)
                | Some(vr180_pipeline::SourceKind::KandaoDng)
                | Some(vr180_pipeline::SourceKind::BlackmagicRaw)
        );
        let mode_before = self.settings.fisheye_output_mode;
        // The frame is a crop of the image circle (the X6 Crop modes).
        let frame_crop = self.clip.as_ref()
            .and_then(|c| c.x6_geometry(self.settings.insv_view))
            .map_or(false, |g| g.cropped);
        // A single-eye clip: the same three outputs, one eye wide.
        let single_eye = self.clip_single_eye();
        let kind = self.clip.as_ref().map(|c| c.source_kind);
        // Which way up the camera was, where the clip on screen's accelerometer says.
        let mount_detected = self.active_clip.and_then(|i| self.batch.get(i)).and_then(|b| b.upside_down);
        let s = &mut self.settings;

        ui.horizontal(|ui| {
            ui.label(tr("Format"));
            egui::ComboBox::from_id_source("fisheye_output_mode")
                .selected_text(tr(s.fisheye_output_mode.label(single_eye)))
                .show_ui(ui, |ui| {
                    use crate::decoder::FisheyeOutputMode as M;
                    for m in [M::HalfEquirect, M::Fisheye, M::Reframe] {
                        ui.selectable_value(&mut s.fisheye_output_mode, m, tr(m.label(single_eye)));
                    }
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

        Self::lens_space_ui(ui, s, kind);

        if is_osv {
            ui.add_space(4.0);
            if !single_eye {
                ui.checkbox(&mut s.fisheye_swap_eyes, tr("Swap L↔R eyes"));
            }
            // 180° output rotation + implicit eye swap (flipping the rig
            // mirrors the eye positions). Toggling reloads the clip. A clip
            // whose accelerometer says which way up the camera was sets it
            // by itself; set by hand, it stays as the user left it.
            if ui.checkbox(&mut s.camera_upside_down, tr("Upside-down mount (180°)")).on_hover_text(tr(UPSIDE_DOWN_HOVER)).changed() {
                s.camera_upside_down_auto = false;
            }
            match (mount_detected, s.camera_upside_down_auto) {
                (Some(_), true) => {
                    ui.label(RichText::new(tr("Set from the clip's accelerometer.")).small().color(Color32::GRAY));
                }
                (Some(detected), false) => {
                    if ui.small_button(tr("Follow the accelerometer"))
                        .on_hover_text(tr("The switch was set by hand. This puts it back to what the clip's accelerometer says."))
                        .clicked()
                    {
                        s.camera_upside_down_auto = true;
                        s.camera_upside_down = detected;
                    }
                }
                (None, _) => {}
            }
        }

        // ── Crop mask — sources whose frame is a crop of the image circle.
        // In the VR180 and fisheye outputs the directions outside the
        // recorded frame are black regardless; these set a black window on
        // top: how much it hides, how round, how soft. The reframed (flat)
        // view has neither: there the frame's edge is repeated, which
        // covers the border stabilization exposes — the sliders gray out.
        if frame_crop {
            let flat = matches!(s.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Reframe);
            ui.add_space(6.0);
            ui.label(RichText::new(tr("Crop mask")).small().color(Color32::GRAY))
                .on_hover_text(tr("A Crop clip holds part of the lens' view: outside the recorded frame the output is black. The mask is fixed in the output, so it stays still under stabilization. The reframed view does not use it."));
            if flat {
                ui.label(RichText::new(tr("Reframed view: no mask — the frame's edge is repeated past the recorded picture, so a stabilization border does not show black."))
                    .small().color(Color32::from_rgb(130, 150, 175)));
            }
            ui.add_enabled_ui(!flat, |ui| {
            let m = &mut s.crop_mask;
            let (mut amount, mut feather, mut round) = (m.amount * 100.0, m.feather * 100.0, m.round * 100.0);
            fine_slider(ui, 1.0, &mut amount, 0.0..=60.0, tr("Mask size (%)"), 1, 1.0, 0.5)
                .on_hover_text(tr("How much of the recorded view the black mask hides on every side. 0 hides nothing."));
            fine_slider(ui, 1.0, &mut feather, 0.0..=50.0, tr("Feather (%)"), 1, 1.0, 0.5)
                .on_hover_text(tr("Soft edge: the picture fades to black over this width inside the mask edge. 0 = hard edge."));
            fine_slider(ui, 1.0, &mut round, 0.0..=100.0, tr("Corner roundness (%)"), 0, 1.0, 1.0)
                .on_hover_text(tr("0 = rectangle, 100 = ellipse."));
            m.amount = (amount / 100.0).clamp(0.0, 0.6);
            m.feather = (feather / 100.0).clamp(0.0, 0.5);
            m.round = (round / 100.0).clamp(0.0, 1.0);
            if !m.is_off() && ui.small_button(tr("Reset mask")).clicked() {
                *m = vr180_pipeline::gpu::CropMask::default();
            }
            });
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
        let single_eye = self.clip_single_eye();
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
        // A single-eye clip has one lens: its settings are the left eye's.
        draw_eye_lens(
            ui, "L", if single_eye { "Lens" } else { "Left eye" }, native_w, native_h, zoom,
            detected.map(|d| d.left),
            &mut s.fisheye_override_left,
            &mut s.fisheye_fov_deg_left,
            &mut s.fisheye_cx_norm_left,
            &mut s.fisheye_cy_norm_left,
            &mut s.fisheye_k_left,
            &mut s.fisheye_k5_left, &mut s.fisheye_p_left, is_osv,
            preset_default_fov, preset_k,
        );
        if !single_eye {
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
        }

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
/// Color panel ▸ Raw development: the white balance and tone controls of a
/// raw video clip, in a raw converter's own terms (kelvin and tint; stops;
/// −100..100). The white balance shows the effective value: the camera's
/// until the user moves it.
///
/// `status`: what the last read of the clip's Camera Raw settings came to,
/// shown under its button; `can_read`: no such read is under way. Returns
/// whether that button was pressed.
/// The tone controls of a clip that is not raw: the six a raw clip has in
/// its development, each a change of exposure in linear light ahead of the
/// LUT (`vr180_pipeline::tone_linear`).
fn draw_tone_section(ui: &mut egui::Ui, tone: &mut vr180_pipeline::tone_linear::ToneSettings) {
    ui.label(RichText::new(tr("Tone")).strong());
    ui.add(egui::Slider::new(&mut tone.exposure, -5.0..=5.0).text(tr("Exposure")).fixed_decimals(2))
        .on_hover_text(tr("In stops, as a change of the light itself: ahead of the LUT, so what the LUT presses against white or black is still there to bring back."));
    let amount = |ui: &mut egui::Ui, v: &mut f32, label: &'static str, hover: &'static str| {
        ui.add(egui::Slider::new(v, -100.0..=100.0).text(tr(label)).fixed_decimals(0)).on_hover_text(tr(hover));
    };
    amount(ui, &mut tone.contrast, "Contrast", "More or less contrast about the middle of the frame's own levels.");
    amount(ui, &mut tone.highlights, "Highlights", "Darkens or brightens the bright parts of the frame and leaves their texture: the change follows the frame's large areas, not each pixel. Below zero, highlights the LUT shows as white come back from what the file holds.");
    amount(ui, &mut tone.shadows, "Shadows", "Brightens or darkens the dark parts of the frame and leaves their texture: the change follows the frame's large areas, not each pixel.");
    amount(ui, &mut tone.whites, "Whites", "Moves the brightest levels: where white sits.");
    amount(ui, &mut tone.blacks, "Blacks", "Moves the darkest levels: where black sits.");
    if ui.button(tr("Reset tone")).clicked() {
        *tone = Default::default();
    }
    ui.label(RichText::new(tr(
        "Changes of exposure in linear light, made ahead of Lift / Gamma / Gain and the LUT. They go by the levels of the whole clip, so a control does the same to every frame of it."
    )).small().color(Color32::GRAY));
}

fn draw_raw_develop_section(
    ui: &mut egui::Ui,
    raw: &mut vr180_pipeline::raw_tone::RawSettings,
    camera: vr180_pipeline::raw_color::RawCamera,
    status: Option<&str>,
    can_read: bool,
    auto: RawAutoShown,
) -> bool {
    ui.label(RichText::new(tr("Raw development")).strong());
    let (shot_k, shot_tint) = camera.as_shot_temperature();
    // Whole numbers, as the sliders show them: a slider handed a fraction
    // writes the rounded value back and reports a change nobody made, which
    // turned "as shot" into a custom white the moment the section was drawn.
    let mut kelvin = raw.temperature.unwrap_or(shot_k as f32).round();
    let mut tint = raw.tint.unwrap_or(shot_tint as f32).round();
    let hover = tr("The white the frame is developed for, as a colour temperature in kelvin and a tint, the way a raw converter states it.");
    if ui.add(egui::Slider::new(&mut kelvin, 2000.0..=50000.0).logarithmic(true).text(tr("Temperature (K)")).fixed_decimals(0))
        .on_hover_text(hover).changed()
    {
        raw.temperature = Some(kelvin.round());
    }
    if ui.add(egui::Slider::new(&mut tint, -150.0..=150.0).text(tr("Tint")).fixed_decimals(0))
        .on_hover_text(hover).changed()
    {
        raw.tint = Some(tint.round());
    }
    ui.add_enabled_ui(raw.temperature.is_some() || raw.tint.is_some(), |ui| {
        if ui.button(tr("As shot")).on_hover_text(tr("The take records no white balance of its own: As shot is the camera's reference white.")).clicked() {
            raw.temperature = None;
            raw.tint = None;
        }
    });
    // Exposure: set by hand, or found by the clip itself frame by frame.
    // While it is found the control shows what the frame on screen has.
    match auto {
        RawAutoShown::Now { stops, .. } if raw.auto_exposure => {
            let mut shown = stops;
            ui.add_enabled(false, egui::Slider::new(&mut shown, -5.0..=8.0).text(tr("Exposure")).fixed_decimals(2));
        }
        _ => {
            ui.add_enabled(!raw.auto_exposure, egui::Slider::new(&mut raw.exposure, -5.0..=5.0).text(tr("Exposure")).fixed_decimals(2));
        }
    }
    ui.checkbox(&mut raw.auto_exposure, tr("Auto exposure")).on_hover_text(tr(
        "The clip sets its own exposure, frame by frame: each frame gets the exposure at which its picture, developed with every other control as it is set here, comes out normally bright. The exposure follows the light smoothly, from both sides; nothing else in the rendering changes with it."));
    if raw.auto_exposure {
        ui.add(egui::Slider::new(&mut raw.auto_ev, -3.0..=3.0).text(tr("EV")).fixed_decimals(2))
            .on_hover_text(tr("Stops added to what the automatic exposure finds: brighter or darker than its own target."));
        ui.add(egui::Slider::new(&mut raw.auto_smooth, 0.0..=10.0).text(tr("Smoothness (s)")).fixed_decimals(1))
            .on_hover_text(tr("How slowly the exposure follows the picture: each frame's exposure is the average over about this many seconds of the clip on either side. 0 follows the picture as it changes; a large value gives the whole clip one exposure."));
        let text = match auto {
            RawAutoShown::Metering(Some(done)) => Some(format!("{} {:.0} %", tr("Metering the clip…"), done * 100.0)),
            RawAutoShown::Metering(None) => Some(tr("Metering the clip…").to_string()),
            RawAutoShown::Now { range, .. } => Some(format!("{} {:+.2} … {:+.2}", tr("Over the clip:"), range.0, range.1)),
            RawAutoShown::Off => None,
        };
        if let Some(text) = text {
            ui.label(RichText::new(text).small().color(Color32::GRAY));
        }
    }
    let amount = |ui: &mut egui::Ui, v: &mut f32, label: &'static str| {
        ui.add(egui::Slider::new(v, -100.0..=100.0).text(tr(label)).fixed_decimals(0));
    };
    amount(ui, &mut raw.contrast, "Contrast");
    amount(ui, &mut raw.highlights, "Highlights");
    amount(ui, &mut raw.shadows, "Shadows");
    amount(ui, &mut raw.whites, "Whites");
    amount(ui, &mut raw.blacks, "Blacks");
    // The detail panel's sharpening, in a raw converter's own terms.
    ui.add_space(4.0);
    ui.label(tr("Sharpening"));
    let sharpen = &mut raw.sharpen;
    ui.add(egui::Slider::new(&mut sharpen.amount, 0.0..=150.0).step_by(1.0).text(tr("Amount")).fixed_decimals(0))
        .on_hover_text(tr("Sharpening of the developed picture, as a raw converter's detail panel does it: the same numbers give the same edges there. The radius is in pixels of the exported frame, so judge it in the zoom view or on a snapshot; the preview shows the export's picture reduced."));
    ui.add_enabled_ui(sharpen.amount >= 0.5, |ui| {
        ui.add(egui::Slider::new(&mut sharpen.radius, 0.5..=3.0).step_by(0.1).text(tr("Radius")).fixed_decimals(1));
        ui.add(egui::Slider::new(&mut sharpen.detail, 0.0..=100.0).step_by(1.0).text(tr("Detail")).fixed_decimals(0))
            .on_hover_text(tr("How much of the finest detail is sharpened. Low values hold halos back, as a raw converter's do: a hard edge is sharpened less, a soft one keeps its steepening."));
        ui.add(egui::Slider::new(&mut sharpen.masking, 0.0..=100.0).step_by(1.0).text(tr("Masking")).fixed_decimals(0))
            .on_hover_text(tr("Keeps the sharpening to edges: the higher, the stronger an edge has to be."));
    });
    // The detail panel's noise reduction: colour only.
    ui.add_space(4.0);
    ui.label(tr("Noise reduction"));
    ui.add(egui::Slider::new(&mut raw.color_noise, 0.0..=100.0).step_by(1.0).text(tr("Color noise")).fixed_decimals(0))
        .on_hover_text(tr("Colour noise reduction, as a raw converter's: the picture's colour is smoothed and its luminance left alone, which takes the coloured speckle out of noise and the colour out of fine lines. 25 is a raw converter's default; 0 turns it off."));
    let mut read = false;
    ui.horizontal(|ui| {
        if ui.button(tr("Reset raw development")).clicked() {
            *raw = Default::default();
        }
        read = ui.add_enabled(can_read, egui::Button::new(tr("Read from Camera Raw")))
            .on_hover_text(tr("Takes the white balance, tone, sharpening and colour noise reduction a frame was left with in Camera Raw. They are read from the Camera original DNG of the frame on screen, or of the nearest frame that has settings, in the folder beside the take. Whatever else was set there is named below and stays there."))
            .clicked();
    });
    if let Some(text) = status {
        ui.label(RichText::new(text).small().color(Color32::GRAY));
    }
    read
}

/// What a read of a take's Camera Raw settings came to, for the line under
/// its button.
fn camera_raw_status(found: &Option<vr180_pipeline::x6_raw::CameraRawSettings>) -> String {
    let Some(found) = found else {
        return tr("No Camera Raw settings found: the folder of Camera original DNGs beside this take has none, or is not there.").to_string();
    };
    // The file, as the folder names it: its channel folder and its name.
    let name = found.file.iter().rev().take(2).collect::<Vec<_>>().into_iter().rev()
        .map(|part| part.to_string_lossy().to_string()).collect::<Vec<_>>().join("/");
    let mut text = format!("{} {name}", tr("Read from"));
    if !found.develop.not_carried.is_empty() {
        let list = found.develop.not_carried.iter()
            .map(|&(what, ref value)| if value.is_empty() { tr(what).to_string() } else { format!("{} {value}", tr(what)) })
            .collect::<Vec<_>>().join(", ");
        text.push_str(&format!(". {} {list}", tr("Not carried over:")));
    }
    if found.eyes_differ {
        text.push_str(&format!(". {}", tr("The other eye's file has other settings: these are channel 0's.")));
    }
    text
}

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
    // A float slider over the SDK's range, showing the effective value
    // (`base` = the value in effect without an override).
    fn f32_slider(ui: &mut egui::Ui, field: &mut Option<f32>, r: Ranged<f32>, base: f32, label: &str, decimals: usize, hover: Option<&str>) {
        let mut v = field.unwrap_or(base);
        let resp = ui.add(egui::Slider::new(&mut v, r.min..=r.max).text(mark(label, field.is_some())).fixed_decimals(decimals));
        if let Some(h) = hover { resp.clone().on_hover_text(h); }
        if resp.changed() { *field = Some(v); }
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

    // Colour science / gamma / gamut. The gamma and gamut lists (and the
    // SDK's defaults) belong to a GENERATION: Gen 6 has no "Blackmagic
    // Design" gamma or gamut at all (a Film clip becomes DaVinci
    // Intermediate / DaVinci Wide Gamut), so the combos list what the chosen
    // generation accepts, and an override it rejects is dropped on switch.
    let gamma_before = p.effective_gamma(info).to_string();
    let gen_before = p.effective_gen(info);
    ui.horizontal(|ui| {
        ui.label(tr("Color science"));
        let cur = gen_before;
        egui::ComboBox::from_id_source("braw_csgen")
            .selected_text(mark(&format!("Gen {cur}"), p.color_science_gen.is_some()))
            .show_ui(ui, |ui| {
                for g in &info.color_science_gen_list {
                    if ui.selectable_label(*g == cur, format!("Gen {g}")).clicked() { p.color_science_gen = Some(*g); }
                }
            });
    });
    let gen_now = p.effective_gen(info);
    if gen_now != gen_before {
        let view = info.for_gen(gen_now);
        if p.gamma.as_deref().map_or(false, |g| !view.accepts_gamma(g)) { p.gamma = None; }
        if p.gamut.as_deref().map_or(false, |g| !view.accepts_gamut(g)) { p.gamut = None; }
    }
    let view = info.for_gen(gen_now);
    ui.horizontal(|ui| {
        ui.label(tr("Gamma"));
        let cur = p.effective_gamma(info).to_string();
        egui::ComboBox::from_id_source("braw_gamma")
            .selected_text(mark(&cur, p.gamma.is_some())).width(220.0)
            .show_ui(ui, |ui| {
                for g in view.gamma_list {
                    if ui.selectable_label(*g == cur, g).clicked() { p.gamma = Some(g.clone()); }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(tr("Gamut"));
        let cur = p.effective_gamut(info).to_string();
        egui::ComboBox::from_id_source("braw_gamut")
            .selected_text(mark(&cur, p.gamut.is_some())).width(220.0)
            .show_ui(ui, |ui| {
                for g in view.gamut_list {
                    if ui.selectable_label(*g == cur, g).clicked() { p.gamut = Some(g.clone()); }
                }
            });
    });
    if gen_now != info.color_science_gen {
        ui.label(RichText::new(tr(
            "Another colour science than the clip's: its own gamma and gamut lists apply (Gen 6 has no Blackmagic Design gammas — a Film clip decodes to DaVinci Intermediate / DaVinci Wide Gamut)."
        )).small().color(Color32::GRAY));
    }
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
        // Into Custom from the combo: start from the curve the previous
        // gamma was built from (Custom's own defaults are a flat, bright
        // curve), so the picture does not jump.
        if gamma_after == vr180_braw::CUSTOM_GAMMA {
            let base = info.tone_baseline(gen_before, &gamma_before);
            seed_tone_curve(&mut s.braw_color, base);
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
    f32_slider(ui, &mut p.exposure, info.exposure, info.exposure.value, tr("Exposure (stops)"), 2, None);
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
    // The SDK applies the tone curve ONLY under its "Blackmagic Design
    // Custom" gamma (measured: under Film / Video / Extended Video / Rec.709
    // every tone-curve attribute is ignored). The sliders show the curve
    // the CURRENT gamma is built from (the SDK's preset for it, e.g.
    // Extended Video = contrast 1.58, saturation 1.4, midpoint 0.409,
    // shadow rolloff 1.25 on the PYXIS), and — like Resolve's Camera Raw
    // panel — moving one switches the gamma to Custom seeded with that
    // preset, so the look continues from where it was and only the moved
    // control changes.
    let gen_now = p.effective_gen(info);
    let has_custom = info.for_gen(gen_now).has_custom_gamma();
    let gamma_now = p.effective_gamma(info).to_string();
    let on_custom = gamma_now == vr180_braw::CUSTOM_GAMMA;
    let base = info.tone_baseline(gen_now, &gamma_now);
    ui.label(RichText::new(tr("Tone curve")).strong());
    if !has_custom {
        // No Custom gamma in this generation (Gen 6): the SDK ignores every
        // tone-curve attribute, so the controls would do nothing.
        ui.label(RichText::new(tr(
            "This colour science has no adjustable (Custom) tone curve — the SDK ignores these controls under it. Pick Gen 5 to use the tone curve."
        )).small().color(Color32::GRAY));
    } else if !on_custom {
        ui.label(RichText::new(tr(
            "Shown: the curve this gamma is built from. The SDK applies edits under its \"Blackmagic Design Custom\" gamma only — moving a control switches to it, continuing from this curve (the clip's Film LUT goes off)."
        )).small().color(Color32::GRAY));
    }
    ui.add_enabled_ui(has_custom, |ui| {
    let tone_before = (p.saturation, p.contrast, p.midpoint, p.highlights, p.shadows, p.black_level, p.white_level, p.video_black_level);
    f32_slider(ui, &mut p.saturation, info.saturation, base.saturation, tr("Saturation"), 2, None);
    f32_slider(ui, &mut p.contrast, info.contrast, base.contrast, tr("Contrast"), 2, None);
    f32_slider(ui, &mut p.midpoint, info.midpoint, base.midpoint, tr("Midpoint"), 2,
        Some(tr("Where the curve's middle sits — the strongest overall brightness control.")));
    f32_slider(ui, &mut p.highlights, info.highlights, base.highlights, tr("Highlight rolloff"), 2,
        Some(tr("Knee shape of the brightest few percent only — lower = softer (highlights compressed), higher = harder. A subtle control by design; for a level change use White level.")));
    f32_slider(ui, &mut p.shadows, info.shadows, base.shadows, tr("Shadow rolloff"), 2,
        Some(tr("Toe shape of the darkest few percent only — lower = softer, which lifts the deepest shadows; higher = harder, which pushes them down. A subtle control by design; for a level change use Black level.")));
    f32_slider(ui, &mut p.black_level, info.black_level, base.black_level, tr("Black level"), 2, None);
    f32_slider(ui, &mut p.white_level, info.white_level, base.white_level, tr("White level"), 2, None);
    flag(ui, &mut p.video_black_level, base.video_black_level as u16, tr("Video black level"));
    let tone_after = (p.saturation, p.contrast, p.midpoint, p.highlights, p.shadows, p.black_level, p.white_level, p.video_black_level);
    if tone_after != tone_before && has_custom && !on_custom {
        // Keep every untouched control at the gamma's own value (explicitly,
        // since Custom would otherwise fall back to the clip's flat curve),
        // then switch. Custom is display-referred: the Film→video LUT no
        // longer fits.
        seed_tone_curve(p, base);
        p.gamma = Some(vr180_braw::CUSTOM_GAMMA.to_string());
        if let Some(lut) = braw_lut {
            if s.lut_path == lut { s.lut_path.clear(); }
        }
    }
    });
}

/// Make the tone-curve overrides that are still "as shot" explicit at the
/// values of `base`, so the Custom gamma reproduces the curve the user was
/// looking at (the moved control keeps its new value).
fn seed_tone_curve(p: &mut vr180_braw::BrawColorParams, base: vr180_braw::TonePreset) {
    p.saturation.get_or_insert(base.saturation);
    p.contrast.get_or_insert(base.contrast);
    p.midpoint.get_or_insert(base.midpoint);
    p.highlights.get_or_insert(base.highlights);
    p.shadows.get_or_insert(base.shadows);
    p.black_level.get_or_insert(base.black_level);
    p.white_level.get_or_insert(base.white_level);
    p.video_black_level.get_or_insert(base.video_black_level);
}

/// The colour of a place where frames are missing (file info, timeline).
const DROPPED_COLOR: Color32 = Color32::from_rgb(255, 120, 70);

/// The file info's word on the frames a clip's camera dropped.
fn dropped_frames_label(holes: &[vr180_pipeline::frame_timeline::Hole], zh: bool) -> String {
    let frames: usize = holes.iter().map(|h| h.frames).sum();
    match (holes.len(), zh) {
        (0, true) => "无".to_string(),
        (0, false) => "none".to_string(),
        (places, true) => format!("{frames} 帧，{places} 处"),
        (1, false) => format!("{frames} in 1 place"),
        (places, false) => format!("{frames} in {places} places"),
    }
}

/// One place where frames are missing: when, and how many.
fn dropped_hole_text(hole: &vr180_pipeline::frame_timeline::Hole, zh: bool) -> String {
    match (zh, hole.frames) {
        (true, n) => format!("{}：丢 {n} 帧", format_time(hole.at_s)),
        (false, 1) => format!("{}: 1 frame dropped", format_time(hole.at_s)),
        (false, n) => format!("{}: {n} frames dropped", format_time(hole.at_s)),
    }
}

/// The file info line's explanation: what a dropped frame is here, then the
/// places (the first [`DROPPED_LISTED`] of them).
fn dropped_frames_hover(holes: &[vr180_pipeline::frame_timeline::Hole], zh: bool) -> String {
    if holes.is_empty() {
        return if zh { "这个片段的画面轨道里没有缺帧。" } else { "No frame is missing from this clip's picture track." }.to_string();
    }
    let mut text = if zh {
        "相机没有记录下来的帧。预览在每一处停在前一帧上；导出时重复那一帧，画面和声音保持同步。时间线上以橙色标出。".to_string()
    } else {
        "Frames the camera did not record. The preview holds the frame before each place; an export repeats that frame, so picture and sound stay in step. Marked in orange on the timeline.".to_string()
    };
    for hole in holes.iter().take(DROPPED_LISTED) {
        text.push('\n');
        text.push_str(&dropped_hole_text(hole, zh));
    }
    if holes.len() > DROPPED_LISTED {
        let more = holes.len() - DROPPED_LISTED;
        text.push('\n');
        text.push_str(&if zh { format!("……另有 {more} 处") } else { format!("… and {more} more") });
    }
    text
}

/// Places the file info's hover text lists one by one.
const DROPPED_LISTED: usize = 24;

/// Bit rate text for the file info. The video streams' total — and, with
/// `per_stream`, each eye's stream of a dual-stream file in brackets
/// (`253.3 Mbps (126.6 + 126.7)`); `≈ … Mbps` from the file size when the
/// container states no stream rate; `None` when neither is known (a still).
fn bitrate_label(video_bps: &[u64], file_bps: Option<u64>, per_stream: bool) -> Option<String> {
    let mbps = |b: u64| b as f64 / 1e6;
    match video_bps {
        [] => file_bps.map(|b| format!("≈ {:.1} Mbps", mbps(b))),
        many if per_stream && many.len() > 1 => Some(format!(
            "{:.1} Mbps ({})",
            mbps(many.iter().sum()),
            many.iter().map(|b| format!("{:.1}", mbps(*b))).collect::<Vec<_>>().join(" + "),
        )),
        some => Some(format!("{:.1} Mbps", mbps(some.iter().sum()))),
    }
}

/// Menu name of an X6 clip's recording mode for the UI: `None` for the
/// factory modes; an unconfirmed 3712² clip says so until the user picks.
fn x6_mode_label(g: &vr180_fisheye::InsvStreamGeometry) -> Option<String> {
    use vr180_fisheye::X6Mode as M;
    match g.detected {
        M::Stock => None,
        M::Unconfirmed7K4 => Some(format!("7.4K60 · {}", tr("view to be confirmed"))),
        M::Unknown => Some(tr("unrecognized X6 mode").to_string()),
        m => m.label().map(str::to_string),
    }
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
