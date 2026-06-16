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
    /// Texture-view format used to register our Rec.709-gamma SBS with egui.
    /// Always `Rgba8Unorm` (pass-through): egui-wgpu 0.34 samples display
    /// textures as gamma/raw bytes ("NOT sRGB-aware"), so an sRGB *view*
    /// would over-darken the preview to `sRGB⁻¹(v)`. See the constructor.
    preview_view_format: wgpu::TextureFormat,

    // ─── User-visible state ──────────────────────────────────────
    loaded_path: Option<PathBuf>,
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
    /// True from `spawn_decoder` until the decoder delivers its FIRST frame.
    /// While set, play/seek respawns are ignored — clicking Play repeatedly
    /// during a slow load would otherwise pile up detached decoder threads
    /// all contending for the same file (the "click play too fast" freeze).
    decoder_starting: bool,
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

/// Output resolution target for the export. `Native` renders at the
/// source's per-eye dimensions (OSV: 3840² per eye → 7680×3840 SBS).
/// `R8k` renders the PROJECTION at 4096² per eye → 8192×4096 SBS —
/// the projection kernel samples the native-res source directly into
/// the 4096 grid (one resample, end-to-end at output res; sharpen /
/// color / encode all run at 4096), NOT a finished-frame upscale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportResolution { Native, R8k }

impl ExportResolution {
    fn label(self) -> &'static str {
        match self {
            ExportResolution::Native => "Native (source)",
            ExportResolution::R8k    => "8192 × 4096 (8K)",
        }
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
    /// Output resolution target (Native / 8192×4096).
    resolution: ExportResolution,
    /// H.265 average bitrate in Mbps. Range 20..=500.
    h265_bitrate_mbps: u32,
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
    /// Camera baseline in mm for APMP `cams/blin`. Typical 60..=70.
    apmp_baseline_mm: f32,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            codec: ExportCodec::H265,
            resolution: ExportResolution::Native,
            h265_bitrate_mbps: 200,
            h265_bit_depth: 10,
            h265_hardware: true,
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
        "Output" => "输出",
        "View" => "视图",
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
        "Codec" => "编码",
        "Bitrate (Mbps)" => "码率 (Mbps)",
        "Bit depth" => "位深",
        "ProRes profile" => "ProRes 配置",
        "VR180 metadata" => "VR180 元数据",
        "H.265 bitrate" => "H.265 码率",
        "Encoder" => "编码器",
        "10-bit (Main10)" => "10 位 (Main10)",
        "8-bit (Main)" => "8 位 (Main)",
        "Hardware (NVENC)" => "硬件 (NVENC)",
        "Software (libx265)" => "软件 (libx265)",
        "Camera baseline" => "相机基线",
        "None (no VR180 metadata)" => "无（不写入 VR180 元数据）",
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
        "Browse…" => "浏览…",
        "Clear" => "清除",
        "Use built-in Osmo 360 D-LogM→709" => "使用内置 Osmo 360 D-LogM→709",
        "Use built-in GoPro GP-Log→709" => "使用内置 GoPro GP-Log→709",
        "Reset detail" => "重置细节",
        // View adjust
        "Global (both eyes)" => "全局（双眼）",
        "Yaw (°)" => "偏航 (°)",
        "Pitch (°)" => "俯仰 (°)",
        "Roll (°)" => "翻滚 (°)",
        "Stereo offset (right = +, left = −)" => "立体偏移（右 = +，左 = −）",
        "Tip: select a field and press ↑ / ↓ to step its value precisely." => "提示：选中字段后按 ↑ / ↓ 键可精细微调数值。",
        "Reset to 0" => "重置为 0",
        // Fisheye output / stab / lens
        "Format" => "格式",
        "Swap L↔R eyes" => "交换左右眼",
        "Upside-down mount (180°)" => "倒装 (180°)",
        "IMU phase (ms)" => "IMU 相位 (ms)",
        "Stabilization (VQF 6D, BRAW)" => "防抖 (VQF 6D, BRAW)",
        "Stabilization (DJI camera quats)" => "防抖（DJI 相机四元数）",
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
        "Copy the active clip's settings to every clip OF THE SAME TYPE (OSV→OSV, .360→.360, …) — other camera types keep their own settings. Per-clip trim & IMU phase are kept."
            => "将当前片段的设置复制到所有相同类型的片段（OSV→OSV、.360→.360 ……）— 其他相机类型保留各自的设置。各片段的裁剪与 IMU 相位保持不变。",
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
        // Apply the persisted IMU-phase tuning value to the pipeline global
        // so it's in effect before the first stab / rolling-shutter compute.
        vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(defaults.dji_imu_phase_ms);
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
        Self {
            pipeline,
            egui_renderer: wgpu_state.renderer.clone(),
            preview_view_format,
            loaded_path: None,
            eye_orientation_applied: (defaults.fisheye_swap_eyes, defaults.camera_upside_down),
            clip: None,
            playing: false,
            decoder_alive: false,
            decoder_starting: false,
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
            audio_player: None,
            export_opts: ExportOptions::default(),
            batch: Vec::new(),
            active_clip: None,
            batch_running: false,
            batch_current: None,
            batch_opts: None,
            batch_out_dir: None,
            batch_skip_flag: false,
            batch_run_indices: Vec::new(),
            batch_started_at: None,
            last_batch_summary: None,
            show_export_settings: false,
            lang: Self::load_lang(),
            merge_groups: std::collections::HashMap::new(),
            pending_import: None,
            preview_zoom: 1.0,
            preview_center: egui::vec2(0.5, 0.5),
            full_res_display: None,
            full_res_key: 0,
            full_res_desired_key: 0,
            full_res_rendered_at: std::time::Instant::now(),
            detail_last_key: 0,
            detail_key_changed_at: std::time::Instant::now(),
            detail_cache: None,
            fisheye_last_seeded: None,
            suppress_reseed_once: false,
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

    /// Overall progress of the current run for the export bar:
    /// `(done_frames, total_frames, eta_secs, item_position, item_total)`.
    /// `None` when no run is in flight. ETA uses the run's observed
    /// frames-per-second so far (so it self-corrects as slower NR clips and
    /// faster plain clips complete), falling back to the live encode rate
    /// before the first item finishes.
    fn overall_run_progress(&self) -> Option<(u64, u64, Option<f64>, usize, usize)> {
        if !self.batch_running || self.batch_run_indices.is_empty() {
            return None;
        }
        let item_frames = |i: usize| -> u64 {
            self.batch.get(i)
                .map(|b| (b.duration_sec * b.fps as f64).round().max(1.0) as u64)
                .unwrap_or(1)
        };
        let total_frames: u64 = self.batch_run_indices.iter().map(|&i| item_frames(i)).sum();
        let mut done_frames: u64 = 0;
        let mut item_pos = 0usize;
        for (pos, &i) in self.batch_run_indices.iter().enumerate() {
            match self.batch.get(i).map(|b| &b.status) {
                Some(BatchStatus::Done | BatchStatus::Failed(_) | BatchStatus::Skipped) => {
                    done_frames += item_frames(i);
                }
                Some(BatchStatus::Running) => {
                    item_pos = pos;
                    if let Some(p) = self.export_job.as_ref().and_then(|j| j.last_progress.as_ref()) {
                        done_frames += p.frame_idx.min(item_frames(i));
                    }
                }
                _ => {}
            }
        }
        let remaining = total_frames.saturating_sub(done_frames);
        let eta = if done_frames > 0 {
            self.batch_started_at.map(|t| {
                let spf = t.elapsed().as_secs_f64() / done_frames as f64;
                remaining as f64 * spf
            })
        } else {
            self.export_job.as_ref()
                .and_then(|j| j.last_progress.as_ref())
                .map(|p| p.fps_avg)
                .filter(|f| *f > 0.1)
                .map(|f| remaining as f64 / f as f64)
        };
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
        let color_stack = settings.build_color_stack();

        // Bitrate / bit-depth come from the options window. ProRes
        // ignores the bitrate field (the profile picks the rate)
        // and is always 10-bit / 12-bit per profile.
        let (bitrate_kbps, bit_depth) = match opts.codec {
            ExportCodec::H265 => (
                opts.h265_bitrate_mbps.saturating_mul(1000),
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
            output_path: output_path.to_path_buf(),
            source_kind,
            audio_track: match settings.audio_format {
                crate::decoder::AudioFormat::Stereo    => vr180_pipeline::fisheye_export::AudioTrack::Stereo,
                crate::decoder::AudioFormat::Ambisonic => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
                crate::decoder::AudioFormat::Apac      => vr180_pipeline::fisheye_export::AudioTrack::Apac,
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
            dji_max_corr_deg: settings.dji_max_corr_deg,
            dji_smooth_ms: settings.dji_smooth_ms,
            dji_responsiveness: settings.dji_responsiveness,
            denoise_strength: settings.denoise_strength,
            view_adjust: vr180_pipeline::panomap::ViewAdjust {
                pano_yaw_deg: settings.pano_yaw_deg,
                pano_pitch_deg: settings.pano_pitch_deg,
                pano_roll_deg: settings.pano_roll_deg,
                stereo_yaw_deg: settings.stereo_yaw_deg,
                stereo_pitch_deg: settings.stereo_pitch_deg,
                stereo_roll_deg: settings.stereo_roll_deg,
                upside_down: settings.camera_upside_down,
            },
            fisheye_preset: settings.fisheye_preset.clone(),
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
            trim_in_s: settings.trim_in_s,
            trim_out_s: settings.trim_out_s,
            color_stack,
            projection,
        }
    }

    /// Spawn the export worker thread for a prepared config and install
    /// it as the (single) running `export_job`. `imu_phase_ms` is the
    /// per-clip IMU sample point — written into the pipeline global
    /// before the worker starts (jobs are strictly sequential, so a
    /// per-job global is safe).
    /// `eac_stab`: for a GoPro `.360` (EAC) export, the `(Settings,
    /// n_frames)` needed to compute the per-eye stabilization (the
    /// preview's `build_per_eye_frames`); `None` for fisheye sources,
    /// which compute stab inside `export_fisheye`.
    fn spawn_export_job(
        &mut self,
        cfg: vr180_pipeline::fisheye_export::FisheyeExportConfig,
        imu_phase_ms: f32,
        eac_stab: Option<(crate::decoder::Settings, usize)>,
    ) {
        let output_path = cfg.output_path.clone();

        // Stop preview playback when it reads the SAME file as the export
        // (they'd race on the decoder / file). Previewing a DIFFERENT clip
        // is fine — that's what lets you keep tuning clips mid-batch.
        if self.loaded_path.as_ref() == Some(&cfg.source_path) {
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
            // The IMU-phase store is THREAD-LOCAL — assert this clip's
            // value on the worker itself (stab + per-frame RS read it
            // here, immune to whatever the preview threads set).
            vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(imu_phase_ms);
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
    fn poll_export_job(&mut self) {
        let Some(job) = self.export_job.as_mut() else { return; };
        // Drain the channel.
        while let Ok(p) = job.progress_rx.try_recv() {
            job.last_progress = Some(p);
        }
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
            K::GoProEac       => "eac",
            K::SbsFisheye     => "sbs",
            K::BlackmagicRaw  => "braw",
            _                 => "other",
        }
    }

    /// Remembered settings template for `kind` (defaults if none saved yet).
    fn kind_template(&self, kind: vr180_pipeline::SourceKind) -> Settings {
        self.kind_settings.get(Self::settings_class_key(kind))
            .cloned().unwrap_or_default()
    }

    /// Fold the active clip's live settings into ITS kind's template, so the
    /// next same-kind clip inherits them (kept independent across kinds).
    fn save_active_kind_template(&mut self) {
        if let Some(kind) = self.clip.as_ref().map(|c| c.source_kind) {
            self.kind_settings.insert(
                Self::settings_class_key(kind).to_string(), self.settings.clone());
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
            let probe = match vr180_pipeline::decode::probe_video(&path) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("batch: probe failed for {}: {e}", path.display());
                    continue;
                }
            };
            let (fisheye_eye_w, fisheye_eye_h) = match source_kind {
                vr180_pipeline::SourceKind::SbsFisheye => (probe.width / 2, probe.height),
                vr180_pipeline::SourceKind::DjiOsv     => (probe.width, probe.height),
                vr180_pipeline::SourceKind::BlackmagicRaw => {
                    if let Ok(info) = vr180_braw::BrawInfo::probe(&path) {
                        if info.is_dual_track() { (info.width / 2, info.height) }
                        else { (info.width, info.height) }
                    } else {
                        (probe.width, probe.height)
                    }
                }
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
                settings.dji_imu_phase_ms =
                    vr180_pipeline::dji_imu::dji_imu_phase_default_ms_for_fps(probe.fps);
                // RS mode + readout are per-clip — auto-detect firmware vs
                // no-firmware RS from the GoPro CORI stream (the toggle still
                // overrides). Non-EAC sources fall back to the default.
                settings.rs_mode = crate::decoder::detect_rs_mode_for_path(&path, source_kind);
                settings.rs_readout_ms = crate::decoder::default_rs_readout_ms();
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
    /// preserving its per-clip fields (trim + IMU phase) — "apply my
    /// grade/stab/output choices", not "copy clip A's trim everywhere".
    fn apply_current_settings_to_item(&mut self, idx: usize) {
        let Some(item) = self.batch.get_mut(idx) else { return; };
        if item.status == BatchStatus::Running { return; }
        let keep_trim = (item.settings.trim_in_s, item.settings.trim_out_s);
        let keep_phase = item.settings.dji_imu_phase_ms;
        let keep_rs = (item.settings.rs_mode, item.settings.rs_readout_ms);
        item.settings = self.settings.clone();
        item.settings.trim_in_s = keep_trim.0;
        item.settings.trim_out_s = keep_trim.1;
        item.settings.dji_imu_phase_ms = keep_phase;
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
        let mut out = Self::default_output_for(&item.path, opts.codec);
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
        let (cfg, imu_phase, eac_stab) = {
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
                item.settings.dji_imu_phase_ms,
                eac_stab,
            )
        };
        self.batch[idx].status = BatchStatus::Running;
        self.batch_current = Some(idx);
        tracing::info!("batch: item {}/{} → {}", idx + 1, self.batch.len(), output.display());
        self.spawn_export_job(cfg, imu_phase, eac_stab);
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
        if self.detail_cache.is_none() && self.preview_zoom > 1.001 {
            self.detail_cache = Some(crate::decoder::DetailCache::new(
                path, kind, fps, self.settings.effective_swap_eyes(), ctx.clone()));
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
        let want = self.decoder_alive && !self.playing && self.preview_zoom > 1.001;
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
            self.decoder_starting = false; // first frame in → load done, ungate respawns
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

    fn pick_and_load_file(&mut self) {
        if let Some(paths) = Self::video_file_dialog().pick_files() {
            self.open_paths(paths);
        }
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

        // 1) The OSV multi-selection → one merge candidate (sorted by name,
        //    which for `CAM_<timestamp>_<idx>_…` is chronological = segment
        //    order). A lone OSV falls through to step 2 as a singleton.
        let mut osv: Vec<PathBuf> = paths.iter()
            .filter(|p| matches!(kind_of(p), SourceKind::DjiOsv))
            .cloned().collect();
        osv.sort();
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

    /// `preserve_clip_settings = true` keeps the per-clip fields (trim +
    /// IMU phase) currently in `self.settings` instead of resetting them
    /// — used when re-activating a clip whose own settings were just
    /// restored from the clip list (and for same-clip reloads like the
    /// eye-orientation toggle).
    fn load_file_inner(&mut self, path: PathBuf, preserve_clip_settings: bool) {
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
        if !preserve_clip_settings {
            self.settings.trim_in_s = None;
            self.settings.trim_out_s = None;
        }
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
        let (cori_count, grav_count, srot_ms, detected_rs_mode) = if source_kind == vr180_pipeline::SourceKind::GoProEac {
            match vr180_pipeline::decode::extract_gpmf_stream(&path) {
                Ok(gpmf) => {
                    let cori = vr180_core::gyro::parse_cori(&gpmf);
                    let raw = vr180_core::gyro::parse_raw_imu(&gpmf);
                    let srot = vr180_core::geoc::find_srot_ms(&gpmf)
                        .or_else(|| {
                            vr180_core::geoc::lookup_srot_s(&path, None)
                                .ok().flatten().map(|s| s * 1000.0)
                        });
                    let rs_mode = crate::decoder::detect_rs_mode(&cori);
                    (cori.len(), raw.grav.len(), srot, rs_mode)
                }
                Err(_) => (0, 0, None, crate::decoder::default_rs_mode()),
            }
        } else {
            (0, 0, None, crate::decoder::default_rs_mode())
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
        // Autoload the embedded log→Rec.709 LUT matching the source —
        // DJI D-Log M for OSV, GoPro GP-Log for `.360` (same behavior as
        // the Python app's bundled "Recommended Lut GPLOG") — but only
        // when no LUT is set, so a remembered choice (a different LUT,
        // or a deliberately cleared one within a session) is preserved.
        // If the OTHER source's builtin is set (clip switch), swap it —
        // the log curves are camera-specific. The user can clear or
        // swap it in the Color panel.
        {
            use crate::decoder::{BUILTIN_OSMO_LUT_PATH, BUILTIN_GPLOG_LUT_PATH};
            let wanted = match source_kind {
                vr180_pipeline::SourceKind::DjiOsv   => Some(BUILTIN_OSMO_LUT_PATH),
                vr180_pipeline::SourceKind::GoProEac => Some(BUILTIN_GPLOG_LUT_PATH),
                _ => None,
            };
            let cur_is_builtin = self.settings.lut_path == BUILTIN_OSMO_LUT_PATH
                || self.settings.lut_path == BUILTIN_GPLOG_LUT_PATH;
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
        // Refresh the (non-persisted) IMU-phase to SROT/2 for THIS clip's fps
        // — fps-aware (9.15 ms @30 fps, 8.11 ms @50 fps), re-seeded on every
        // load so a prior clip's tweak never carries over. Set before the
        // snapshot below so the decoder spawns with it and gen isn't bumped
        // on frame 0.
        if !preserve_clip_settings {
            let imu_phase = vr180_pipeline::dji_imu::dji_imu_phase_default_ms_for_fps(probe.fps);
            self.settings.dji_imu_phase_ms = imu_phase;
            // RS mode + readout are per-clip — auto-detected from the CORI
            // stream above (firmware vs no-firmware), re-seeded on each fresh
            // load so a prior clip's tweak never carries over (same lifecycle
            // as the IMU phase above). The UI toggle still overrides it.
            self.settings.rs_mode = detected_rs_mode;
            self.settings.rs_readout_ms = crate::decoder::default_rs_readout_ms();
        }
        vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(self.settings.dji_imu_phase_ms);

        // Snapshot settings so the next spawn_decoder doesn't bump
        // generation unnecessarily on the first frame.
        self.last_pushed_settings = self.settings.clone();

        let segments = self.clip_segments(&path);
        tracing::info!("load_timing: after clip_segments ({} segs) @ {:?}",
            segments.len(), t_load.elapsed());

        // Duration + frame count span the WHOLE segment chain (GS01…,
        // GS02…, …) — the timeline, seek, and trim all run on global
        // clip time. Single-segment: just this file's probe.
        let (duration_sec, frame_count) = if segments.len() > 1 {
            let total_dur: f64 = segments.iter()
                .filter_map(|s| vr180_pipeline::decode::probe_video(s).ok())
                .map(|p| p.duration_sec)
                .sum();
            let total_frames: u32 = segments.iter()
                .filter_map(|s| vr180_pipeline::decode::probe_video(s).ok())
                .map(|p| (p.duration_sec * p.fps as f64).round() as u32)
                .sum();
            tracing::info!(
                "multi-segment: {} chapters → {:.1}s, {} frames total",
                segments.len(), total_dur, total_frames);
            (total_dur, total_frames)
        } else {
            (probe.duration_sec, (probe.duration_sec * probe.fps as f64).round() as u32)
        };

        self.clip = Some(ClipInfo {
            width: probe.width, height: probe.height,
            fps: probe.fps, duration_sec,
            frame_count,
            segments,
            eac_tile_w: dims.tile_w(),
            cori_count, grav_count, srot_ms,
            source_kind,
            fisheye_eye_w, fisheye_eye_h,
        });
        self.loaded_path = Some(path);

        // Drop the zoom-magnifier cache so it can't serve the PREVIOUS clip's
        // decoded native frame / still on the first zoom-in into this one.
        // (poll_full_res only resets these on leaving detail mode, not on a
        // clip swap while still paused+zoomed.)
        if let Some(prev) = self.full_res_display.take() {
            self.egui_renderer.write().free_texture(&prev.egui_id);
        }
        self.detail_cache = None;
        self.full_res_key = 0;
        self.full_res_desired_key = 0;
        self.detail_last_key = 0;

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
            settings: parking_lot::RwLock::new(self.settings.clone()),
            settings_generation: std::sync::atomic::AtomicU64::new(0),
            detected_calib: parking_lot::Mutex::new(None),
            want_detail: std::sync::atomic::AtomicBool::new(false),
            detail_tx: parking_lot::Mutex::new(Some(detail_tx)),
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
        self.detail_frame_rx = Some(detail_rx);
        self.cmd_tx = Some(cmd_tx);
        self.control = Some(control);
        self.playing = !start_paused;
        self.decoder_alive = true;
        self.decoder_starting = true;
        self.last_pushed_settings = self.settings.clone();
        self.fps_stats = FpsStats::default();

        // Spawn the audio player alongside the video decoder. It runs
        // its own thread + cpal output stream and is held by the App
        // for the lifetime of the playback session. Multi-segment: feed
        // it an ffconcat playlist so the chain's audio plays + seeks as
        // one continuous timeline (matches the video chaining).
        if let Some(audio_path) = self.loaded_path.clone() {
            let audio_path = self.clip.as_ref()
                .filter(|c| c.segments.len() > 1)
                .and_then(|c| vr180_pipeline::audio::build_segment_playlist(&c.segments, "preview"))
                .unwrap_or(audio_path);
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
        self.detail_frame_rx = None;
        self.cmd_tx = None;
        self.control = None;
        self.playing = false;
        self.decoder_alive = false;
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
        if detected == self.fisheye_last_seeded {
            return; // unchanged (same clip, or both absent) — nothing to do
        }
        self.fisheye_last_seeded = detected;
        let Some(d) = detected else { return; }; // cleared mid-load → wait for next
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
        let fmt_label = format!("{} · {}{}",
            self.export_opts.codec.label(),
            self.export_opts.resolution.label(),
            match self.export_opts.codec {
                ExportCodec::H265 => format!(" · {} Mbps · {}-bit",
                    self.export_opts.h265_bitrate_mbps, self.export_opts.h265_bit_depth),
                ExportCodec::ProRes => format!(" · {}", self.export_opts.prores_profile.label()),
            });
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
                    });
                }
            });
        });

        if do_choose_dir {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() { self.batch_out_dir = Some(dir); }
        }
        if do_clear_dir { self.batch_out_dir = None; }
        if dismiss_summary { self.last_batch_summary = None; }
        if toggle_settings { self.show_export_settings = !self.show_export_settings; }
        if do_export_selected { self.export_selected(); }
        if do_export_all { self.start_batch(); }
        if do_skip { self.skip_current_batch_item(); }
        if do_stop { self.stop_batch(); }

        // Format settings popover — a real (non-modal) window so ComboBoxes
        // and sliders stay open while you edit them.
        if self.show_export_settings {
            let mut open = true;
            egui::Window::new(tr("Export format"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -150.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(360.0);
                    Self::export_options_ui(ui, &mut self.export_opts, "export");
                });
            self.show_export_settings = open;
        }
    }

    /// The export-output controls (resolution / codec / codec-specific /
    /// VR180 metadata) shown in the export bar's Format dropdown. `id_prefix`
    /// keeps the ComboBox ids distinct.
    fn export_options_ui(ui: &mut egui::Ui, opts: &mut ExportOptions, id_prefix: &str) {
        ui.label(RichText::new(tr("Resolution")).strong());
        egui::ComboBox::from_id_source(format!("{id_prefix}_resolution"))
            .selected_text(opts.resolution.label())
            .width(220.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut opts.resolution,
                    ExportResolution::Native, ExportResolution::Native.label());
                ui.selectable_value(&mut opts.resolution,
                    ExportResolution::R8k, ExportResolution::R8k.label());
            });
        ui.add_space(8.0);

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
                ui.add(egui::Slider::new(&mut opts.h265_bitrate_mbps, 20..=500)
                    .text("Mbps"));
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

}

impl eframe::App for App {
    // eframe 0.34 made `ui` the required method and deprecated `update`.
    // Our UI builds multiple panels (top/side/central) directly from
    // `ctx`, so we keep overriding `update` (eframe's runner still calls
    // it) and leave `ui` empty.
    fn ui(&mut self, _ui: &mut egui::Ui, _frame: &mut eframe::Frame) {}

    #[allow(deprecated)]
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Mirror the chosen language into the process global that `tr` reads.
        set_cur_lang(self.lang);
        // Drain decoder frames first so the most recent texture is
        // ready by the time we render the central panel.
        self.drain_frames(ctx);
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
                // Same clip — keep its trim + IMU phase across the reload.
                self.load_file_inner(p, true);
            }
        }
        // Drain export-job progress + check for completion.
        self.poll_export_job();
        // Drive the full-res still for the zoom magnifier.
        self.poll_full_res(ctx);
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
                    ui.label(RichText::new("▶ VR180 Silver Bullet 2.0")
                        .strong().color(Color32::from_rgb(90, 166, 255)));
                    ui.separator();
                    if ui.button(tr("Load video…")).clicked() {
                        self.pick_and_load_file();
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

                    ui.separator();
                    ui.label(RichText::new(tr("Preview size")).color(Color32::GRAY));
                    egui::ComboBox::from_id_source("preview_w")
                        .selected_text(format!("{}", self.settings.preview_eye_w))
                        .show_ui(ui, |ui| {
                            for &w in &[512_u32, 768, 1024, 1280, 1536] {
                                ui.selectable_value(&mut self.settings.preview_eye_w, w,
                                    format!("{w}"));
                            }
                        });

                    ui.separator();
                    ui.label(RichText::new(tr("View")).color(Color32::GRAY));
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

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(env!("CARGO_PKG_VERSION"))
                            .small().color(Color32::GRAY));

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
                if matches!(kind, Some(k) if k.is_fisheye()) {
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
                    && self.clip.as_ref().map_or(false, |c| c.source_kind.is_exportable());
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
                    let label = tr(if self.playing { "Pause" } else { "▶ Play" });
                    if ui.button(label).clicked() { self.toggle_play_pause(); }
                    if ui.button(tr("■ Stop")).clicked() { self.stop_playback(); }
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
        egui::TopBottomPanel::bottom("timeline").exact_height(34.0).show(ctx, |ui| {
            self.draw_timeline(ui);
        });

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
            let use_full_res = self.preview_zoom > 1.001 && !self.playing
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
    /// Source panel = the CLIP LIST (click to select → preview + sidebar
    /// edit that clip's settings) + the active clip's metadata below.
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
        ui.horizontal(|ui| {
            if ui.small_button(tr("+ Add clips…")).clicked() {
                if let Some(paths) = Self::video_file_dialog().pick_files() {
                    self.open_paths(paths);
                }
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
                                    settings. Per-clip trim & IMU phase are kept."))
                    .clicked()
                {
                    do_apply_all = true;
                }
            }
        });

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
    }

    /// Remove a clip from the list, keeping `active_clip` /
    /// `batch_current` indices coherent. The preview keeps playing the
    /// file even if its list entry goes away (it just loses selection).
    fn remove_clip(&mut self, idx: usize) {
        if idx >= self.batch.len() { return; }
        if self.batch[idx].status == BatchStatus::Running { return; }
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
    }

    /// The shared open-dialog filter for every "add video" entry point.
    fn video_file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new()
            .add_filter(
                "All supported video",
                &["360", "osv", "OSV", "braw", "BRAW",
                  "mp4", "MP4", "mov", "MOV"],
            )
            .add_filter("DJI Osmo 360 (.osv)", &["osv", "OSV"])
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
                ui.horizontal(|ui| {
                    ui.label(RichText::new(tr("FPS")).color(Color32::GRAY).small());
                    ui.label(format!("{:.3}", c.fps));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new(tr("Duration")).color(Color32::GRAY).small());
                    ui.label(format!("{:.2}s ({} frames)", c.duration_sec, c.frame_count));
                });
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
            ui.add(egui::Slider::new(&mut s.smooth_ms, 0.0..=3000.0).text(tr("Smooth (ms)")));
            ui.add(egui::Slider::new(&mut s.max_corr_deg, 0.0..=45.0).text(tr("Max corr (°)")));
            ui.add(egui::Slider::new(&mut s.gyro_responsiveness, 0.2..=3.0)
                .text(tr("Response")))
                .on_hover_text(tr("Velocity response curve: <1 follows motion early, \
                    1 linear, >1 holds longer then catches up (cinematic lag)."));
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
        let s = &mut self.settings;
        ui.horizontal(|ui| {
            ui.label(tr("Export track"));
            egui::ComboBox::from_id_source("audio_format")
                .selected_text(s.audio_format.as_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.audio_format, AF::Stereo, AF::Stereo.as_str());
                    ui.selectable_value(&mut s.audio_format, AF::Ambisonic, AF::Ambisonic.as_str());
                    ui.selectable_value(&mut s.audio_format, AF::Apac, AF::Apac.as_str());
                });
        });
        let hint = match s.audio_format {
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
        let s = &mut self.settings;

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

        ui.separator();

        // 3D LUT.
        ui.label(RichText::new(tr("3D LUT (.cube)")).strong());
        ui.horizontal(|ui| {
            let mut label = if s.lut_path.is_empty() {
                "(none)".to_string()
            } else if s.lut_path == crate::decoder::BUILTIN_OSMO_LUT_PATH {
                crate::decoder::BUILTIN_OSMO_LUT_NAME.to_string()
            } else if s.lut_path == crate::decoder::BUILTIN_GPLOG_LUT_PATH {
                crate::decoder::BUILTIN_GPLOG_LUT_NAME.to_string()
            } else {
                std::path::Path::new(&s.lut_path)
                    .file_name().and_then(|f| f.to_str()).unwrap_or("(invalid)")
                    .to_string()
            };
            ui.add(egui::TextEdit::singleline(&mut label).interactive(false).desired_width(160.0));
            if ui.button(tr("Browse…")).clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Cube LUT", &["cube", "CUBE"])
                    .pick_file()
                {
                    s.lut_path = p.to_string_lossy().to_string();
                }
            }
            if ui.button(tr("Clear")).clicked() {
                s.lut_path.clear();
            }
        });
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
        let s = &mut self.settings;
        ui.label(RichText::new(tr("Global (both eyes)")).small().color(Color32::GRAY));
        if zoom > 1.001 {
            ui.label(RichText::new(format!("fine drag ×{:.0} (zoomed)", zoom))
                .small().color(Color32::from_rgb(120, 180, 120)));
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
        // Stereo offset: Up/Down arrows step 0.01° per press.
        let stereo_key_step = 0.01;
        fine_slider(ui, zoom, &mut s.stereo_yaw_deg, -10.0..=10.0, tr("Yaw (°)"), 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_pitch_deg, -10.0..=10.0, tr("Pitch (°)"), 3, rot_fine, stereo_key_step);
        fine_slider(ui, zoom, &mut s.stereo_roll_deg, -10.0..=10.0, tr("Roll (°)"), 3, rot_fine, stereo_key_step);
        ui.label(RichText::new(tr("Tip: select a field and press ↑ / ↓ to step its value precisely."))
            .small().color(Color32::from_rgb(130, 150, 175)));
        ui.add_space(4.0);
        if ui.button(tr("Reset to 0")).clicked() {
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
                tr("No gyro stabilization available for this source yet.")
            ).small().color(Color32::GRAY));
            return;
        }
        let stab_label = if is_braw {
            tr("Stabilization (VQF 6D, BRAW)")
        } else {
            tr("Stabilization (DJI camera quats)")
        };
        ui.checkbox(&mut s.stabilize, stab_label);
        ui.add_enabled_ui(s.stabilize, |ui| {
            if is_osv {
                // smooth_ms = 0 → sharp camera-lock (legacy).
                // smooth_ms > 0 → soft-stab (GoPro-style).
                ui.add(egui::Slider::new(&mut s.dji_smooth_ms, 0.0..=3000.0)
                    .text(tr("Smooth (ms)")));
                ui.add(egui::Slider::new(&mut s.dji_max_corr_deg, 0.0..=45.0)
                    .text(tr("Max corr (°)")));
                // Velocity→smoothing response curve (Python "Response").
                // <1 follows motion early; >1 holds longer, then catches up.
                ui.add(egui::Slider::new(&mut s.dji_responsiveness, 0.2..=3.0)
                    .fixed_decimals(1)
                    .text(tr("Response")));
                // IMU sample timing is fixed at SROT/2 (readout midpoint),
                // re-seeded per clip from its own fps — no manual override, so
                // the slider is hidden. Still assert the main-thread thread-local
                // so paused detail-still renders use this clip's phase.
                vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(s.dji_imu_phase_ms);
            }
        });
    }

    /// Output panel for fisheye sources — projection target + L↔R swap.
    fn draw_fisheye_output_panel(&mut self, ui: &mut egui::Ui) {
        let is_osv = matches!(
            self.clip.as_ref().map(|c| c.source_kind),
            Some(vr180_pipeline::SourceKind::DjiOsv)
        );
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
                });
        });

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
        // k5 (5th radial coeff) only applies to DJI OSV's 5-coefficient
        // model — gate the override slider on the loaded clip being OSV.
        let is_osv = self.clip.as_ref()
            .map(|c| c.source_kind == vr180_pipeline::SourceKind::DjiOsv)
            .unwrap_or(false);
        let s = &mut self.settings;

        // ── Camera preset — hidden for OSV: the .osv file carries the full
        //    per-lens calibration, so a camera/profile picker is redundant.
        if !is_osv {
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
                        .filter(|p| p.name == "DJI Osmo 360" || p.name == "Custom")
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
        if !is_osv {
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
        } // hide Gyroflow loader for OSV

        ui.add_space(6.0);
        ui.label(RichText::new(tr(
            "With Override off, each eye uses the in-file (OSV) / preset \
             calibration. Turn Override on for an eye to set its FOV, \
             principal point and distortion by hand."
        )).small().color(Color32::GRAY));

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

        // KB distortion (zoom-scaled fine drag). k5 only for OSV (DJI's
        // 5-coefficient model); other cameras use the 4-coeff KB.
        let k_title = "KB parameters";
        ui.collapsing(k_title, |ui| {
            for (i, ki) in k.iter_mut().enumerate() {
                fine_slider(ui, zoom, ki, -0.5..=0.5, &format!("k{}", i + 1), 6, 1.0, 0.0);
            }
            if is_osv {
                // 5th radial coeff (θ¹¹ term). Keeps the projection monotonic
                // past ~90° out to the full lens FOV.
                fine_slider(ui, zoom, k5, -0.05..=0.05, "k5", 6, 1.0, 0.0);
                // Brown-Conrady tangential (decentering) — small but visible at
                // the rim. Seeded from the file (field 20) when Override engages.
                fine_slider(ui, zoom, &mut p[0], -0.01..=0.01, "p1 (tangential)", 6, 1.0, 0.0);
                fine_slider(ui, zoom, &mut p[1], -0.01..=0.01, "p2 (tangential)", 6, 1.0, 0.0);
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
