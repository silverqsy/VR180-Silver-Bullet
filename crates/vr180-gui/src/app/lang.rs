//! The interface's language: the English strings and their Chinese text, and the font that draws it.

use super::*;

/// UI language. English is the source of truth; `tr` maps the core chrome
/// strings to Chinese, falling back to the English literal for anything not
/// in the table (so untranslated strings degrade gracefully).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lang {
    En,
    Zh,
}

/// Current UI language, stored as a process global so `tr` can be called
/// anywhere (in any egui closure) without threading the language through.
/// Set once per frame at the top of `update` from `self.lang`.
pub(super) static UI_LANG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(super) fn cur_lang() -> Lang {
    if UI_LANG.load(std::sync::atomic::Ordering::Relaxed) == 1 { Lang::Zh } else { Lang::En }
}
pub(super) fn set_cur_lang(l: Lang) {
    UI_LANG.store(if l == Lang::Zh { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
}

/// The toolbar switch that shows the preview as Apple's players will show
/// the export (`display_colour`), and what it says about itself.
pub(super) const MATCH_PLAYBACK_LABEL: &str = "Match QuickTime";
pub(super) const MATCH_PLAYBACK_HOVER: &str = "Show the preview the way Apple's players (QuickTime, Safari, Apple Vision Pro) show the exported file: read with Apple's Rec.709 video curve, so the mid-tones come out a little brighter. Off, the picture is shown as sRGB, which is closer to most players on Windows. Only the preview changes: the exported file is the same either way.";

/// What the Upside-down mount switch says about itself.
pub(super) const UPSIDE_DOWN_HOVER: &str = "The camera was mounted upside down. The picture is turned upright, the two eyes change sides, and the sound is turned with them. The stereo alignment stays with its lens and the view angles act as you see them, so one set of values serves both ways the camera can be mounted. An X6 clip or raw take sets this by itself from its accelerometer.";

/// What the export bar's "Join into one file" says about itself.
pub(super) const JOIN_HOVER: &str = "Export the clips in the order of the list and write them as ONE file, one after the other, with hard cuts. Every clip keeps its own trim and settings. They have to come out at the same frame size and frame rate (pick one size for all of them in Format). The file's sound is the first clip's Audio choice: stereo, or Ambisonic / APAC when every clip has four-channel sound. Clips in the Lens Space view through the camera's lens are joined as an Apple Immersive Video file (.aivu, macOS 26 or later, for Apple Vision Pro): it holds a lens for every clip, so clips of different recording modes and cameras can share it. Drag a clip by its handle (☰) to change the order.";
/// And the toolbar's Project menu.
pub(super) const PROJECT_HOVER: &str = "The clip list, with every clip's trim and settings and the export choices, as a file to save and open. The list also comes back by itself the next time the app starts.";

/// Translate a (static English) UI string for the current language.
pub(super) fn tr(en: &'static str) -> &'static str {
    if cur_lang() == Lang::En { en } else { tr_zh(en) }
}

/// The Chinese text of a UI string: the English literal for anything not in
/// the table.
pub(super) fn tr_zh(en: &'static str) -> &'static str {
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
        "DNG sequence" => "DNG 序列",
        "Archive copy" => "存档副本",
        "RAW Footage Archiving" => "RAW 素材存档",
        "Compressed archive" => "压缩存档",
        "Camera original DNGs" => "相机原片 DNG",
        "The take's folder again with a smaller capture, which opens like the take: every frame, its timing and the files beside it are kept. The take itself is left as it is."
            => "再写一份素材文件夹，采集文件更小，打开方式和原素材一样：每一帧、它的时间信息和旁边的文件都保留。原素材保持不动。",
        "Every sample of the picture is kept as the camera recorded it, and each frame is read back and compared before it is written. Most takes come out a quarter to two fifths smaller. A take that is noise for the most part gains little: where its copy would not be smaller, none is made."
            => "画面内每个样本都和相机记录的一样，每一帧写入前都会读回来逐一核对。大多数素材能小四分之一到五分之二。以噪声为主的素材收益很小：副本不会更小时不制作。",
        "The copy is lossy: at 1 bit a pixel a frame is off by less than its own noise; below that the shadows lose their grain when they are lifted."
            => "副本是有损的：每像素 1 bit 时，一帧的偏差小于它自身的噪声；再低，暗部提亮后会失去颗粒。",
        "A copy of a stretch is a shorter clip. What the app takes from a whole clip, such as the tone statistics and the stabilizer's reference, is then taken from the stretch, so its picture can differ slightly from the same stretch of the take."
            => "只含一段的副本是一条更短的素材。软件按整条素材取的量，比如影调统计和防抖基准，会改成从这一段里取，画面可能与原素材的同一段略有不同。",
        "The folder of that name beside the take is also where the app looks for enhanced frames and for Camera Raw settings."
            => "素材旁边这个名字的文件夹，也是软件查找增强帧和 Camera Raw 设置的地方。",
        "This clip is a lossy archive copy: the frames written are the copy's, close to what the sensor recorded but not the same."
            => "这个片段是有损存档副本：写出的是副本里的帧，接近传感器的记录，但不完全相同。",
        "Lossy archive copies are among the ticked clips: the frames written of those are the copies', close to what the sensor recorded but not the same."
            => "勾选的片段里有有损存档副本：它们写出的是副本里的帧，接近传感器的记录，但不完全相同。",
        "This take" => "当前素材",
        "Ticked takes" => "勾选的素材",
        "The take can now be moved away or deleted. That is yours to do: the app moves and deletes nothing. Once the take is gone the app reads its copy in its place, as long as the copy stays beside where the take was, or in the output folder."
            => "原素材现在可以移走或删除。这一步由你来做，软件不移动也不删除任何东西。原素材不在以后，软件会改读它的副本，前提是副本留在原素材原来位置的旁边，或者在输出文件夹里。",
        "The take this clip belongs to is not at hand: its archive copy is read in its place. When the take is back, it is read again."
            => "这条素材的原素材不在，现在读的是它的存档副本。原素材回来以后会重新读原素材。",
        "In place of" => "顶替原素材",
        "A run is in flight: this can start when it is over." => "有任务正在进行：结束后才能开始。",
        "Range" => "范围",
        "In to Out" => "入点到出点",
        "Whole take" => "整条素材",
        "every sample kept" => "每个样本都保留",
        "This clip is an archive copy of a take, made by this app: not the camera's own recording." => "这个片段是素材的存档副本，由本软件制作，不是相机自己的记录。",
        "High quality: 1 bit a pixel" => "高质量：每像素 1 bit",
        "Lossless: every sample of the picture kept" => "无损：画面内每个样本都保留",
        "Custom" => "自定义",
        "bit a pixel" => "bit 每像素",
        "A smaller copy of the take for keeping, which opens like the take." => "素材的一个更小的副本，用于存档，打开方式和原素材一样。",
        "This clip is an archive copy of a take: a copy is made from the camera's take, not from another copy."
            => "这个片段本身就是素材的存档副本：副本要从相机的原素材制作，不能从另一个副本制作。",
        "Use enhanced frames" => "使用增强后的帧",
        "Enhanced frames found:" => "已找到增强后的帧：",
        "No enhanced frame found for this clip: it exports from the capture." => "没有找到这个片段的增强帧：导出取自采集文件。",
        "Looking for enhanced frames…" => "正在查找增强后的帧…",
        "This size is the stabilized frame scaled to fit the NVIDIA hardware encoder, which takes 8192 pixels a side: a few percent smaller than recorded, and many times faster to export as H.265 than the native size, which is encoded in software." => "这个尺寸是把防抖后的画幅缩小到 NVIDIA 硬件编码器能接受的范围（单边 8192 像素）：比拍摄时小百分之几，导出 H.265 比原生尺寸快很多倍（原生尺寸只能软件编码）。",
        "Enhanced frames (a raw converter's Super Resolution, in the folder of Camera original DNGs) are read on macOS only for now: this clip exports from the capture." => "增强后的帧（RAW 转换软件的超分辨率，位于相机原片 DNG 文件夹中）目前只有 macOS 能读取：这个片段从采集文件导出。",
        "This clip's enhanced frames hold more than 8192 × 4096 shows: the two larger sizes keep it. A stretch without enhanced frames exports at the size picked without them." => "这个片段的增强帧包含的细节多于 8192 × 4096 能呈现的：两档更大的尺寸可以保留这些细节。没有增强帧的区段按未使用增强帧时所选的尺寸导出。",
        "This clip's enhanced frames hold more than its own size shows: the two larger sizes keep it. At these the frame is scaled to the size picked, the stabilizer's margin inside it. A stretch without enhanced frames exports at the size picked without them." => "这个片段的增强帧包含的细节多于片段自身尺寸能呈现的：两档更大的尺寸可以保留这些细节。选这两档时画幅缩放到所选尺寸，防抖余量也包含在这个尺寸之内。没有增强帧的区段按未使用增强帧时所选的尺寸导出。",
        "When the folder of Camera original DNGs is beside the take and a raw converter has enhanced its frames in place (Camera Raw's Super Resolution), each frame is taken from there, at the enhanced size. A frame that was not enhanced comes from the capture." => "素材旁边有相机原片 DNG 文件夹，并且其中的帧已在 RAW 转换软件里原地增强过时（Camera Raw 的超分辨率），每一帧从那里读取，按增强后的尺寸处理。没有增强过的帧仍取自采集文件。",
        "Every frame as the sensor recorded it: one Bayer DNG per eye, named by the camera's frame number. No projection, no stabilization, no grading — for running the frames through a raw converter first." => "每一帧按传感器记录的原样导出：每眼一个 Bayer DNG，以相机的帧号命名。不投影、不防抖、不调色，适合先把每一帧交给 RAW 转换软件处理。",
        "Every frame as a linear DNG in a folder, with this clip's raw development settings inside each file." => "每一帧导出为一个线性 DNG，放进同一个文件夹，每个文件里带着这个片段的 RAW 显影设置。",
        "📷 Snapshot" => "📷 截图",
        "Original DNG" => "原片 DNG",
        "Save this frame as the sensor recorded it: one Bayer DNG per eye, where the snapshot goes and under the same name (Shift+S)" => "把当前帧按传感器记录的原样保存：每眼一个 Bayer DNG，与截图放在同一处、用同样的名字（Shift+S）",
        "Writing the camera's frame" => "正在写出相机原片",
        "Camera original frame failed" => "相机原片导出失败",
        "Save this frame as a JPEG in the export folder (S)" => "把当前帧存为 JPEG 到导出文件夹（S）",
        "Save this frame as a JPEG and as a linear DNG in the export folder (S)" => "把当前帧存为 JPEG 和线性 DNG 到导出文件夹（S）",
        "Nothing to snapshot yet" => "还没有可截图的画面",
        "Saved" => "已保存",
        "Rendering snapshot" => "正在渲染截图",
        "preview size — full-size still never arrived" => "预览尺寸——未能获得全尺寸画面",
        "Snapshot failed" => "截图失败",
        "3D display" => "3D 显示器",
        "Show the stereo pair on a side-by-side 3D monitor or AR glasses that appear as one wide screen (3840×1080): left eye in the left half, right eye in the right. Goes fullscreen on that screen automatically when one is connected; otherwise opens a window to drag onto it (F = fullscreen). Esc closes it." => "在以单个宽屏（3840×1080）形式出现的左右格式 3D 显示器或 AR 眼镜上显示立体画面：左半为左眼，右半为右眼。检测到这样的屏幕时自动在其上全屏；否则打开一个窗口，拖到该屏幕后按 F 全屏。按 Esc 关闭。",
        "SBS while the 3D display is on" => "3D 显示器开启时固定为 SBS",
        MATCH_PLAYBACK_LABEL => "匹配 QuickTime",
        MATCH_PLAYBACK_HOVER => "按苹果播放器（QuickTime、Safari、Apple Vision Pro）显示导出文件的方式显示预览：用苹果的 Rec.709 视频曲线解读画面，中间调会略亮一些。关闭时画面按 sRGB 显示，更接近 Windows 上多数播放器的效果。只改变预览，导出的文件两种情况下完全相同。",
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
        "DNG output: the Tone section (Exposure to Blacks), the 3D LUT, sharpening and automatic eye matching cannot travel as DNG metadata — hidden, and not applied to the preview." => "DNG 输出：“影调”一组（曝光到黑色色阶）、3D LUT、锐化和自动眼间匹配无法作为 DNG 元数据保存——已隐藏，预览中也不应用。",
        "Still" => "照片",
        "Your grade travels as metadata: white balance → AsShotNeutral, lift / gamma / gain → profile tone curve, saturation → profile look table (Adobe); per-eye trims are baked. Not carried: the Tone section, 3D LUT, sharpen." => "调色以元数据形式写入：白平衡 → AsShotNeutral，提升 / 伽马 / 增益 → 配置文件色调曲线，饱和度 → 配置文件 Look 表（Adobe）；双眼微调会烘入像素。不带入：“影调”一组、3D LUT、锐化。",
        "single photo, 14-bit raw" => "单张照片，14 位 RAW",
        "DNG keeps the camera-native linear data with the raw colour tags — grade it in a raw developer." => "DNG 保留相机原生线性数据和 RAW 色彩标签——可在 RAW 软件里继续调色。",
        "Still photo — no stabilization. Use the View adjust / stereo controls to level and align." => "静态照片——没有防抖。用视角调整 / 立体控制来调平和对齐。",
        "Codec" => "编码",
        "Bitrate (Mbps)" => "码率 (Mbps)",
        "Bit depth" => "位深",
        "ProRes profile" => "ProRes 配置",
        "VR180 metadata" => "VR180 元数据",
        "Lens Space" => "镜头空间",
        "Lens" => "镜头",
        "Camera lens" => "相机镜头",
        "Equidistant" => "等距",
        "Each eye is written the way the camera recorded it, through one lens that the file describes (Apple Projected Media Profile, parametric immersive projection). A fisheye clip keeps everything its lens saw, past 180° where the lens goes that far; a Crop clip fills its frame instead of sitting in a canvas that is mostly black. Stabilization, rolling shutter, eye alignment, the crop mask and colour are applied as always. Apple Vision Pro plays it; other headsets and YouTube need the VR180 format."
            => "按相机拍摄时的样子写出每只眼，并在文件里写明所用的同一个镜头（Apple Projected Media Profile 的参数化沉浸投影）。鱼眼片段保留镜头拍到的全部视角，镜头超过 180° 的部分也在；Crop 片段则填满画幅，不再放在大半是黑色的画布中央。防抖、卷帘快门校正、双眼对齐、裁切遮罩和调色照常生效。Apple Vision Pro 可以播放；其他头显和 YouTube 需要 VR180 格式。",
        "An ideal equidistant fisheye: a disc in a square frame, the form that players and tools for fisheye video expect. With the Apple Vision Pro metadata on, the file states this lens too, and the headset shows the disc at its true angles."
            => "理想的等距鱼眼：正方形画幅里的一个圆盘，支持鱼眼视频的播放器和后期工具认的就是这种形式。开启 Apple Vision Pro 元数据时，文件里也会写明这个镜头，头显按真实角度显示圆盘。",
        "The recording's own frame through the camera's lens: its whole field of view, for Apple Vision Pro."
            => "透过相机镜头的拍摄画幅本身：保留镜头的完整视角，供 Apple Vision Pro 使用。",
        "equidistant disc, for players and tools that take fisheye video."
            => "等距圆盘，供支持鱼眼视频的播放器和后期工具使用。",
        "Lens Space, camera lens: the frame is the recording's own, at the density it was recorded, so the clip's own size already holds all of its detail, and nothing the lens saw is cut off at 180°. It plays immersively on Apple Vision Pro (visionOS 26 or later) and nowhere else: for other headsets and YouTube, export the clip in the VR180 format as well. The file is named …_Lens."
            => "镜头空间，相机镜头：画幅就是拍摄时的画幅，密度与拍摄时相同，所以片段自身的尺寸已经包含全部细节，镜头拍到的内容也不会在 180° 处被裁掉。只有 Apple Vision Pro（visionOS 26 或更高版本）能沉浸播放；其他头显和 YouTube 请再用 VR180 格式导出一次。文件名以 _Lens 结尾。",
        "Stabilization: the frame has no spare canvas, so a stabilized frame is 5 % larger on each side than the recording, the same for every clip (an 8K copy keeps its size and scales the picture instead). Where the picture does not reach, the frame is black, as around the picture of a VR180 export; a shift beyond the margin is cut at the frame's edge. With camera lock the picture leaves the frame as the camera turns: keep that for a camera that stays put."
            => "防抖：画幅没有多余的画布，所以开启防抖时画幅每边比拍摄画幅大 5%，每条片段都一样（8K 缩小版保持尺寸不变，改为缩小画面）。画面没有覆盖到的地方是黑色，和 VR180 导出里画面四周一样；超出余量的偏移会在画幅边缘被裁掉。开启相机锁定时，相机一转画面就会移出画幅，只适合机位不动的镜头。",
        "A Lens Space file through the camera's lens carries its own metadata: the lens, for Apple Vision Pro. The choice of VR180 metadata applies to the clips that export as VR180."
            => "透过相机镜头写出的镜头空间文件自带元数据：写给 Apple Vision Pro 的镜头定义。VR180 元数据的选择只作用于按 VR180 导出的片段。",
        "This clip exports an equidistant disc, which is not VR180: with Apple Vision Pro chosen the file states the disc's own lens instead, and for YouTube, which has no fisheye form, nothing is written."
            => "这条片段导出的是等距圆盘，不是 VR180：选择 Apple Vision Pro 时，文件里写的是圆盘自己的镜头定义；YouTube 没有鱼眼格式，选它时不写任何元数据。",
        "H.265 bitrate" => "H.265 码率",
        "Spatial video (MV-HEVC)" => "空间视频 (MV-HEVC)",
        "per eye" => "每眼",
        "Bitrate (both eyes)" => "码率（双眼合计）",
        "Field of view written" => "写入的水平视角",
        "Disparity adjustment" => "视差调整",
        "Disparity adjustment moves the whole scene away from you (positive) or towards you. 2 % keeps the far background just behind the window."
            => "视差调整把整个场景推远（正值）或拉近。2 % 可让远处背景刚好落在窗口后面。",
        "A spatial video must be rectilinear: with Defish above 0 this clip exports as H.265 side by side instead. Set Defish to 0 in the reframed view."
            => "空间视频必须是直线透视：去鱼眼大于 0 时，这个片段会改为导出 H.265 左右并排。请在重新取景视图里把去鱼眼设为 0。",
        "Plays as a spatial video on Apple Vision Pro; other players show the left eye in 2D."
            => "在 Apple Vision Pro 上按空间视频播放；其他播放器以 2D 显示左眼。",
        SPATIAL_VIEW_SCALED_NOTE
            => "Apple Vision Pro 的沉浸模式最多按 {max_w}° 宽、{max_h}° 高以真实大小显示空间视频。当前视角（{view}）会被缩小显示为 {shown}，画面里的东西都会变小。想保持真实大小，请把变焦设到 {limit}° 以内。",
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
        "Dropped frames" => "丢帧",
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
        "Size" => "尺寸",
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
        "Tone" => "影调",
        "Lift / Gamma / Gain" => "Lift / Gamma / Gain（提升 / 伽马 / 增益）",
        "Shadow (earlier)" => "阴影（旧）",
        "Highlight (earlier)" => "高光（旧）",
        "Reset lift / gamma / gain" => "重置提升 / 伽马 / 增益",
        "From a grade made before Shadows and Highlights above existed. They work on the codes; the ones above work in linear light. Reset takes them away."
            => "来自上方“阴影”“高光”出现之前做的调色。它们作用在码值上，上方的两个在线性光里工作。按重置即可去掉。",
        "In stops, as a change of the light itself: ahead of the LUT, so what the LUT presses against white or black is still there to bring back."
            => "以档为单位，直接改变光量：在 LUT 之前进行，所以被 LUT 压到纯白或纯黑的部分仍然可以拉回来。",
        "More or less contrast about the middle of the frame's own levels."
            => "围绕画面自身亮度的中部增减对比度。",
        "Darkens or brightens the bright parts of the frame and leaves their texture: the change follows the frame's large areas, not each pixel. Below zero, highlights the LUT shows as white come back from what the file holds."
            => "压暗或提亮画面的亮部并保留其中的纹理：变化跟随画面的大块区域，而不是逐个像素。小于零时，被 LUT 显示成纯白的高光会从文件里实际记录的信息中恢复出来。",
        "Brightens or darkens the dark parts of the frame and leaves their texture: the change follows the frame's large areas, not each pixel."
            => "提亮或压暗画面的暗部并保留其中的纹理：变化跟随画面的大块区域，而不是逐个像素。",
        "Moves the brightest levels: where white sits." => "移动最亮的部分：决定白点的位置。",
        "Moves the darkest levels: where black sits." => "移动最暗的部分：决定黑点的位置。",
        "Changes of exposure in linear light, made ahead of Lift / Gamma / Gain and the LUT. They go by the levels of the whole clip, so a control does the same to every frame of it."
            => "都是线性光里的曝光变化，在 Lift / Gamma / Gain 和 LUT 之前进行。依据的是整段素材的亮度分布，所以同一个设置对每一帧的作用相同。",
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
        "Shown: the curve this gamma is built from. The SDK applies edits under its \"Blackmagic Design Custom\" gamma only — moving a control switches to it, continuing from this curve (the clip's Film LUT goes off)."
            => "显示的是当前伽马所基于的曲线。SDK 只在 “Blackmagic Design Custom” 伽马下应用修改——拖动任一控件会切换到该伽马并从这条曲线继续（素材的 Film LUT 随之关闭）。",
        "Highlight rolloff" => "高光过渡",
        "Another colour science than the clip's: its own gamma and gamut lists apply (Gen 6 has no Blackmagic Design gammas — a Film clip decodes to DaVinci Intermediate / DaVinci Wide Gamut)."
            => "已选择与素材不同的色彩科学：使用该代际自己的伽马/色域列表（Gen 6 没有任何 Blackmagic Design 伽马——Film 素材会按 DaVinci Intermediate / DaVinci Wide Gamut 解码）。",
        "This colour science has no adjustable (Custom) tone curve — the SDK ignores these controls under it. Pick Gen 5 to use the tone curve."
            => "该色彩科学没有可调的（Custom）色调曲线——SDK 在其下会忽略这些控件。要使用色调曲线请选择 Gen 5。",
        "Shadow rolloff" => "阴影过渡",
        "Where the curve's middle sits — the strongest overall brightness control." => "曲线中点的位置——整体亮度最强的控制项。",
        "Knee shape of the brightest few percent only — lower = softer (highlights compressed), higher = harder. A subtle control by design; for a level change use White level."
            => "只影响最亮几个百分点的拐点形状——数值越低越柔和（高光被压缩），越高越硬。设计上作用就很轻微；要改电平请用“白电平”。",
        "Toe shape of the darkest few percent only — lower = softer, which lifts the deepest shadows; higher = harder, which pushes them down. A subtle control by design; for a level change use Black level."
            => "只影响最暗几个百分点的趾部形状——数值越低越柔和（最深的暗部被抬起），越高越硬（暗部被压下）。设计上作用就很轻微；要改电平请用“黑电平”。",
        "native" => "原始",
        "native + margin" => "原始 + 余量",
        "full" => "全尺寸",
        "Above 8192 × 4096 the Mac's hardware ProRes encoder refuses the frame — the software encoder takes over (several times slower)."
            => "超过 8192 × 4096 时 Mac 的硬件 ProRes 编码器不接受该画幅——改用软件编码器（慢数倍）。",
        "Above 8192 × 4096 the file exceeds HEVC Level 6.2 — it encodes in hardware, but most players cannot decode it."
            => "超过 8192 × 4096 时文件超出 HEVC Level 6.2——可用硬件编码，但多数播放器无法解码。",
        "Above 8192 pixels a side the NVIDIA hardware encoder (NVENC) refuses the frame — the software encoder (libx265) takes over, many times slower. The file also exceeds HEVC Level 6.2: most players cannot decode it."
            => "任一边超过 8192 像素时 NVIDIA 硬件编码器（NVENC）不接受该画幅——改用软件编码器（libx265），慢很多倍。文件同时超出 HEVC Level 6.2，多数播放器无法解码。",
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
        "Mode" => "档位",
        "Bitrate" => "比特率",
        "native sharpness" => "原生清晰度",
        "Native sharpness: the 180° canvas at the pixel density of the Crop frame's centre, so nothing recorded is scaled down. The clip's own size puts the same frame on a smaller canvas."
            => "原生清晰度：按 Crop 画面中心的像素密度铺 180° 画布，录到的细节不被缩小。选素材自身尺寸时，同一画面会铺在更小的画布上。",
        "Video bit rate from the file. Two numbers in brackets are the two eyes' streams. ≈ means the whole file's size over its duration."
            => "文件里的视频比特率。括号里的两个数是两眼各自的码流。≈ 表示按整个文件大小除以时长估算。",
        "view to be confirmed" => "视角待确认",
        "unrecognized X6 mode" => "未识别的 X6 档位",
        "View" => "视角",
        "Wide" => "广角",
        "This 3712 × 3712 clip has no usable window record, so the file cannot say whether it is the wide 7.4K or the 7.4K Crop. Pick the one it was shot in: the lens calibration and the rolling-shutter timing follow it."
            => "这条 3712 × 3712 素材没有可用的窗口记录，文件本身无法区分广角 7.4K 和 7.4K Crop。请选拍摄时的档位：镜头标定和果冻校正时序会随之切换。",
        "The lens geometry of this recording mode is not verified: both eyes are shown through the generic rule so the clip can be inspected."
            => "该档位的镜头几何未经验证：两眼按通用规则显示，便于检查素材。",
        "Crop mask" => "Crop 遮罩",
        "Reframed view: no mask — the frame's edge is repeated past the recorded picture, so a stabilization border does not show black."
            => "重构视图：不使用遮罩——画面以外重复边缘像素，防抖露出的边不会是黑的。",
        "Mask size (%)" => "遮罩范围 (%)",
        "Feather (%)" => "羽化 (%)",
        "Corner roundness (%)" => "圆角 (%)",
        "Reset mask" => "重置遮罩",
        "How much of the recorded view the black mask hides on every side. 0 hides nothing."
            => "黑色遮罩在每一边遮住多少画面。0 表示不遮。",
        "Soft edge: the picture fades to black over this width inside the mask edge. 0 = hard edge."
            => "软边：画面在遮罩边缘内侧这段宽度上渐隐到黑。0 = 硬边。",
        "0 = rectangle, 100 = ellipse."
            => "0 = 矩形，100 = 椭圆。",
        "A Crop clip holds part of the lens' view: outside the recorded frame the output is black. The mask is fixed in the output, so it stays still under stabilization. The reframed view does not use it."
            => "Crop 素材只记录了镜头视野的一部分：画面以外输出为黑。遮罩固定在输出画面上，开启防抖时不会跟着晃。重构视图不使用遮罩。",
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
        "ETTR recovery (I-Log)" => "ETTR 还原（I-Log）",
        "Shot over by" => "拍摄时多曝",
        "stops" => "档",
        "For a clip exposed brighter than normal on purpose (exposing to the right). The extra exposure is taken back out in linear light, ahead of the tone controls and the LUT. Clipped highlights stay clipped."
            => "用于拍摄时有意多给曝光（向右曝光）的素材。多给的曝光在线性光下还原，再进入影调和 LUT。已经剪切的高光无法恢复。",
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
        "The camera was mounted upside down. The picture is turned upright, the two eyes change sides, and the sound is turned with them. The stereo alignment stays with its lens and the view angles act as you see them, so one set of values serves both ways the camera can be mounted. An X6 clip or raw take sets this by itself from its accelerometer."
            => "相机是倒着装的。画面转正，左右眼互换，声音也跟着转。两眼对位跟着镜头走，视角调整按你看到的方向起作用，所以同一套数值正装倒装都能用。X6 的片段和 raw 素材会按加速度计自动设置这个开关。",
        "Set from the clip's accelerometer." => "已按素材的加速度计自动设置。",
        "Follow the accelerometer" => "跟随加速度计",
        "The switch was set by hand. This puts it back to what the clip's accelerometer says." => "这个开关是手动设置的。点这里恢复为素材加速度计给出的结果。",
        "Stabilization (VQF 6D, BRAW)" => "防抖 (VQF 6D, BRAW)",
        "Stabilization (DJI camera quats)" => "防抖（DJI 相机四元数）",
        "Stabilization (Insta360 gyro)" => "防抖（Insta360 陀螺仪）",
        "Raw development" => "RAW 显影",
        "Temperature (K)" => "色温 (K)",
        "Exposure" => "曝光",
        "Auto exposure" => "自动曝光",
        "The clip sets its own exposure, frame by frame: each frame gets the exposure at which its picture, developed with every other control as it is set here, comes out normally bright. The exposure follows the light smoothly, from both sides; nothing else in the rendering changes with it."
            => "素材逐帧自己决定曝光：每一帧取这样一个曝光值，使它按这里其余设置显影出来的画面达到正常亮度。曝光从前后两侧平滑地跟随光线变化，渲染的其余部分不随之改变。",
        "Stops added to what the automatic exposure finds: brighter or darker than its own target."
            => "在自动曝光找到的数值上再加的档数：比它自己的目标更亮或更暗。",
        "Smoothness (s)" => "平滑度（秒）",
        "How slowly the exposure follows the picture: each frame's exposure is the average over about this many seconds of the clip on either side. 0 follows the picture as it changes; a large value gives the whole clip one exposure."
            => "曝光跟随画面的快慢：每一帧的曝光是它前后各约这么多秒范围内的平均。0 表示紧跟画面变化；数值很大时整段素材用同一个曝光。",
        "Metering the clip…" => "正在为素材测光…",
        "Over the clip:" => "整段素材：",
        "Whites" => "白色色阶",
        "Blacks" => "黑色色阶",
        "As shot" => "原始设置",
        "Sharpening" => "锐化",
        "Noise reduction" => "降噪",
        "Color noise" => "颜色",
        "Colour noise reduction, as a raw converter's: the picture's colour is smoothed and its luminance left alone, which takes the coloured speckle out of noise and the colour out of fine lines. 25 is a raw converter's default; 0 turns it off." => "颜色降噪，与 RAW 转换软件的做法一致：只平滑画面的颜色，亮度不动，用来去掉噪点里的彩色斑点和细线上的杂色。25 是 RAW 转换软件的默认值，0 为关闭。",
        "How much of the finest detail is sharpened. Low values hold halos back, as a raw converter's do: a hard edge is sharpened less, a soft one keeps its steepening." => "最细的细节锐化多少。数值低时会像 RAW 转换软件那样抑制光晕：硬边锐化得少，柔和的边缘仍然变陡。",
        "Amount" => "数量",
        "Masking" => "蒙版",
        "Sharpening of the developed picture, as a raw converter's detail panel does it: the same numbers give the same edges there. The radius is in pixels of the exported frame, so judge it in the zoom view or on a snapshot; the preview shows the export's picture reduced." => "对显影后的画面做锐化，与 RAW 转换软件“细节”面板的做法一致：同样的数值在那里得到同样的边缘。半径以导出画面的像素计，请在放大视图或快照上判断；预览显示的是导出画面缩小后的样子。",
        "Keeps the sharpening to edges: the higher, the stronger an edge has to be." => "把锐化限制在边缘上：数值越高，边缘要越强才会被锐化。",
        "Reset raw development" => "重置 RAW 显影",
        "Read from Camera Raw" => "从 Camera Raw 读取",
        "Reading…" => "正在读取…",
        "Read from" => "已读取自",
        "Not carried over:" => "未带回：",
        "Takes the white balance, tone, sharpening and colour noise reduction a frame was left with in Camera Raw. They are read from the Camera original DNG of the frame on screen, or of the nearest frame that has settings, in the folder beside the take. Whatever else was set there is named below and stays there."
            => "读取某一帧在 Camera Raw 里留下的白平衡、影调、锐化和颜色降噪。来源是素材旁边文件夹里当前画面那一帧的相机原片 DNG，该帧没有设置时取最近的有设置的一帧。在那里做的其他调整会在下面列出，不会带回。",
        "No Camera Raw settings found: the folder of Camera original DNGs beside this take has none, or is not there."
            => "没有找到 Camera Raw 设置：素材旁边的相机原片 DNG 文件夹里没有，或者文件夹不存在。",
        "The other eye's file has other settings: these are channel 0's." => "另一眼的文件设置不同：这里用的是通道 0 的。",
        "Texture" => "纹理",
        "Clarity" => "清晰度",
        "Dehaze" => "去除薄雾",
        "Vibrance" => "自然饱和度",
        "Luminance noise reduction" => "明亮度降噪",
        "Color noise reduction detail" => "颜色降噪的细节",
        "Color noise reduction smoothness" => "颜色降噪的平滑度",
        "Grain" => "颗粒",
        "Tone curve" => "曲线",
        "Color mixer" => "混色器",
        "Color grading" => "颜色分级",
        "Calibration" => "校准",
        "Vignette" => "晕影",
        "Lens corrections" => "镜头校正",
        "Geometry" => "几何",
        "Crop" => "裁剪",
        "Black & white" => "黑白",
        "Masks" => "蒙版",
        "Healing" => "修复",
        "Profile" => "配置文件",
        "The white the frame is developed for, as a colour temperature in kelvin and a tint, the way a raw converter states it." => "显影所用的白点，以色温（开尔文）和色调表示，与 RAW 转换软件的写法一致。",
        "The take records no white balance of its own: As shot is the camera's reference white." => "该素材没有记录自己的白平衡：“原始设置”是相机的参考白点。",
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
        // Several clips as one file
        "Join into one file" => "合成一个文件",
        JOIN_HOVER => "按列表顺序导出各片段，并首尾相接写成一个文件（硬切）。每个片段保留自己的裁切和设置。各片段的输出尺寸和帧率必须相同（在“格式”里为它们选同一尺寸）。合成文件的声音跟随第一个片段的音频选项：立体声；当每个片段都有 4 声道声音时，也可以是 Ambisonic 或 APAC。使用相机镜头的“镜头空间”视图的片段会合成为 Apple Immersive Video 文件（.aivu，需要 macOS 26 或更高版本，供 Apple Vision Pro 播放）：文件里为每个片段各存一个镜头，所以不同拍摄模式、不同相机的片段可以放在一起。拖动片段左侧的手柄（☰）可以调整顺序。",
        "has no four-channel sound: a joined file with Ambisonic or APAC sound needs it in every clip. Set the first clip's Audio to Stereo, or leave this clip out." => "没有 4 声道声音：要让合成文件带 Ambisonic 或 APAC 音轨，每个片段都必须有 4 声道声音。请把第一个片段的音频改为立体声，或者不要包含这个片段。",
        "Join all" => "合成全部",
        "Join selected" => "合成所选",
        "Export the ticked clips as one file" => "把勾选的片段导出为一个文件",
        "joining the clips…" => "正在合成片段…",
        "enhanced frames left out" => "未使用增强帧",
        "In the Lens Space view a stabilized clip's frame is larger than that of a clip that is not stabilized: turn stabilization on for both clips, or off for both." => "镜头空间视图下，开启防抖的片段画幅比未开启防抖的片段大：请让两个片段都开启防抖，或都关闭。",
        "Not joined" => "未合成",
        "the clips exported so far are in" => "已导出的片段保留在",
        "was not exported" => "没有导出成功",
        "no clip was exported" => "没有片段导出成功",
        "stopped" => "已停止",
        "Drag to change the order" => "拖动以调整顺序",
        "is a still photo: a joined file is made of video clips." => "是照片：合成文件只能由视频片段组成。",
        "A DNG sequence is a folder of frames for each clip: it cannot be joined. Pick a video format in Format." => "DNG 序列是每个片段一个帧文件夹，无法合成。请在“格式”里选择视频格式。",
        "A spatial video cannot be joined: export the clips one by one, or pick H.265 or ProRes in Format." => "空间视频无法合成：请逐个导出，或在“格式”里改用 H.265 或 ProRes。",
        "is in the Lens Space view through the camera's lens. Clips in that view are joined as an Apple Immersive Video file, which needs macOS 26 or later and two eyes in every clip: here they cannot be joined. Pick the Equidistant lens, or another view." => "用的是相机镜头的镜头空间视图。这种视图的片段要合成为 Apple Immersive Video 文件，需要 macOS 26 或更高版本，而且每个片段都得是双眼画面：这里无法合成。请改用等距镜头，或换一种视图。",
        // The list as a project
        "Project" => "项目",
        PROJECT_HOVER => "把片段列表连同每个片段的裁切、设置和导出选项存成一个文件，可以保存和打开。下次启动时，列表也会自动恢复。",
        "New project" => "新建项目",
        "Open project…" => "打开项目…",
        "Save project" => "保存项目",
        "Save project as…" => "项目另存为…",
        "Save project…" => "保存项目…",
        "Don't save" => "不保存",
        "Replace the clip list?" => "替换当前片段列表？",
        "The clip list has changes that are not saved in a project file." => "当前片段列表有尚未保存到项目文件的更改。",
        "looking…" => "查找中…",
        "missing" => "文件缺失",
        "The file was not found where the project says it is. Click to look again." => "在项目记录的位置没有找到这个文件。点击可重新查找。",
        _ => en,
    }
}

/// Load a system CJK font as a FALLBACK so Chinese (中文) renders instead of
/// tofu boxes. Pushed after the default Latin fonts for both families, so
/// Latin keeps the bundled font and only CJK glyphs fall through to this.
/// No-op (logs a warning) if no system Chinese font is found.
pub(super) fn install_cjk_font(ctx: &egui::Context) {
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

impl App {
    /// Pick an output path and spawn the export worker. No-op if no
    /// clip is loaded, an export is already in flight, or the source
    /// is GoPro EAC (the existing CLI handles that family).
    /// Top-toolbar Export button — opens the options window. The actual
    /// export kicks off from `commit_export()` once the user confirms.
    /// Export just the ACTIVE clip — a "batch of one". Routes through the
    /// SAME queue runner as "Export all": a single export is a one-item run,
    /// so there's one code path, one set of output settings, one progress UI.
    /// `…/VR180SilverBullet2.0/lang` — the UI-language preference file.
    pub(super) fn lang_config_path() -> Option<PathBuf> {
        crate::decoder::Settings::config_path()
            .and_then(|p| p.parent().map(|d| d.join("lang")))
    }

    pub(super) fn load_lang() -> Lang {
        match Self::lang_config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.trim().to_string())
            .as_deref()
        {
            Some("zh") => Lang::Zh,
            _ => Lang::En,
        }
    }

    pub(super) fn save_lang(&self) {
        if let Some(p) = Self::lang_config_path() {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, if self.lang == Lang::Zh { "zh" } else { "en" });
        }
    }
}
