//! What is known of the loaded clip, and the File info that shows it.

use super::*;

/// A clip is taken as Insta360 I-Log when it is an X6 video whose picture
/// profile (`gamma_mode`) does not say otherwise: every recording mode can be
/// shot in another profile, and a file that states none is taken as I-Log.
/// The built-in I-Log LUT and the ETTR recovery both go by this.
pub(super) fn takes_ilog(kind: vr180_pipeline::SourceKind, x6: Option<&X6ClipInfo>) -> bool {
    kind == vr180_pipeline::SourceKind::Insta360Insv && x6.map_or(true, |x| x.ilog != Some(false))
}

/// What an Insta360 X6 clip's trailer says about how it was recorded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct X6ClipInfo {
    /// The file's `window_crop_info` — the recording mode's signature
    /// (`None`: no record, or no readable trailer).
    pub(super) window: Option<vr180_fisheye::InsvWindowCrop>,
    /// `gamma_mode` is I-Log (`None`: the file does not say).
    pub(super) ilog: Option<bool>,
    /// Mean centre scale of the two lenses, native pixels per radian (from
    /// the file's calibration, else the generic profile).
    pub(super) paraxial_native: f64,
    pub(super) single_lens: bool,
    pub(super) single_eye: bool,
}

impl X6ClipInfo {
    /// Read it from a clip's trailer (a few KB from the end of the file).
    /// A clip whose trailer cannot be read still gets an entry: its mode
    /// then follows the frame size alone.
    pub(super) fn read(path: &std::path::Path) -> (Self, Option<String>) {
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
    pub(super) fn sharp_eye(&self, eye_w: u32, eye_h: u32, view: vr180_fisheye::X6ViewChoice) -> Option<u32> {
        let g = self.geometry(eye_w, eye_h, view);
        g.cropped.then(|| vr180_pipeline::insv_imu::native_sharpness_side(self.paraxial_native * g.scale_x))
    }

    /// The clip's stream geometry under a view choice (which only an
    /// unconfirmed 3712² clip listens to).
    pub(super) fn geometry(&self, eye_w: u32, eye_h: u32, view: vr180_fisheye::X6ViewChoice) -> vr180_fisheye::InsvStreamGeometry {
        if self.single_lens {
            return vr180_fisheye::InsvStreamGeometry::resolve_single_lens(self.window.as_ref(), eye_w, eye_h);
        }
        vr180_fisheye::InsvStreamGeometry::resolve(self.window.as_ref(), eye_w, eye_h, view)
    }
}

#[derive(Debug, Clone)]
pub(super) struct ClipInfo {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fps: f32,
    pub(super) duration_sec: f64,
    pub(super) frame_count: u32,
    pub(super) segments: Vec<PathBuf>,
    pub(super) eac_tile_w: u32,
    pub(super) cori_count: usize,
    pub(super) grav_count: usize,
    pub(super) srot_ms: Option<f32>,
    /// Detected source family. Drives which settings panels show in
    /// the sidebar — GoPro EAC keeps the stab / RS panels; fisheye
    /// sources (OSV / SBS / BRAW) get the camera-preset / FOV / KB
    /// panel instead.
    pub(super) source_kind: vr180_pipeline::SourceKind,
    /// One eye dimensions (set for fisheye sources; (0,0) for EAC).
    /// Used by the GUI to label the FOV slider with a realistic
    /// "max recommended" hint based on the working resolution.
    pub(super) fisheye_eye_w: u32,
    pub(super) fisheye_eye_h: u32,
    /// GoPro `.360` GEOC calibration dimension (square sensor cal space,
    /// 4216 for GoPro Max). `None` for non-EAC sources or when the file
    /// has no GEOC. The ".360 Lens calibration" panel displays cx/cy in
    /// this pixel space (matching CTRX/CTRY units).
    pub(super) eac_cal_dim: Option<u32>,
    /// Insta360 X6 recorded in "360° Audio" mode: the source carries a
    /// 4-channel first-order AmbiX track (so the Audio section applies).
    pub(super) foa_audio: bool,
    /// Kandao QooCam with its 4-channel microphone track (mixed to stereo
    /// for playback and export — `audio::QuadEncoder`).
    pub(super) quad_audio: bool,
    /// Blackmagic RAW: the SDK's colour processing as shot (values, ranges,
    /// lists) for the Color panel's Blackmagic RAW section.
    pub(super) braw_color: Option<std::sync::Arc<vr180_braw::BrawColorInfo>>,
    /// Blackmagic RAW: the clip's embedded display LUT written out as a
    /// `.cube` (auto-applied; switched off under a non-Film gamma).
    pub(super) braw_lut: Option<String>,
    /// Insta360 X6 video: the trailer's recording-mode signature and
    /// picture profile.
    pub(super) x6: Option<X6ClipInfo>,
    /// A raw take of one lens ([`raw_take_of_one_lens`]).
    pub(super) raw_one_lens: bool,
    /// A raw take's frame in its lens's view ([`RawFrame`]).
    pub(super) raw_frame: Option<RawFrame>,
    /// Bit rate of each video stream of the recording, bits per second, as
    /// the container reports it (two entries = the two eyes of a
    /// dual-stream file; the first chapter's for a chain). Empty when the
    /// container does not say.
    pub(super) video_bitrates_bps: Vec<u64>,
    /// The clip's size over its duration, bits per second — shown when the
    /// container states no stream rate (Blackmagic RAW). `None` for a still.
    pub(super) file_bitrate_bps: Option<u64>,
    /// Where frames are missing from the clip's track, in order (empty: none
    /// are); `None` when the track's own frame times could not be read.
    pub(super) holes: Option<Vec<vr180_pipeline::frame_timeline::Hole>>,
}

impl ClipInfo {
    /// The frames the camera dropped, for a clip of the Insta360 X6 (its
    /// video files and its raw takes leave a hole in the track where the
    /// sensor lost frames): the places, in order. `None` for another camera
    /// and when the track could not be read.
    pub(super) fn x6_holes(&self) -> Option<&[vr180_pipeline::frame_timeline::Hole]> {
        let x6 = self.source_kind == vr180_pipeline::SourceKind::Insta360Insv || self.source_kind.is_raw_video();
        self.holes.as_deref().filter(|_| x6)
    }

    /// The source holds ONE eye: the preview shows it alone and an export
    /// writes it as the whole frame.
    pub(super) fn single_eye(&self) -> bool {
        self.x6.is_some_and(|x| x.single_eye) || self.raw_one_lens
    }

    /// The clip is taken as Insta360 I-Log (see [`takes_ilog`]).
    pub(super) fn ilog(&self) -> bool {
        takes_ilog(self.source_kind, self.x6.as_ref())
    }

    /// Stream geometry of an X6 clip under `view` (`None` for other
    /// sources): a video file's by its recording mode, a raw take's as its
    /// sensor record places it (no view to choose).
    pub(super) fn x6_geometry(&self, view: vr180_fisheye::X6ViewChoice) -> Option<vr180_fisheye::InsvStreamGeometry> {
        match self.raw_frame {
            Some(frame) => Some(frame.geometry(self.fisheye_eye_w, self.fisheye_eye_h)),
            None => self.x6.map(|x| x.geometry(self.fisheye_eye_w, self.fisheye_eye_h, view)),
        }
    }

    /// Per-eye side of the native-sharpness canvas (`None` unless the clip's
    /// frame is a crop of the lens' view — see `ExportResolution::Sharp`).
    pub(super) fn sharp_eye(&self, view: vr180_fisheye::X6ViewChoice) -> Option<u32> {
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
pub(super) struct RawFrame {
    pub(super) stated: vr180_fisheye::insta360::StatedFrame,
    /// Mean centre scale of the take's lenses, native pixels per radian.
    pub(super) paraxial_native: f64,
    pub(super) one_lens: bool,
}

impl RawFrame {
    /// Read it from a take's small files. `None` for a source that is no
    /// raw take, and for a take without a sensor record (its frame is the
    /// camera's own video frame, the whole view).
    pub(super) fn read(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> Option<Self> {
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
    pub(super) fn geometry(&self, eye_w: u32, eye_h: u32) -> vr180_fisheye::InsvStreamGeometry {
        vr180_fisheye::InsvStreamGeometry::stated(&self.stated, eye_w, eye_h, self.one_lens)
    }

    /// Per-eye side of the 180° canvas at the frame's native sharpness,
    /// when the frame is a crop of the lens' view (see
    /// `ExportResolution::Sharp`).
    pub(super) fn sharp_eye(&self, eye_w: u32, eye_h: u32) -> Option<u32> {
        let g = self.geometry(eye_w, eye_h);
        g.cropped.then(|| vr180_pipeline::insv_imu::native_sharpness_side(self.paraxial_native * g.scale_x))
    }
}

impl App {
    pub(super) fn draw_source_info(&self, ui: &mut egui::Ui) {
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
}

/// The colour of a place where frames are missing (file info, timeline).
pub(super) const DROPPED_COLOR: Color32 = Color32::from_rgb(255, 120, 70);

/// The file info's word on the frames a clip's camera dropped.
pub(super) fn dropped_frames_label(holes: &[vr180_pipeline::frame_timeline::Hole], zh: bool) -> String {
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
pub(super) fn dropped_hole_text(hole: &vr180_pipeline::frame_timeline::Hole, zh: bool) -> String {
    match (zh, hole.frames) {
        (true, n) => format!("{}：丢 {n} 帧", format_time(hole.at_s)),
        (false, 1) => format!("{}: 1 frame dropped", format_time(hole.at_s)),
        (false, n) => format!("{}: {n} frames dropped", format_time(hole.at_s)),
    }
}

/// The file info line's explanation: what a dropped frame is here, then the
/// places (the first [`DROPPED_LISTED`] of them).
pub(super) fn dropped_frames_hover(holes: &[vr180_pipeline::frame_timeline::Hole], zh: bool) -> String {
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
pub(super) const DROPPED_LISTED: usize = 24;

/// Bit rate text for the file info. The video streams' total — and, with
/// `per_stream`, each eye's stream of a dual-stream file in brackets
/// (`253.3 Mbps (126.6 + 126.7)`); `≈ … Mbps` from the file size when the
/// container states no stream rate; `None` when neither is known (a still).
pub(super) fn bitrate_label(video_bps: &[u64], file_bps: Option<u64>, per_stream: bool) -> Option<String> {
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
pub(super) fn x6_mode_label(g: &vr180_fisheye::InsvStreamGeometry) -> Option<String> {
    use vr180_fisheye::X6Mode as M;
    match g.detected {
        M::Stock => None,
        M::Unconfirmed7K4 => Some(format!("7.4K60 · {}", tr("view to be confirmed"))),
        M::Unknown => Some(tr("unrecognized X6 mode").to_string()),
        m => m.label().map(str::to_string),
    }
}
