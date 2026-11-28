//! The clip list: an entry, what is known of its clip, and the settings it keeps.

use super::*;

/// One entry in the SOURCE CLIP LIST — the app's primary working set.
/// Every loaded file lives here with its OWN `settings`; the ACTIVE
/// entry (`App::active_clip`) is what the preview shows and what the
/// sidebar edits (its settings mirror `App::settings` live, and sync
/// back on clip switch / batch start). The batch export simply runs
/// this list. Probe data is captured at add time (cheap header read).
pub(super) struct BatchItem {
    pub(super) path: PathBuf,
    pub(super) fps: f32,
    pub(super) duration_sec: f64,
    pub(super) source_kind: vr180_pipeline::SourceKind,
    pub(super) fisheye_eye_w: u32,
    pub(super) fisheye_eye_h: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) settings: crate::decoder::Settings,
    pub(super) status: BatchStatus,
    /// Ticked in the Source list to include in "Export selected".
    pub(super) selected: bool,
    /// Insta360 X6 video: the trailer's recording-mode signature (the
    /// native-sharpness export size of a Crop clip follows from it).
    pub(super) x6: Option<X6ClipInfo>,
    /// A raw take of one lens ([`raw_take_of_one_lens`]).
    pub(super) raw_one_lens: bool,
    /// A raw take's frame in its lens's view ([`RawFrame`]).
    pub(super) raw_frame: Option<RawFrame>,
    /// The clip has four-channel sound a spatial track can be made of
    /// ([`four_channel_sound`]).
    pub(super) four_channel_sound: bool,
    /// A raw take's own size, and what it says of itself as an archive copy.
    pub(super) take: Option<TakeFacts>,
    /// Which way up the camera was, where the clip's accelerometer says
    /// ([`mounted_upside_down`]): the Upside-down mount switch follows it.
    pub(super) upside_down: Option<bool>,
    /// The entry's file is not at hand (see [`OfflineClip`]); what the
    /// fields above say of the file is then empty.
    pub(super) offline: Option<OfflineClip>,
    /// The status the entry had when a run that writes the takes themselves
    /// began ([`TakeRun`]): that run leaves the export queue as it found it.
    pub(super) before_take: Option<BatchStatus>,
    /// The archive copy the last such run made of this take.
    pub(super) copy: Option<TakeCopy>,
    /// Why the last such run did not write this take, and what it was to
    /// write (the export queue's own status is back by then).
    pub(super) take_failed: Option<(TakeJob, String)>,
    /// The entry reads an archive copy in the place of its take, which is
    /// not at hand ([`StandIn`]): `path` is then the copy.
    pub(super) stand_in: Option<StandIn>,
}

/// A clip's sound is four channels an Ambisonic or APAC track can be
/// written from: the X6's 360° Audio (a clip's own or a raw take's), the
/// QooCam's four capsules, a `.360`'s ambisonic track. Reads a header.
pub(super) fn four_channel_sound(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> bool {
    use vr180_pipeline::audio::{probe_foa_aac, probe_spatial_4ch};
    use vr180_pipeline::SourceKind as K;
    let found = |info: vr180_pipeline::Result<Option<vr180_pipeline::audio::AmbisonicInfo>>| info.ok().flatten().is_some();
    match kind {
        K::Insta360Insv => found(probe_foa_aac(&vr180_pipeline::x6_pair::audio_source(path).0)),
        K::X6Raw => vr180_pipeline::x6_raw::audio_source(path).is_some_and(|(file, _)| found(probe_foa_aac(&file))),
        K::KandaoQoocam => found(probe_foa_aac(path)),
        K::GoProEac => found(probe_spatial_4ch(path)),
        _ => false,
    }
}

/// A clip of an opened project, or of the last session, whose file is not
/// at hand: being looked for, or not found. The entry keeps its place in
/// the list and its settings, exports nothing, and goes back into the
/// project as it was read.
pub(super) struct OfflineClip {
    /// The request that looks for it ([`App::poll_restore`]).
    pub(super) token: u64,
    /// Still being looked for (`false`: not found).
    pub(super) looking: bool,
    pub(super) file: crate::project::FileRef,
    pub(super) segments: Option<Vec<crate::project::FileRef>>,
    /// The folder of the project file it was read from.
    pub(super) base: Option<PathBuf>,
}

/// What the list keeps of a clip's file, read from its header when the
/// clip joins the list.
pub(super) struct ClipFacts {
    pub(super) source_kind: vr180_pipeline::SourceKind,
    pub(super) fps: f32,
    pub(super) duration_sec: f64,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fisheye_eye_w: u32,
    pub(super) fisheye_eye_h: u32,
    pub(super) x6: Option<X6ClipInfo>,
    pub(super) raw_one_lens: bool,
    pub(super) raw_frame: Option<RawFrame>,
    pub(super) four_channel_sound: bool,
    pub(super) take: Option<TakeFacts>,
    pub(super) upside_down: Option<bool>,
}

/// Whether the camera was mounted upside down for the clip at `path`, from
/// its accelerometer: an X6 clip's or raw take's first seconds of it.
/// `None` for a source without one (a still, another camera), and where the
/// record says nothing. Touches the file.
pub(super) fn mounted_upside_down(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> Option<bool> {
    match kind {
        vr180_pipeline::SourceKind::X6Raw => vr180_pipeline::x6_raw::mounted_upside_down(path),
        vr180_pipeline::SourceKind::Insta360Insv => vr180_pipeline::insv_imu::file_mounted_upside_down(path),
        _ => None,
    }
}

/// The Upside-down mount switch of a clip whose accelerometer says
/// `detected`: it follows the accelerometer unless the user has set it by
/// hand. Nothing changes for a clip without such a record.
pub(super) fn follow_the_mount(settings: &mut crate::decoder::Settings, detected: Option<bool>) {
    if let (true, Some(upside_down)) = (settings.camera_upside_down_auto, detected) {
        settings.camera_upside_down = upside_down;
    }
}

impl ClipFacts {
    /// Read them. `None`: not a source the app exports, or not readable
    /// (said in the log). Touches the file: a clip of a project is read on
    /// a thread.
    pub(super) fn read(path: &std::path::Path) -> Option<Self> {
        let source_kind = vr180_pipeline::source_kind::detect(path)
            .unwrap_or(vr180_pipeline::SourceKind::Unknown);
        if !source_kind.is_exportable() {
            tracing::warn!("batch: skipping non-exportable source {}", path.display());
            return None;
        }
        let probe = if source_kind.is_still() {
            // One Bayer frame, both eyes side by side: no video probe.
            match vr180_fisheye::dng::probe(path) {
                Ok((w, h)) => vr180_pipeline::decode::VideoProbe { width: w, height: h, fps: 1.0, duration_sec: 0.0, hlg: false, video_bitrates_bps: [0; 2] },
                Err(e) => { tracing::warn!("batch: DNG probe failed for {}: {e}", path.display()); return None; }
            }
        } else { match vr180_pipeline::decode::probe_video(path) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("batch: probe failed for {}: {e}", path.display());
                return None;
            }
        } };
        let (fisheye_eye_w, fisheye_eye_h) = match source_kind {
            vr180_pipeline::SourceKind::SbsFisheye
            | vr180_pipeline::SourceKind::Insta360Dng
            | vr180_pipeline::SourceKind::KandaoDng => (probe.width / 2, probe.height),
            vr180_pipeline::SourceKind::KandaoQoocam => (probe.width, probe.height / 2),
            vr180_pipeline::SourceKind::DjiOsv
            | vr180_pipeline::SourceKind::X6Raw
            | vr180_pipeline::SourceKind::Insta360Insv => (probe.width, probe.height),
            // The probe is the helper's own (`probe_video` routes `.braw`
            // to it): the frame holds both eyes side by side, unless the
            // first frame shows one centred circle (`braw_layout`).
            vr180_pipeline::SourceKind::BlackmagicRaw => braw_eye_dims(path, probe.width, probe.height),
            _ => (0, 0),
        };
        Some(Self {
            source_kind,
            fps: probe.fps,
            duration_sec: probe.duration_sec,
            width: probe.width,
            height: probe.height,
            fisheye_eye_w, fisheye_eye_h,
            // Insta360 X6: the recording-mode signature (a few KB from the
            // file's end) — a Crop clip's native-sharpness size follows from it.
            x6: (source_kind == vr180_pipeline::SourceKind::Insta360Insv).then(|| X6ClipInfo::read(path).0),
            raw_one_lens: raw_take_of_one_lens(source_kind, path),
            raw_frame: RawFrame::read(source_kind, path),
            four_channel_sound: four_channel_sound(source_kind, path),
            take: TakeFacts::read(source_kind, path),
            upside_down: mounted_upside_down(source_kind, path),
        })
    }
}

impl BatchItem {
    /// An entry for a clip of a project, before its file has been looked
    /// for.
    pub(super) fn offline(clip: &crate::project::ProjectClip, token: u64, base: Option<PathBuf>) -> Self {
        Self {
            path: clip.file.path.clone(),
            fps: 0.0,
            duration_sec: 0.0,
            source_kind: vr180_pipeline::SourceKind::Unknown,
            fisheye_eye_w: 0, fisheye_eye_h: 0,
            width: 0, height: 0,
            settings: clip.settings.clone(),
            status: BatchStatus::Idle,
            selected: clip.selected,
            x6: None,
            raw_one_lens: false,
            raw_frame: None,
            four_channel_sound: false,
            take: None,
            upside_down: None,
            offline: Some(OfflineClip { token, looking: true, file: clip.file.clone(), segments: clip.segments.clone(), base }),
            before_take: None,
            copy: None,
            take_failed: None,
            stand_in: None,
        }
    }

    /// The file a project names for this entry, and its trim there: the
    /// take's, also while a copy is read in the take's place.
    pub(super) fn as_saved(&self, settings: &crate::decoder::Settings) -> (PathBuf, crate::decoder::Settings) {
        match &self.stand_in {
            None => (self.path.clone(), settings.clone()),
            Some(stand_in) => {
                let mut settings = settings.clone();
                (settings.trim_in_s, settings.trim_out_s) = stand_in.trim_in_take((settings.trim_in_s, settings.trim_out_s), self.fps);
                (stand_in.take.clone(), settings)
            }
        }
    }

    /// The entry's file is `path`, and says `facts` of itself.
    pub(super) fn take_facts(&mut self, path: PathBuf, facts: ClipFacts) {
        self.path = path;
        self.source_kind = facts.source_kind;
        self.fps = facts.fps;
        self.duration_sec = facts.duration_sec;
        self.width = facts.width;
        self.height = facts.height;
        self.fisheye_eye_w = facts.fisheye_eye_w;
        self.fisheye_eye_h = facts.fisheye_eye_h;
        self.x6 = facts.x6;
        self.raw_one_lens = facts.raw_one_lens;
        self.raw_frame = facts.raw_frame;
        self.four_channel_sound = facts.four_channel_sound;
        self.take = facts.take;
        self.upside_down = facts.upside_down;
        self.offline = None;
        // Which way up the camera was is the file's to say.
        follow_the_mount(&mut self.settings, self.upside_down);
    }

    /// A raw take as the camera wrote it: what an archive copy is made of.
    pub(super) fn camera_take(&self) -> bool {
        self.ready() && self.source_kind.is_raw_video() && self.take.as_ref().is_some_and(|t| t.archive.is_none())
    }

    /// The samples of one of its frames, both eyes together.
    pub(super) fn frame_samples(&self) -> u64 {
        self.width as u64 * self.height as u64 * if self.single_eye() { 1 } else { 2 }
    }

    /// The file is at hand and its header has been read.
    pub(super) fn ready(&self) -> bool {
        self.offline.is_none()
    }

    /// What a joined run's plan takes from this entry's own probe, with
    /// whether its export reads its take's enhanced frames (`enhanced`).
    pub(super) fn part_source(&self, enhanced: bool) -> vr180_pipeline::join::PartSource {
        vr180_pipeline::join::PartSource {
            one_eye: self.single_eye(),
            eye: native_eye_dims(self.source_kind, self.fisheye_eye_w, self.fisheye_eye_h, self.width, self.height),
            enhanced,
        }
    }

    /// The source holds ONE eye (see `ClipInfo::single_eye`).
    pub(super) fn single_eye(&self) -> bool {
        self.x6.is_some_and(|x| x.single_eye) || self.raw_one_lens
    }

    /// The clip is taken as Insta360 I-Log (see [`takes_ilog`]).
    pub(super) fn ilog(&self) -> bool {
        takes_ilog(self.source_kind, self.x6.as_ref())
    }

    /// The settings bucket this entry belongs to (see
    /// `App::settings_class_key`): "Apply settings to all" stays inside it.
    pub(super) fn settings_class(&self) -> &'static str {
        App::settings_class_key(self.source_kind, self.single_eye())
    }

    /// The settings this entry exports with: its own, taken as a single-eye
    /// source takes them when it is one, and told whether its picture is
    /// I-Log.
    pub(super) fn export_settings(&self) -> crate::decoder::Settings {
        let mut s = if self.single_eye() { self.settings.for_single_eye() } else { self.settings.clone() };
        s.ilog_source = self.ilog();
        s.raw_source = self.source_kind.is_raw_video();
        s.raw_camera = s.raw_source.then(|| vr180_pipeline::x6_raw::camera_at(&self.path));
        s
    }

    /// Per-eye side of the native-sharpness canvas for this entry, under
    /// its own view choice (`None` unless its frame is a crop of the lens'
    /// view — see `ExportResolution::Sharp`).
    pub(super) fn sharp_eye(&self) -> Option<u32> {
        match self.raw_frame {
            Some(frame) => frame.sharp_eye(self.fisheye_eye_w, self.fisheye_eye_h),
            None => self.x6.and_then(|x| x.sharp_eye(self.fisheye_eye_w, self.fisheye_eye_h, self.settings.insv_view)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum BatchStatus {
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
    pub(super) fn label(&self) -> String {
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

/// One eye of a clip at its own size, pixels: the eye of a fisheye source's
/// frame, the side of a `.360` clip's cross, the frame itself of anything
/// else. What an export at the clip's own size writes per eye, and what a
/// lens frame is sized against.
pub(super) fn native_eye_dims(kind: vr180_pipeline::SourceKind, fisheye_eye_w: u32, fisheye_eye_h: u32, width: u32, height: u32) -> (u32, u32) {
    if fisheye_eye_w > 0 {
        (fisheye_eye_w, fisheye_eye_h)
    } else if kind.is_eac() {
        // GoPro EAC: a SQUARE half-equirect per eye at the cross's
        // native resolution (cross_w = 2·tile_w + 1920; 3936 on Max).
        // Full quality vs the preview's ≤2048-clamped square.
        let cross_w = 2 * (width.saturating_sub(1920) / 4) + 1920;
        (cross_w.max(512), cross_w.max(512))
    } else {
        (width.max(512), height.max(512))
    }
}

/// A clip's name in a message: its file's (a raw take's folder's).
pub(super) fn clip_label(path: &std::path::Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

impl App {
    // ─── Source clip list + batch export ──────────────────────────

    /// Save the live sidebar settings back into the ACTIVE clip's list
    /// entry. Call before switching clips / starting a batch so edits
    /// aren't lost (the sidebar edits `self.settings`, not the entry).
    /// Persisted-settings bucket for a source kind. OSV / EAC / SBS / BRAW
    /// each keep an independent setup; anything else shares `"other"`.
    pub(super) fn settings_class_key(kind: vr180_pipeline::SourceKind, single_eye: bool) -> &'static str {
        use vr180_pipeline::SourceKind as K;
        match kind {
            K::DjiOsv         => "osv",
            // A single-eye clip keeps its own setup: what is tuned for the
            // camera's two eyes (their alignment above all) is not its.
            K::Insta360Insv if single_eye => "insv_single_eye",
            K::Insta360Insv   => "insv",
            K::X6Raw if single_eye => "x6raw_single_eye",
            K::X6Raw          => "x6raw",
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
    pub(super) fn kind_template(&self, kind: vr180_pipeline::SourceKind, single_eye: bool) -> Settings {
        let mut s = self.kind_settings.get(Self::settings_class_key(kind, single_eye))
            .cloned().unwrap_or_default();
        self.apply_shared_stereo(kind, single_eye, &mut s);
        s
    }

    /// The two template keys of a camera that writes video and stills from
    /// the same two lenses (the X6's `.insv` + `.dng`, the QooCam's `.mp4` +
    /// `.dng`): its stereo offset (the rig's own eye alignment) is ONE value
    /// shared by both kinds — whichever was edited last. Video key first.
    pub(super) fn stereo_shared_keys(kind: vr180_pipeline::SourceKind) -> Option<[&'static str; 2]> {
        let sib = kind.stereo_sibling()?;
        let (video, still) = if kind.is_still() { (sib, kind) } else { (kind, sib) };
        Some([Self::settings_class_key(video, false), Self::settings_class_key(still, false)])
    }

    /// The camera's shared stereo offset `(yaw, pitch, roll)`: both templates
    /// carry it (kept equal by `save_active_kind_template`), so read either —
    /// the video's first, for a user who has only ever tuned video.
    pub(super) fn shared_stereo(&self, kind: vr180_pipeline::SourceKind) -> Option<(f32, f32, f32)> {
        Self::stereo_shared_keys(kind)?.iter().find_map(|k| self.kind_settings.get(*k)
            .map(|s| (s.stereo_yaw_deg, s.stereo_pitch_deg, s.stereo_roll_deg)))
    }

    pub(super) fn apply_shared_stereo(&self, kind: vr180_pipeline::SourceKind, single_eye: bool, s: &mut Settings) {
        // One eye has no alignment to share.
        if single_eye { return; }
        if let Some((yaw, pitch, roll)) = self.shared_stereo(kind) {
            s.stereo_yaw_deg = yaw;
            s.stereo_pitch_deg = pitch;
            s.stereo_roll_deg = roll;
        }
    }

    /// Fold the active clip's live settings into ITS kind's template, so the
    /// next same-kind clip inherits them (kept independent across kinds —
    /// except a camera's shared stereo offset, written to both its templates).
    pub(super) fn save_active_kind_template(&mut self) {
        if let Some((kind, single_eye)) = self.clip.as_ref().map(|c| (c.source_kind, c.single_eye())) {
            self.kind_settings.insert(
                Self::settings_class_key(kind, single_eye).to_string(), self.settings.clone());
            if let Some(keys) = Self::stereo_shared_keys(kind).filter(|_| !single_eye) {
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

    pub(super) fn sync_active_clip_settings(&mut self) {
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
    pub(super) fn select_clip(&mut self, idx: usize) {
        if self.active_clip == Some(idx) { return; }
        // A clip whose file is not at hand is looked for again instead.
        if self.batch.get(idx).is_some_and(|b| !b.ready()) {
            self.look_again(idx);
            return;
        }
        // Its take first, an archive copy only while the take is not at hand.
        self.settle_source(idx);
        let Some(path) = self.batch.get(idx).map(|b| b.path.clone()) else { return; };
        self.sync_active_clip_settings();
        self.settings = self.batch[idx].settings.clone();
        // An X6 clip (video or still) takes the camera's shared stereo
        // offset — the latest edit on either kind — over its own snapshot.
        let (kind, single_eye) = (self.batch[idx].source_kind, self.batch[idx].single_eye());
        let mut restored = std::mem::take(&mut self.settings);
        self.apply_shared_stereo(kind, single_eye, &mut restored);
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
    pub(super) fn add_batch_files(&mut self, paths: Vec<PathBuf>) {
        let mut first_new: Option<usize> = None;
        for path in paths {
            if self.batch.iter().any(|b| b.path == path) {
                continue; // already in the list
            }
            // The take an entry reads a copy in the place of: it is back, and that entry reads it again.
            if let Some(i) = self.batch.iter().position(|b| b.stand_in.as_ref().is_some_and(|s| s.take == path)) {
                self.settle_source(i);
                continue;
            }
            if vr180_pipeline::x6_pair::plan_for(&path)
                .is_some_and(|take| self.batch.iter().any(|b| take.paths.contains(&b.path)))
            {
                continue;
            }
            let Some(facts) = ClipFacts::read(&path) else { continue };
            let ClipFacts { source_kind, fisheye_eye_w, fisheye_eye_h, x6, raw_one_lens, raw_frame, four_channel_sound, upside_down, .. } = facts;
            let take = facts.take.clone();
            let is_loaded_clip = self.loaded_path.as_deref() == Some(path.as_path());
            // A NEW clip inherits ITS kind's template (OSV / .360 / … kept
            // independent); the already-loaded clip keeps its live edits.
            let mut settings = if is_loaded_clip {
                self.settings.clone()
            } else {
                self.kind_template(source_kind, x6.is_some_and(|x| x.single_eye) || raw_one_lens)
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
                // How a clip was exposed is its own, and its file does not
                // say: never inherited from the last clip's setup.
                settings.ettr_stops = 0.0;
                // Which way up the camera was is the clip's own too, and its
                // accelerometer says: the last clip's switch is not passed on.
                settings.camera_upside_down_auto = true;
            }
            follow_the_mount(&mut settings, upside_down);
            self.batch.push(BatchItem {
                path,
                fps: facts.fps,
                duration_sec: facts.duration_sec,
                source_kind,
                fisheye_eye_w, fisheye_eye_h,
                width: facts.width, height: facts.height,
                settings,
                status: BatchStatus::Idle,
                selected: false,
                x6,
                raw_one_lens,
                raw_frame,
                four_channel_sound,
                take,
                upside_down,
                offline: None,
                before_take: None,
                copy: None,
                take_failed: None,
                stand_in: None,
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
    pub(super) fn apply_current_settings_to_item(&mut self, idx: usize) {
        let Some(item) = self.batch.get_mut(idx) else { return; };
        if item.status == BatchStatus::Running { return; }
        let keep_trim = (item.settings.trim_in_s, item.settings.trim_out_s);
        let keep_rs = (item.settings.rs_mode, item.settings.rs_readout_ms);
        let keep_mount = (item.settings.camera_upside_down, item.settings.camera_upside_down_auto);
        item.settings = self.settings.clone();
        item.settings.trim_in_s = keep_trim.0;
        item.settings.trim_out_s = keep_trim.1;
        // Which way up the camera was is each clip's own where its
        // accelerometer says: one clip's switch is not another's. (A clip
        // without such a record takes the switch like any other setting.)
        if item.upside_down.is_some() {
            (item.settings.camera_upside_down, item.settings.camera_upside_down_auto) = keep_mount;
        }
        // RS mode + readout are per-clip — keep this item's own, don't copy mine.
        item.settings.rs_mode = keep_rs.0;
        item.settings.rs_readout_ms = keep_rs.1;
        // The ETTR recovery travels with the grade, to the clips it can
        // apply to: anything that is not I-Log keeps none, so a value it
        // could neither show nor use never rides along to the next copy.
        if !item.ilog() {
            item.settings.ettr_stops = 0.0;
        }
        // Re-arm a previously finished item so the change exports on the
        // next Start (while a run writes the takes themselves, the export
        // status is the one put aside for it).
        let status = item.before_take.as_mut().unwrap_or(&mut item.status);
        if !matches!(status, BatchStatus::Queued) {
            *status = BatchStatus::Idle;
        }
    }

    /// Put a clip of the list in another place; what points into the list
    /// by place follows.
    pub(super) fn move_clip(&mut self, m: crate::reorder::Move) {
        if m.from >= self.batch.len() || m.to >= self.batch.len() || m.from == m.to {
            return;
        }
        m.apply(&mut self.batch);
        self.active_clip = self.active_clip.map(|i| m.place_of(i));
        self.batch_current = self.batch_current.map(|i| m.place_of(i));
        for i in &mut self.batch_run_indices {
            *i = m.place_of(*i);
        }
    }

    /// Remove a clip from the list, keeping `active_clip` /
    /// `batch_current` indices coherent. If the removed entry is the
    /// clip currently loaded (or still loading), the preview is unloaded
    /// with it: the neighbouring entry becomes active when one remains,
    /// otherwise the app returns to the empty "no clip loaded" state.
    pub(super) fn remove_clip(&mut self, idx: usize) {
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
}
