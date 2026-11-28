//! Opening clips: what is picked or dropped, how files make up a clip, and the load itself.

use super::*;

/// One recording detected among the files the user just picked. GoPro
/// `.360` members are auto-detected chapters (`detect_segments`); OSV
/// members are exactly the files the user multi-selected (OSV segments are
/// indistinguishable from separate clips by name, so the user chooses
/// them). A group with >1 member is offered "merge into one clip vs import
/// individually".
pub(super) struct ImportGroup {
    /// Files the user actually selected for this group (segment order).
    pub(super) picked: Vec<PathBuf>,
    /// Full recording in segment order — chapters for GoPro, == picked for OSV.
    pub(super) members: Vec<PathBuf>,
    /// Short human label for the prompt (e.g. "GoPro · 4 chapters").
    pub(super) label: String,
}

impl ImportGroup {
    pub(super) fn mergeable(&self) -> bool { self.members.len() > 1 }
}

/// An import awaiting the user's merge/individual decision (modal).
pub(super) struct PendingImport {
    pub(super) groups: Vec<ImportGroup>,
    /// First picked path — re-activated after the import completes.
    pub(super) first: PathBuf,
}

pub(super) fn raw_take_of_one_lens(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> bool {
    vr180_pipeline::source_kind::is_single_eye(kind, path) && kind.is_raw_video()
}

/// A raw take without sound leaves its folder as the "audio path": there
/// is nothing to open.
pub(super) fn take_audio_missing(audio_path: &std::path::Path) -> bool {
    audio_path.is_dir()
}

/// Raw video takes are FOLDERS. Turn what the user opened into clip paths:
/// a take's folder, or one of its own files, becomes the folder; any other
/// folder stands for the takes directly inside it; everything else passes
/// through. A take that cannot be opened (never closed by the camera, an
/// incomplete copy, no frames) is left out and named in the returned note.
pub(super) fn resolve_raw_takes(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Option<String>) {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut take = |dir: PathBuf, out: &mut Vec<PathBuf>, notes: &mut Vec<String>| {
        match vr180_pipeline::x6_raw::describe(&dir) {
            Ok(_) => { if !out.contains(&dir) { out.push(dir); } }
            Err(e) => {
                let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                tracing::warn!("import: raw take {}: {e}", dir.display());
                notes.push(format!("{name}: {}", e.to_string().trim_start_matches("ffmpeg: ").rsplit(": ").next().unwrap_or_default()));
            }
        }
    };
    for p in paths {
        if let Some(dir) = vr180_pipeline::x6_raw::take_dir(&p) {
            take(dir, &mut out, &mut notes);
        } else if p.is_dir() {
            let mut inside: Vec<PathBuf> = std::fs::read_dir(&p).into_iter().flatten().flatten()
                .map(|e| e.path())
                .filter(|c| c.is_dir() && vr180_pipeline::x6_raw::is_take(c))
                .collect();
            inside.sort();
            if inside.is_empty() {
                notes.push(format!("{}: no raw video takes in this folder", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
            }
            for dir in inside { take(dir, &mut out, &mut notes); }
        } else if !out.contains(&p) {
            out.push(p);
        }
    }
    // A take and its archive copy are one recording: opened together, the take is the one that is opened.
    let (out, left) = without_copies_of_opened_takes(out, |p| vr180_pipeline::x6_raw::archive_note(p).and_then(|note| note.source));
    for copy in &left {
        tracing::info!("import: {} is an archive copy of a take opened with it: left out", copy.display());
        notes.push(format!("{}: an archive copy of a take that is opened with it, left out", clip_label(copy)));
    }
    (out, (!notes.is_empty()).then(|| notes.join("\n")))
}

/// An async clip load in flight (see `App::load_pending`). The worker
/// thread does all the blocking source I/O and posts a [`LoadOutcome`]
/// back on `rx`; the UI thread shows a "Loading…" state until it arrives.
pub(super) struct LoadPending {
    pub(super) token: u64,
    pub(super) path: PathBuf,
    pub(super) rx: Receiver<LoadOutcome>,
    pub(super) started: std::time::Instant,
}

/// Result of the off-thread metadata read: either the resolved clip
/// metadata to apply, or a human-readable error string to log.
pub(super) type LoadOutcome = std::result::Result<LoadedMeta, String>;

/// One BRAW eye at load: the layout's (memoised per clip; side by side
/// unless the first frame holds one centred circle), from the probe's
/// composed frame size.
pub(super) fn braw_eye_dims(path: &std::path::Path, frame_w: u32, frame_h: u32) -> (u32, u32) {
    match vr180_braw::BrawInfo::probe(path) {
        Ok(info) => vr180_pipeline::braw_imu::braw_layout(path, &info).eye_dims(&info),
        Err(_) => (frame_w / 2, frame_h),
    }
}

/// Everything `finish_load` needs to build the `ClipInfo`, seed per-clip
/// settings, and spawn the decoder — all gathered off the UI thread.
pub(super) struct LoadedMeta {
    pub(super) token: u64,
    pub(super) path: PathBuf,
    pub(super) preserve_clip_settings: bool,
    pub(super) source_kind: vr180_pipeline::SourceKind,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fps: f32,
    pub(super) duration_sec: f64,
    pub(super) frame_count: u32,
    /// See [`ClipInfo::holes`].
    pub(super) holes: Option<Vec<vr180_pipeline::frame_timeline::Hole>>,
    pub(super) segments: Vec<PathBuf>,
    pub(super) cori_count: usize,
    pub(super) grav_count: usize,
    pub(super) srot_ms: Option<f32>,
    pub(super) detected_rs_mode: crate::decoder::RsMode,
    pub(super) fisheye_eye_w: u32,
    pub(super) fisheye_eye_h: u32,
    pub(super) eac_cal_dim: Option<u32>,
    /// Insta360 X6 recorded in "360° Audio" mode: the source carries a
    /// 4-channel first-order AmbiX track (so the Audio section applies).
    pub(super) foa_audio: bool,
    /// Kandao QooCam 4-channel microphone track (see `ClipInfo::quad_audio`).
    pub(super) quad_audio: bool,
    /// The stream is HLG (the QooCam's HDR mode) — auto-apply the HLG LUT.
    pub(super) hlg: bool,
    /// Blackmagic RAW: the clip's embedded display LUT written out as a
    /// `.cube` by the helper (off the UI thread), to auto-apply.
    pub(super) braw_lut: Option<String>,
    /// Blackmagic RAW: the SDK's colour processing as shot (the probe's
    /// `color` block), for the Color panel.
    pub(super) braw_color: Option<vr180_braw::BrawColorInfo>,
    /// Insta360 X6 video: recording-mode signature + picture profile.
    pub(super) x6: Option<X6ClipInfo>,
    /// A raw take of one lens ([`raw_take_of_one_lens`]).
    pub(super) raw_one_lens: bool,
    /// A raw take's frame in its lens's view ([`RawFrame`]).
    pub(super) raw_frame: Option<RawFrame>,
    /// See `ClipInfo::video_bitrates_bps` / `file_bitrate_bps`.
    pub(super) video_bitrates_bps: Vec<u64>,
    pub(super) file_bitrate_bps: Option<u64>,
}

impl App {
    // ─── File loading ────────────────────────────────────────────

    pub(super) fn pick_and_load_file(&mut self, ctx: &egui::Context) {
        self.spawn_dialog(ctx, || {
            Self::video_file_dialog().pick_files().map(DialogResult::Videos)
        });
    }

    /// Open one or more files: fisheye sources join the clip list (and
    /// the first picked becomes active); legacy non-fisheye sources
    /// (GoPro `.360` EAC) bypass the list and load directly, exactly as
    /// before — they aren't batchable.
    pub(super) fn open_paths(&mut self, paths: Vec<PathBuf>) {
        // A project file is opened as one (the first, if several came).
        let is_project = |p: &PathBuf| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(crate::project::EXTENSION));
        if let Some(project) = paths.iter().find(|p| is_project(p)) {
            self.request_project(ProjectAction::Open(project.clone()));
            return;
        }
        let (paths, notice) = resolve_raw_takes(paths);
        self.import_notice = notice;
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
    pub(super) fn group_imports(&self, paths: &[PathBuf]) -> Vec<ImportGroup> {
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
    pub(super) fn apply_import(&mut self, groups: Vec<ImportGroup>, merge: bool, first: PathBuf) {
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
    pub(super) fn clip_segments(&self, path: &std::path::Path) -> Vec<PathBuf> {
        self.merge_groups.get(path).cloned()
            .unwrap_or_else(|| vr180_core::segments::detect_segments(path))
    }

    /// Modal asking whether to merge a multi-file recording. Shown while
    /// `pending_import` is `Some`; resolves to `apply_import(merge)`.
    pub(super) fn draw_import_prompt(&mut self, ctx: &egui::Context) {
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

    pub(super) fn load_file(&mut self, path: PathBuf) {
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
    pub(super) fn unload_clip(&mut self, preserve_clip_settings: bool) {
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
            self.settings.ettr_stops = 0.0;
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
    pub(super) fn load_file_inner(&mut self, path: PathBuf, preserve_clip_settings: bool) {
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
        self.arm_session_guard();

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
    pub(super) fn load_clip_metadata(
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
            vr180_pipeline::decode::VideoProbe { width: w, height: h, fps: 1.0, duration_sec: 0.0, hlg: false, video_bitrates_bps: [0; 2] }
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
            | vr180_pipeline::SourceKind::X6Raw
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
        // Frames actually in the track: a clip with dropped frames has fewer
        // than duration × fps (the X6 leaves holes where the sensor dropped
        // frames). BRAW and stills have no MP4 sample table — keep the count.
        let timeline = if source_kind == vr180_pipeline::SourceKind::BlackmagicRaw || source_kind.is_still() {
            None
        } else {
            vr180_pipeline::frame_timeline::FrameTimeline::from_segments(&segments).filter(|t| !t.is_empty())
        };
        let frame_count = timeline.as_ref().map(|t| t.len() as u32).unwrap_or(frame_count);
        // The same table says where frames are missing (file info, timeline).
        let holes = timeline.as_ref().map(|t| t.holes());
        if let Some(t) = timeline.as_ref().filter(|t| t.dropped_frames() > 0) {
            tracing::info!("load: frame timeline — {}", t.summary());
        }

        // Bit rate for the file info: the video streams' own rates come with
        // the probe; the whole clip's bytes over its duration stand in where
        // the container states none (Blackmagic RAW). File sizes only.
        let file_bitrate_bps = (!source_kind.is_still() && duration_sec > 0.0).then(|| {
            let bytes: u64 = segments.iter().filter_map(|s| std::fs::metadata(s).ok()).map(|m| m.len()).sum();
            (bytes as f64 * 8.0 / duration_sec) as u64
        }).filter(|b| *b > 0);
        tracing::info!(
            "load: bit rate — video streams {:?} bps, whole file {:?} bps",
            probe.video_bitrates_bps, file_bitrate_bps
        );

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
        let foa_audio = match source_kind {
            vr180_pipeline::SourceKind::Insta360Insv =>
                vr180_pipeline::audio::probe_foa_aac(&vr180_pipeline::x6_pair::audio_source(&path).0).ok().flatten().is_some(),
            // A raw take's sound is its own file beside the capture.
            vr180_pipeline::SourceKind::X6Raw => vr180_pipeline::x6_raw::audio_source(&path)
                .is_some_and(|(file, _)| vr180_pipeline::audio::probe_foa_aac(&file).ok().flatten().is_some()),
            _ => false,
        };
        if foa_audio { tracing::info!("load: X6 360° Audio track (4-ch AmbiX) detected"); }
        // Kandao QooCam: a 4-channel AAC of its microphone capsules (not a
        // sound field) — mixed to stereo by the player and the export.
        let quad_audio = source_kind == vr180_pipeline::SourceKind::KandaoQoocam
            && vr180_pipeline::audio::probe_foa_aac(&path).ok().flatten().is_some();
        if quad_audio { tracing::info!("load: QooCam 4-channel microphone track detected"); }
        // Insta360 X6: the trailer's window record (the recording mode's
        // signature — 7.4K / 10K and their Crop variants share nothing else
        // that tells them apart) and the picture profile. A clip whose
        // trailer cannot be read still gets an entry: its mode then follows
        // the frame size alone. A few KB from the end of the file.
        let x6 = (source_kind == vr180_pipeline::SourceKind::Insta360Insv).then(|| {
            let (info, gamma) = X6ClipInfo::read(&path);
            let g = info.geometry(probe.width, probe.height, vr180_fisheye::X6ViewChoice::Auto);
            tracing::info!(
                "load: X6 recording mode {:?} ({} × {}, window {:?}, gamma {:?})",
                g.detected, probe.width, probe.height,
                info.window.map(|w| (w.dst_w, w.dst_h, w.offset_x, w.offset_y)), gamma,
            );
            info
        });
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
        let raw_one_lens = raw_take_of_one_lens(source_kind, &path);
        let raw_frame = RawFrame::read(source_kind, &path);
        Ok(LoadedMeta {
            token, path, preserve_clip_settings, source_kind, raw_one_lens, raw_frame,
            width: probe.width, height: probe.height, fps: probe.fps,
            duration_sec, frame_count, holes, segments,
            cori_count, grav_count, srot_ms,
            detected_rs_mode, fisheye_eye_w, fisheye_eye_h,
            eac_cal_dim, foa_audio, quad_audio, hlg: probe.hlg, braw_lut, braw_color, x6,
            video_bitrates_bps: probe.video_bitrates_bps.iter().copied().filter(|b| *b > 0).collect(), file_bitrate_bps,
        })
    }

    /// Apply a finished async load on the UI thread: seed the per-clip
    /// settings, publish `ClipInfo`, and spawn the (paused) decoder. By now
    /// the worker has already read the source, so it's warm — `spawn_decoder`
    /// and its audio open don't block.
    pub(super) fn finish_load(&mut self, meta: LoadedMeta) {
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
                | vr180_pipeline::SourceKind::X6Raw
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
                // The I-Log LUT follows the file's picture profile
                // (`gamma_mode`), never the frame size: every recording
                // mode can be shot in another profile. A file that does not
                // state one is taken as I-Log, as before.
                vr180_pipeline::SourceKind::Insta360Insv
                    if takes_ilog(source_kind, meta.x6.as_ref()) => Some(BUILTIN_X6_LUT_PATH),
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
            s.tone = Default::default();
            s.temperature = 0.0; s.tint = 0.0; s.saturation = 1.0;
            if let (Some(info), Some(lut)) = (meta.braw_color.as_ref(), meta.braw_lut.as_deref()) {
                if !vr180_braw::BrawColorInfo::is_film_gamma(s.braw_color.effective_gamma(info)) && s.lut_path == lut { s.lut_path.clear(); }
            }
        }
        // The ETTR recovery needs the source's log curve: a clip that is not
        // I-Log holds none, whatever route its settings came by.
        if !takes_ilog(source_kind, meta.x6.as_ref()) {
            self.settings.ettr_stops = 0.0;
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
            x6: meta.x6,
            raw_one_lens: meta.raw_one_lens,
            raw_frame: meta.raw_frame,
            video_bitrates_bps: meta.video_bitrates_bps,
            file_bitrate_bps: meta.file_bitrate_bps,
            holes: meta.holes,
        });
        self.loaded_path = Some(meta.path);
        // A single-eye clip has nothing to put on a 3D display.
        if self.clip_single_eye() && self.three_d.enabled {
            self.disable_3d_display();
        }
        // The decoder about to start is built for the current view choice.
        self.insv_view_applied = self.settings.insv_view;

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
    pub(super) fn poll_load(&mut self, ctx: &egui::Context) {
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
    pub(super) fn spawn_decoder(&mut self, start_paused: bool) {
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
        let worker_settings = self.worker_settings();
        let cfg = DecoderConfig {
            path,
            segments,
            settings: worker_settings.clone(),
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
            settings: parking_lot::RwLock::new(worker_settings),
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
            trim_looped_to: parking_lot::Mutex::new(None),
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
            // A raw take: the sound file beside the capture, from the
            // first frame's time in it (nothing to play when it has none).
            let raw_take = self.clip.as_ref().is_some_and(|c| c.source_kind == vr180_pipeline::SourceKind::X6Raw);
            let take_audio = if raw_take { vr180_pipeline::x6_raw::audio_source(&audio_path) } else { None };
            let (audio_path, audio_base_s) = match take_audio {
                Some(a) => a,
                None => vr180_pipeline::x6_pair::audio_source(&audio_path),
            };
            // (A camera mounted upside down: the sound's sides change places with the picture's.)
            let turned = self.settings.camera_upside_down && self.clip.as_ref().is_some_and(|c| c.source_kind.is_x6_video());
            self.audio_player = if raw_take && take_audio_missing(&audio_path) { None } else { match crate::audio_player::AudioPlayer::open_at(
                audio_path, start_paused, mix, audio_base_s, turned,
            ) {
                Ok(Some(player)) => Some(player),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("audio_player: open failed: {e}");
                    None
                }
            } };
            tracing::info!("load_timing: AudioPlayer::open @ {:?}", t_audio.elapsed());
        }
    }

    /// The shared open-dialog filter for every "add video" entry point.
    pub(super) fn video_file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new()
            .add_filter(
                "All supported video",
                &["360", "osv", "OSV", "insv", "INSV", "braw", "BRAW",
                  "mp4", "MP4", "mov", "MOV", "dng", "DNG", "x6b", "X6B"],
            )
            .add_filter("Camera still (.dng: Insta360 X6, Kandao QooCam 3 Ultra)", &["dng", "DNG"])
            .add_filter("DJI Osmo 360 (.osv)", &["osv", "OSV"])
            .add_filter("Insta360 (.insv)", &["insv", "INSV"])
            .add_filter("Insta360 X6 raw video (capture.x6b in the take's folder)", &["x6b", "X6B"])
            .add_filter("Kandao QooCam 3 Ultra (.mp4)", &["mp4", "MP4"])
            .add_filter("Blackmagic RAW (.braw)", &["braw", "BRAW"])
            .add_filter("Side-by-side fisheye (.mp4 / .mov)",
                &["mp4", "MP4", "mov", "MOV"])
    }
}
