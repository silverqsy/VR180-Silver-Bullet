//! The clip list as a project file, and the last session.

use super::*;

/// A clip of a project, looked for and read on a thread: the entry it is
/// for, and where its file and the files of its recording are with what
/// their headers say (`None`: not found, or not a readable source).
pub(super) struct RestoredClip {
    pub(super) token: u64,
    pub(super) found: Option<Located>,
}

/// Where a project's clip is now and what its file says.
pub(super) struct Located {
    pub(super) path: PathBuf,
    /// The files of its recording, where it is made of several.
    pub(super) segments: Option<Vec<PathBuf>>,
    pub(super) facts: ClipFacts,
    /// `path` is an archive copy read in the place of the take the project
    /// names, which is not at hand.
    pub(super) stand_in: Option<StandIn>,
}

/// Where a project's clip is now, and what its file says: the file the
/// project names, or, for a raw take that is not there, an archive copy of
/// it beside where it was or in `also` (the output folder). Touches the
/// disk.
pub(super) fn locate_clip(
    file: &crate::project::FileRef,
    segments: Option<&[crate::project::FileRef]>,
    base: Option<&std::path::Path>,
    also: Option<&std::path::Path>,
) -> Option<Located> {
    let Some(path) = file.resolve(base) else {
        // Where the take was when the project was written, and where that is from the project file now.
        let moved = base.zip(file.relative.as_deref()).map(|(base, relative)| crate::project::join_relative(base, relative));
        return [Some(file.path.clone()), moved].into_iter().flatten().find_map(|take| {
            let (path, note) = copy_standing_in(&take, also)?;
            let facts = ClipFacts::read(&path)?;
            let extent = CopyExtent::of(&note, facts.duration_sec);
            // (The entry's trim is put in when the entry takes this in.)
            let stand_in = StandIn::of(take, extent, (None, None), facts.fps);
            Some(Located { path, segments: None, facts, stand_in: Some(stand_in) })
        });
    };
    // A recording of several files is there when all of them are.
    let segments = match segments {
        None => None,
        Some(files) => Some(files.iter().map(|f| f.resolve(base)).collect::<Option<Vec<_>>>()?),
    };
    let facts = ClipFacts::read(&path)?;
    Some(Located { path, segments, facts, stand_in: None })
}

/// What replaces the list.
#[derive(Clone)]
pub(super) enum ProjectAction {
    /// An empty list.
    New,
    /// The list of this project file.
    Open(PathBuf),
}

/// New / Open, held until the user has said what becomes of the list's
/// unsaved changes.
pub(super) struct PendingProject {
    pub(super) action: ProjectAction,
    /// "Save project…" was chosen: the action runs once the file is written.
    pub(super) after_save: bool,
}

// ─── The list as a project file, and the last session ───────────────

impl App {
    /// The last session's list, and whether its clip stays off the screen
    /// (`project::take_session`).
    pub(super) fn read_session() -> (Option<Project>, bool) {
        match crate::project::config_dir() {
            Some(dir) => crate::project::take_session(&dir),
            None => (None, false),
        }
    }

    /// The list and the output choices as a project whose file is in the
    /// folder `base` (`None`: no folder to name the clips from).
    pub(super) fn project_snapshot(&self, base: Option<&std::path::Path>) -> Project {
        use crate::project::{FileRef, ProjectClip};
        let clips = self.batch.iter().enumerate().map(|(i, item)| match &item.offline {
            // As it was read: its file may only be away for now.
            Some(off) => ProjectClip {
                file: if base.is_some() { FileRef::new(&off.file.path, base) } else { off.file.clone() },
                segments: off.segments.clone(),
                selected: item.selected,
                settings: item.settings.clone(),
            },
            None => {
                // The sidebar edits the clip on screen in `self.settings`.
                let live = self.active_clip == Some(i) && self.loaded_path.as_ref() == Some(&item.path);
                // (An entry that reads a copy in its take's place is saved as the take, with its trim there.)
                let (file, settings) = item.as_saved(if live { &self.settings } else { &item.settings });
                ProjectClip {
                    file: FileRef::new(&file, base),
                    segments: self.merge_groups.get(&item.path)
                        .map(|files| files.iter().map(|f| FileRef::new(f, base)).collect()),
                    selected: item.selected,
                    settings,
                }
            }
        }).collect();
        Project {
            version: crate::project::VERSION,
            written_by: format!("VR180 Silver Bullet {}", env!("CARGO_PKG_VERSION")),
            clips,
            active: self.active_clip,
            join_clips: self.join_clips,
            output_folder: self.batch_out_dir.clone(),
            export: self.export_opts,
            project_file: None,
        }
    }

    /// The folder the open project's clips are named from.
    pub(super) fn project_base(&self) -> Option<PathBuf> {
        self.project_path.as_ref().and_then(|p| p.parent()).map(|d| d.to_path_buf())
    }

    /// The list has changes that no project file holds. (Which clip is on
    /// screen is not one.)
    pub(super) fn project_dirty(&self) -> bool {
        // While a project's clips are still being looked for, the list is
        // the file's.
        if self.batch.is_empty() || self.restore_owed > 0 {
            return false;
        }
        let Some(saved) = &self.project_saved else { return true };
        let mut now = self.project_snapshot(self.project_base().as_deref());
        now.active = saved.active;
        now != *saved
    }

    /// Replace the list with `project`'s (`file`: the project file it was
    /// belongs to, if any). The clips show at once, in their places; their
    /// files are looked for and read on a thread ([`Self::poll_restore`]),
    /// so that a network drive that has to wake does not hold the window.
    /// `autoload`: the project's clip goes on screen when it is found.
    /// `from_file`: `project` was read from `file` just now (the session's
    /// copy of a project may hold changes the file does not).
    pub(super) fn apply_project(&mut self, project: Project, file: Option<PathBuf>, autoload: bool, from_file: bool) {
        self.unload_clip(false);
        self.batch.clear();
        self.merge_groups.clear();
        self.active_clip = None;
        self.pending_import = None;
        self.import_notice = None;
        self.last_batch_summary = None;
        self.join_notice = None;
        self.restore_activate = None;
        self.restore_clicked = None;
        self.export_opts = project.export.settled();
        self.join_clips = project.join_clips;
        self.batch_out_dir = project.output_folder.clone();
        self.project_path = file;
        self.project_saved = None;
        let base = self.project_base();
        let mut requests = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (i, clip) in project.clips.iter().enumerate() {
            // One entry per file, as everywhere in the list.
            if !seen.insert(clip.file.path.clone()) {
                continue;
            }
            self.restore_seq += 1;
            let token = self.restore_seq;
            self.batch.push(BatchItem::offline(clip, token, base.clone()));
            let request = (token, clip.file.clone(), clip.segments.clone(), base.clone());
            if autoload && project.active == Some(i) {
                // The clip that goes on screen is looked for first.
                self.restore_activate = Some(token);
                requests.insert(0, request);
            } else {
                requests.push(request);
            }
        }
        self.restore_autoload = autoload;
        self.restore_from_file = from_file;
        tracing::info!("project: {} clips{}", self.batch.len(),
            self.project_path.as_ref().map(|p| format!(" from {}", p.display())).unwrap_or_default());
        if requests.is_empty() {
            self.restore_done();
        } else {
            self.look_for(requests);
        }
    }

    /// Look for clips of a project on a thread; the answers arrive in
    /// [`Self::poll_restore`].
    #[allow(clippy::type_complexity)]
    pub(super) fn look_for(&mut self, requests: Vec<(u64, crate::project::FileRef, Option<Vec<crate::project::FileRef>>, Option<PathBuf>)>) {
        self.restore_owed += requests.len();
        let tx = self.restore_tx.clone();
        let ctx = self.egui_ctx.clone();
        // (Where an archive copy may lie besides next to its take.)
        let also = self.batch_out_dir.clone();
        std::thread::spawn(move || {
            for (token, file, segments, base) in requests {
                // A reader that gives up on a file must not leave its entry
                // "looking" for good: it counts as not found.
                let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    locate_clip(&file, segments.as_deref(), base.as_deref(), also.as_deref())
                })).unwrap_or(None);
                if tx.send(RestoredClip { token, found }).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
    }

    /// Look again for a clip that was not found: the user clicked it.
    pub(super) fn look_again(&mut self, idx: usize) {
        let token = self.restore_seq + 1;
        let Some(off) = self.batch.get_mut(idx).and_then(|b| b.offline.as_mut()).filter(|o| !o.looking) else { return };
        off.looking = true;
        off.token = token;
        let request = (token, off.file.clone(), off.segments.clone(), off.base.clone());
        self.restore_seq = token;
        self.restore_clicked = Some(token);
        self.look_for(vec![request]);
    }

    /// Take in the clips that have been looked for.
    pub(super) fn poll_restore(&mut self, ctx: &egui::Context) {
        while let Ok(restored) = self.restore_rx.try_recv() {
            self.restore_owed = self.restore_owed.saturating_sub(1);
            self.take_restored(restored);
            if self.restore_owed == 0 {
                self.restore_done();
            }
        }
        if self.restore_owed > 0 {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    pub(super) fn take_restored(&mut self, restored: RestoredClip) {
        let token = restored.token;
        // The entry may be gone: removed while it was looked for, or the
        // list replaced.
        let Some(idx) = self.batch.iter().position(|b| b.offline.as_ref().is_some_and(|o| o.token == token)) else { return };
        let Some(Located { path, segments, facts, stand_in }) = restored.found else {
            if let Some(off) = self.batch[idx].offline.as_mut() {
                off.looking = false;
            }
            tracing::warn!("project: {} was not found", self.batch[idx].path.display());
            return;
        };
        // The same file came into the list while this entry was looked for.
        if self.batch.iter().any(|b| b.ready() && b.path == path) {
            self.remove_clip(idx);
            return;
        }
        let fps = facts.fps;
        self.batch[idx].take_facts(path.clone(), facts);
        // The take is not at hand and a copy of it is read in its place: the project's trim, set on the take,
        // lies in the copy where the copy lies in the take.
        if let Some(found) = stand_in {
            let item = &mut self.batch[idx];
            let stand_in = StandIn::of(found.take, found.extent, (item.settings.trim_in_s, item.settings.trim_out_s), fps);
            (item.settings.trim_in_s, item.settings.trim_out_s) = stand_in.trim_on_copy;
            tracing::info!("project: {} is not at hand: its archive copy {} is read in its place", stand_in.take.display(), path.display());
            item.stand_in = Some(stand_in);
        }
        if let Some(files) = segments {
            self.merge_groups.insert(path, files);
        }
        if self.restore_clicked == Some(token) {
            self.restore_clicked = None;
            self.select_clip(idx);
        } else if self.restore_activate == Some(token) {
            self.restore_activate = None;
            self.show_restored(idx);
        }
    }

    /// Put a project's clip on screen, unless the user has put one there
    /// meanwhile.
    pub(super) fn show_restored(&mut self, idx: usize) {
        if self.active_clip.is_none() && self.loaded_path.is_none() && self.load_pending.is_none() {
            self.select_clip(idx);
        }
    }

    /// Every clip of the project has been looked for.
    pub(super) fn restore_done(&mut self) {
        self.restore_activate = None;
        if std::mem::take(&mut self.restore_autoload) {
            if let Some(idx) = self.batch.iter().position(BatchItem::ready) {
                self.show_restored(idx);
            }
        }
        let missing = self.batch.iter().filter(|b| !b.ready()).count();
        if missing > 0 {
            tracing::warn!("project: {missing} of {} clips were not found", self.batch.len());
        }
        // The takes that are not at hand and are read from their archive copies: said once.
        let zh = cur_lang() == Lang::Zh;
        let standing: Vec<&BatchItem> = self.batch.iter().filter(|b| b.stand_in.is_some()).collect();
        match standing.as_slice() {
            [] => {}
            [one] => self.import_notice = one.stand_in.as_ref().map(|s| stand_in_notice(&s.take, &one.path, true, zh)),
            many => self.import_notice = Some(stand_ins_notice(many.len(), zh)),
        }
        // What the project file holds is the list as it now stands.
        if self.project_path.is_some() && std::mem::take(&mut self.restore_from_file) {
            self.project_saved = Some(self.project_snapshot(self.project_base().as_deref()));
        }
    }

    /// New / Open: at once when the list holds nothing unsaved, else after
    /// the user's word ([`Self::draw_project_prompt`]). Not while exporting.
    pub(super) fn request_project(&mut self, action: ProjectAction) {
        if self.batch_running || self.export_job.is_some() {
            return;
        }
        if self.project_dirty() {
            self.pending_project = Some(PendingProject { action, after_save: false });
        } else {
            self.run_project_action(action);
        }
    }

    pub(super) fn run_project_action(&mut self, action: ProjectAction) {
        match action {
            ProjectAction::New => {
                // A new list; how things are exported stays as it is set.
                let blank = Project { export: self.export_opts, output_folder: self.batch_out_dir.clone(), ..Default::default() };
                self.apply_project(blank, None, false, false);
            }
            ProjectAction::Open(path) => match crate::project::load::<ExportOptions>(&path) {
                Ok(project) => self.apply_project(project, Some(path), true, true),
                Err(e) => {
                    tracing::warn!("project: {e}");
                    self.snapshot_status = Some((e, std::time::Instant::now()));
                }
            },
        }
    }

    /// Write the list to the project file `path` (its extension added
    /// where none was typed), which becomes the open project.
    pub(super) fn save_project_to(&mut self, path: PathBuf) {
        let path = if path.extension().is_none() { path.with_extension(crate::project::EXTENSION) } else { path };
        self.sync_active_clip_settings();
        let snapshot = self.project_snapshot(path.parent());
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match crate::project::save(&snapshot, &path) {
            Ok(()) => {
                tracing::info!("project: saved {}", path.display());
                self.project_path = Some(path);
                self.project_saved = Some(snapshot);
                self.snapshot_status = Some((format!("{} {name}", tr("Saved")), std::time::Instant::now()));
                if let Some(pending) = self.pending_project.take() {
                    if pending.after_save {
                        self.run_project_action(pending.action);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("project: {e}");
                self.snapshot_status = Some((e, std::time::Instant::now()));
                self.pending_project = None;
            }
        }
    }

    /// Save: to the open project's file, or ask where.
    pub(super) fn save_project(&mut self, ctx: &egui::Context) {
        match self.project_path.clone() {
            Some(path) => self.save_project_to(path),
            None => self.pick_project_save(ctx),
        }
    }

    pub(super) fn pick_project_open(&mut self, ctx: &egui::Context) {
        let dir = self.project_base();
        self.spawn_dialog(ctx, move || {
            let mut dialog = rfd::FileDialog::new().add_filter("VR180 Silver Bullet project", &[crate::project::EXTENSION]);
            if let Some(dir) = dir {
                dialog = dialog.set_directory(dir);
            }
            dialog.pick_file().map(DialogResult::ProjectOpen)
        });
    }

    pub(super) fn pick_project_save(&mut self, ctx: &egui::Context) {
        // Offered: the open project's own name and folder, else the first
        // clip's.
        let first = self.batch.first().map(|b| b.path.clone());
        let dir = self.project_base().or_else(|| first.as_ref().and_then(|p| p.parent()).map(|d| d.to_path_buf()));
        let name = self.project_path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                let stem = first.as_ref().and_then(|p| p.file_stem()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "project".into());
                format!("{stem}.{}", crate::project::EXTENSION)
            });
        self.spawn_dialog(ctx, move || {
            let mut dialog = rfd::FileDialog::new()
                .add_filter("VR180 Silver Bullet project", &[crate::project::EXTENSION])
                .set_file_name(name);
            if let Some(dir) = dir {
                dialog = dialog.set_directory(dir);
            }
            dialog.save_file().map(DialogResult::ProjectSave)
        });
    }

    /// The question asked before New / Open replaces a list with unsaved
    /// changes.
    pub(super) fn draw_project_prompt(&mut self, ctx: &egui::Context) {
        if !self.pending_project.as_ref().is_some_and(|p| !p.after_save) {
            return;
        }
        let (mut save, mut discard, mut cancel) = (false, false, false);
        egui::Window::new(tr("Replace the clip list?"))
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(420.0)
            .show(ctx, |ui| {
                ui.label(tr("The clip list has changes that are not saved in a project file."));
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button(RichText::new(tr("Save project…")).strong()).clicked() { save = true; }
                    ui.add_space(6.0);
                    if ui.button(tr("Don't save")).clicked() { discard = true; }
                    ui.add_space(6.0);
                    if ui.button(tr("Cancel")).clicked() { cancel = true; }
                });
            });
        if save {
            if let Some(pending) = self.pending_project.as_mut() {
                pending.after_save = true;
            }
            self.save_project(ctx);
        } else if discard {
            if let Some(pending) = self.pending_project.take() {
                self.run_project_action(pending.action);
            }
        } else if cancel {
            self.pending_project = None;
        }
    }

    /// Keep the session file in step with the list: looked at every second
    /// and a half, and at once when the window closes.
    pub(super) fn persist_session(&mut self, ctx: &egui::Context) {
        let closing = ctx.input(|i| i.viewport().close_requested());
        let every = std::time::Duration::from_millis(1500);
        let since = self.session_checked_at.elapsed();
        if !closing && since < every {
            // Whatever drew this frame may have changed the list, and a
            // window that then sits still draws no further one: come back
            // when the look is due, so that what was done last is not left
            // unwritten. (That frame is the look itself and asks for
            // nothing more: a still window stays still.)
            ctx.request_repaint_after(every - since + std::time::Duration::from_millis(50));
            return;
        }
        self.write_session();
    }

    /// Write the session file if the list is not what it holds.
    pub(super) fn write_session(&mut self) {
        self.session_checked_at = std::time::Instant::now();
        // Not before the last session has been put back: it would be
        // written over with an empty list.
        if self.pending_session.is_some() {
            return;
        }
        let mut session = self.project_snapshot(self.project_base().as_deref());
        session.project_file = self.project_path.clone();
        if self.session_saved.as_ref() == Some(&session) {
            return;
        }
        let Some(path) = crate::project::session_path() else { return };
        match crate::project::save(&session, &path) {
            Ok(()) => {
                self.session_saved = Some(session);
                self.session_error = None;
            }
            // Not marked as written: the next look tries again (a file that
            // was held by another program for a moment must not cost the
            // last changes).
            Err(e) => {
                if self.session_error.as_ref() != Some(&e) {
                    tracing::warn!("session: {e}");
                }
                self.session_error = Some(e);
            }
        }
    }

    /// A clip load starts: leave the mark that a launch which finds it
    /// reads as "died loading" (`project::GuardStage`).
    pub(super) fn arm_session_guard(&mut self) {
        use crate::project::GuardStage;
        if self.session_guard != Some(GuardStage::Loading) {
            GuardStage::Loading.leave();
            self.session_guard = Some(GuardStage::Loading);
        }
    }

    /// The clip shows, or the app closes in order: the mark goes.
    pub(super) fn clear_session_guard(&mut self) {
        if self.session_guard.take().is_some() {
            crate::project::GuardStage::clear();
        }
    }

    /// The marks' upkeep, once a frame: a clip that shows has loaded, and
    /// a list that has been drawn a few times is back.
    pub(super) fn tend_session_guard(&mut self) {
        use crate::project::GuardStage;
        if self.session_guard == Some(GuardStage::Loading) && self.current_display.is_some() {
            self.clear_session_guard();
        }
        if let Some(frames) = self.session_restore_frames.as_mut() {
            *frames += 1;
            if *frames >= 3 {
                self.session_restore_frames = None;
                if self.session_guard == Some(GuardStage::Restoring) {
                    self.clear_session_guard();
                }
            }
        }
    }
}
