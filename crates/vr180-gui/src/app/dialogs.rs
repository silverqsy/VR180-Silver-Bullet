//! File pickers on a worker thread, and the updater's popover.

use super::*;

/// What a worker-thread file picker came back with. One variant per call
/// site; `Cancelled` covers the user dismissing the panel.
pub(super) enum DialogResult {
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

impl App {
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
    pub(super) fn spawn_dialog<F>(&mut self, ctx: &egui::Context, pick: F)
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
    pub(super) fn poll_dialogs(&mut self) {
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

    pub(super) fn poll_updater(&mut self, ctx: &egui::Context) {
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
    pub(super) fn draw_update_popover(&mut self, ctx: &egui::Context) {
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
}
