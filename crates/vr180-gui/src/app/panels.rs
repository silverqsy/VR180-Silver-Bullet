//! The side panels for the source, stabilization, sound and rolling shutter, and the slider they share with the rest.

use super::*;

impl App {
    pub(super) fn draw_source_panel(&mut self, ui: &mut egui::Ui) {
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

    pub(super) fn draw_stab_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn draw_audio_panel(&mut self, ui: &mut egui::Ui) {
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

    pub(super) fn draw_rs_panel(&mut self, ui: &mut egui::Ui) {
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
}

pub(super) fn fine_slider(
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
