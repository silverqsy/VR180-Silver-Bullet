//! The Colour panel: the grade, the tone controls, a raw clip's development, Blackmagic RAW colour, and noise reduction.

use super::*;

/// The most the ETTR recovery takes back, in stops. Four stops over, a 100 %
/// white already sits at 16 of the 22 the I-Log scale reaches at code 1.0:
/// nothing shot brighter than that on purpose has highlights left.
pub(super) const ETTR_MAX_STOPS: f32 = 4.0;

impl App {
    /// Take over a raw clip's Camera Raw settings once they are read: they
    /// become the clip's raw development when it is still the clip loaded.
    pub(super) fn poll_raw_settings(&mut self, ctx: &egui::Context) {
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
    pub(super) fn denoise_supported() -> bool {
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

    pub(super) fn draw_color_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn draw_denoise_panel(&mut self, ui: &mut egui::Ui) {
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
pub(super) fn draw_tone_section(ui: &mut egui::Ui, tone: &mut vr180_pipeline::tone_linear::ToneSettings) {
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

pub(super) fn draw_raw_develop_section(
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
pub(super) fn camera_raw_status(found: &Option<vr180_pipeline::x6_raw::CameraRawSettings>) -> String {
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
pub(super) fn draw_braw_color_section(
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
pub(super) fn seed_tone_curve(p: &mut vr180_braw::BrawColorParams, base: vr180_braw::TonePreset) {
    p.saturation.get_or_insert(base.saturation);
    p.contrast.get_or_insert(base.contrast);
    p.midpoint.get_or_insert(base.midpoint);
    p.highlights.get_or_insert(base.highlights);
    p.shadows.get_or_insert(base.shadows);
    p.black_level.get_or_insert(base.black_level);
    p.white_level.get_or_insert(base.white_level);
    p.video_black_level.get_or_insert(base.video_black_level);
}
