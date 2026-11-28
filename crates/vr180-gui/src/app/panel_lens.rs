//! The panels for the lens and the view: view adjustment, alignment, the fisheye source and its output.

use super::*;

impl App {
    pub(super) fn draw_view_adjust_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn start_auto_align(&mut self) {
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
    pub(super) fn poll_auto_align(&mut self, ctx: &egui::Context) {
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
    pub(super) fn draw_fisheye_stab_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn lens_space_ui(ui: &mut egui::Ui, s: &mut crate::decoder::Settings, kind: Option<vr180_pipeline::SourceKind>) {
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
    pub(super) fn draw_fisheye_output_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn draw_eac_lens_panel(&mut self, ui: &mut egui::Ui) {
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
    pub(super) fn draw_fisheye_panel(&mut self, ui: &mut egui::Ui) {
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
pub(super) fn draw_eye_lens(
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
