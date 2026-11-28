//! The second window for a 3D display.

use super::*;

/// External **3D display** output: a second, chrome-less viewport that shows
/// the SBS preview with the left eye in the left half and the right eye in
/// the right half — the stereo input a side-by-side 3D monitor or AR glasses
/// (one wide screen such as 3840×1080) take. See `draw_3d_display`.
#[derive(Default)]
pub(super) struct ThreeDDisplay {
    pub(super) enabled: bool,
    pub(super) displays: Vec<crate::displays::DisplayInfo>,
    /// Index into `displays` the output sits on; `None` → the windowed
    /// fallback (drag it onto the 3D screen, `F` = fullscreen).
    pub(super) target: Option<usize>,
    /// Current fullscreen state (`F` toggles it in the output window).
    pub(super) fullscreen: bool,
    /// Auto-placed output wants NATIVE fullscreen (macOS with separate
    /// Spaces per display — a borderless window would sit under that
    /// display's menu bar). It is requested in a second step, once the
    /// borderless window is confirmed on the target screen, so macOS
    /// fullscreens THAT screen (asking for it at creation fullscreened
    /// the main screen and covered the app).
    pub(super) native_fullscreen: bool,
    /// The two-step fullscreen request has been sent.
    pub(super) fullscreen_sent: bool,
    /// Settings changed on enable, restored on disable.
    pub(super) restore_preview_w: Option<u32>,
    pub(super) restore_preview_mode: Option<crate::decoder::PreviewMode>,
}

impl App {
    /// Source panel = the CLIP LIST (click to select → preview + sidebar
    /// edit that clip's settings) + the active clip's metadata below.
    /// Turn the external 3D display output on: pick the screen, force the
    /// plain SBS compose, and render each eye at half that screen's width.
    pub(super) fn enable_3d_display(&mut self) {
        use crate::decoder::PreviewMode;
        let displays = crate::displays::all_displays();
        let target = crate::displays::pick_3d_display(&displays);
        let d = &mut self.three_d;
        d.displays = displays;
        d.target = target;
        d.native_fullscreen = target.is_some() && crate::displays::displays_have_separate_spaces();
        d.fullscreen_sent = false;
        d.fullscreen = false;
        d.restore_preview_mode = None;
        d.restore_preview_w = None;
        // Stereo needs the plain L|R compose — anaglyph / overlay / single
        // eye are alignment aids for the main window only.
        if self.settings.preview_mode != PreviewMode::Sbs {
            d.restore_preview_mode = Some(self.settings.preview_mode);
            self.settings.preview_mode = PreviewMode::Sbs;
        }
        // Render each eye at half the screen width so the output is 1:1
        // (a 768-px preview eye stretched to 1920 px looks soft). The eye
        // size binds at decoder start, so this reloads the clip.
        let mut reload = false;
        if let Some(t) = target {
            let want = (d.displays[t].pixel_w / 2).clamp(512, 2048);
            if self.settings.preview_eye_w < want {
                d.restore_preview_w = Some(self.settings.preview_eye_w);
                self.settings.preview_eye_w = want;
                reload = true;
            }
        }
        d.enabled = true;
        match target {
            Some(t) => tracing::info!("3D display: on → {} ({}) at {:?}, {}×{} pt, {}",
                d.displays[t].label(), d.displays[t].name, d.displays[t].origin,
                d.displays[t].size.x, d.displays[t].size.y,
                if d.native_fullscreen { "native fullscreen once placed (separate Spaces)" } else { "borderless at bounds" }),
            None => tracing::info!("3D display: on — no 3840×1080-class screen connected, windowed fallback"),
        }
        if reload {
            if let Some(p) = self.loaded_path.clone() { self.load_file_inner(p, true); }
        }
    }

    pub(super) fn disable_3d_display(&mut self) {
        let d = &mut self.three_d;
        d.enabled = false;
        if let Some(m) = d.restore_preview_mode.take() { self.settings.preview_mode = m; }
        let mut reload = false;
        if let Some(w) = d.restore_preview_w.take() {
            self.settings.preview_eye_w = w;
            reload = true;
        }
        tracing::info!("3D display: off");
        if reload {
            if let Some(p) = self.loaded_path.clone() { self.load_file_inner(p, true); }
        }
    }

    /// The external 3D display viewport: a chrome-less window on the chosen
    /// screen (or a movable fallback) painting the SBS preview texture with
    /// the left eye fitted into the left half and the right eye into the
    /// right half. Immediate viewport — it renders as part of this frame, so
    /// it follows every preview update; not calling it closes the window.
    pub(super) fn draw_3d_display(&mut self, ctx: &egui::Context) {
        if !self.three_d.enabled { return; }
        let id = egui::ViewportId::from_hash_of("vr180_3d_display");
        let target = self.three_d.target.and_then(|t| self.three_d.displays.get(t).cloned());
        let mut builder = egui::ViewportBuilder::default()
            .with_title(format!("VR180 Silver Bullet — {}", tr("3D display")));
        let want_native_fs = self.three_d.native_fullscreen && !self.three_d.fullscreen_sent;
        builder = match &target {
            // Always created as a borderless window on the screen's exact
            // bounds. macOS with "Displays have separate Spaces" ON (the
            // default): that display carries its own menu bar and the window
            // gets pushed under it, so once it is confirmed on the target
            // screen we ask for native fullscreen (below) — which macOS
            // applies to the screen the window is on. Requesting fullscreen
            // at creation fullscreened the MAIN screen instead. With the
            // setting OFF the borderless window is the final state: native
            // fullscreen there makes its own Space and the desktop flips.
            Some(d) => builder
                .with_position(d.origin)
                .with_inner_size(d.size)
                .with_decorations(false)
                .with_resizable(false),
            None => builder.with_inner_size([1280.0, 360.0]),
        };
        let frame = self.current_display.as_ref().map(|d| (d.egui_id, d.width, d.height));
        let matched = self.matched_picture(false);
        let windowed = target.is_none();
        let fullscreen_now = self.three_d.fullscreen;
        let mut close = false;
        let mut toggle_fullscreen = false;
        let mut go_native_fs = false;
        ctx.show_viewport_immediate(id, builder, |ctx, _class| {
            // Step two of the separate-Spaces path: the window exists and
            // its centre is on the target screen → native fullscreen there.
            if want_native_fs {
                if let Some(d) = &target {
                    let on_target = ctx.input(|i| i.viewport().outer_rect)
                        .map(|r| egui::Rect::from_min_size(d.origin, d.size).contains(r.center()))
                        .unwrap_or(false);
                    if on_target { go_native_fs = true; }
                }
            }
            // Windows, mixed per-monitor DPI: the builder's logical position
            // is resolved against the wrong monitor's scale at creation (the
            // window landed on the MAIN screen when the primary ran at 200%),
            // so correct the placement afterwards against the window's OWN
            // reported scale, in physical pixels. Converges in ≤2 frames:
            // the first move lands the window on the target screen (its
            // scale then becomes the target's), the next pass fixes the
            // size under that scale and then matches exactly. No-op once
            // the physical rect is right; macOS keeps the builder placement
            // (CG points are the native space there — never entered).
            #[cfg(target_os = "windows")]
            if let Some(d) = &target {
                let (cur, ppp) = ctx.input(|i| {
                    let vp = i.viewport();
                    let ppp = vp.native_pixels_per_point.unwrap_or_else(|| i.pixels_per_point());
                    (vp.outer_rect.zip(vp.inner_rect), ppp)
                });
                if let Some((outer, inner)) = cur {
                    let pos_ok = (outer.min.x * ppp - d.origin_px.x).abs() <= 1.5
                        && (outer.min.y * ppp - d.origin_px.y).abs() <= 1.5;
                    let size_ok = (inner.width() * ppp - d.pixel_w as f32).abs() <= 1.5
                        && (inner.height() * ppp - d.pixel_h as f32).abs() <= 1.5;
                    if !pos_ok || !size_ok {
                        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(
                            egui::pos2(d.origin_px.x / ppp, d.origin_px.y / ppp)));
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                            egui::vec2(d.pixel_w as f32 / ppp, d.pixel_h as f32 / ppp)));
                        ctx.request_repaint();
                    }
                }
            }
            if ctx.input(|i| i.viewport().close_requested()) { close = true; }
            ctx.input(|i| {
                if i.key_pressed(egui::Key::Escape) { close = true; }
                // F = native fullscreen, windowed fallback only (the
                // auto-placed output already covers its screen).
                if windowed && i.key_pressed(egui::Key::F) && i.modifiers.is_none() { toggle_fullscreen = true; }
            });
            if fullscreen_now || !windowed { ctx.set_cursor_icon(egui::CursorIcon::None); }
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(Color32::BLACK))
                .show(ctx, |ui| {
                    let full = ui.max_rect();
                    let half = egui::vec2(full.width() * 0.5, full.height());
                    let halves = [
                        egui::Rect::from_min_size(full.min, half),
                        egui::Rect::from_min_size(egui::pos2(full.min.x + half.x, full.min.y), half),
                    ];
                    match frame {
                        Some((tex, w, h)) if w > 0 && h > 0 => {
                            // The SBS texture is L|R: each eye is half its
                            // width. Fit each eye into its half of the screen
                            // (exact fill for a 16:9 reframed eye on a
                            // 3840×1080 screen; letterboxed otherwise).
                            let eye_aspect = (w as f32 * 0.5) / h as f32;
                            for (i, r) in halves.iter().enumerate() {
                                let fit = fit_aspect(*r, eye_aspect);
                                let u0 = i as f32 * 0.5;
                                let uv = egui::Rect::from_min_max(
                                    egui::pos2(u0, 0.0), egui::pos2(u0 + 0.5, 1.0));
                                if let Some((picture, encoding)) = &matched {
                                    use crate::display_colour::PaintSlot;
                                    ui.painter().add(crate::display_colour::PreviewPaint {
                                        picture: picture.clone(),
                                        slot: if i == 0 { PaintSlot::ThreeDLeft } else { PaintSlot::ThreeDRight },
                                        uv, encoding: *encoding,
                                    }.shape(fit));
                                } else {
                                    let sized = egui::load::SizedTexture::new(tex, fit.size());
                                    egui::Image::new(sized).uv(uv).paint_at(ui, fit);
                                }
                            }
                        }
                        _ => {
                            ui.centered_and_justified(|ui| {
                                ui.label(RichText::new(tr("3D display — play a clip to see it here"))
                                    .size(16.0).color(Color32::GRAY));
                            });
                        }
                    }
                    if windowed && !fullscreen_now {
                        ui.painter().text(
                            full.left_top() + egui::vec2(10.0, 8.0), egui::Align2::LEFT_TOP,
                            tr("Drag this window onto the 3D screen · F = fullscreen · Esc = close"),
                            egui::FontId::proportional(13.0), Color32::from_rgb(200, 200, 200));
                    }
                });
        });
        if go_native_fs {
            self.three_d.fullscreen_sent = true;
            self.three_d.fullscreen = true;
            tracing::info!("3D display: window placed on the target screen — going native fullscreen there");
            ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Fullscreen(true));
        }
        if toggle_fullscreen {
            self.three_d.fullscreen = !self.three_d.fullscreen;
            ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Fullscreen(self.three_d.fullscreen));
        }
        if close { self.disable_3d_display(); }
    }
}
