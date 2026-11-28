//! The preview: the frames the worker sends, the timeline, and the settings sent to the worker.

use super::*;

/// What a clip's statistics were sampled for: its path, and what makes its
/// decoder give other pictures (`App::clip_stats_for`).
pub(super) type ClipStatsFor = (PathBuf, String);
/// Where a sampling's outcome arrives.
pub(super) type ClipStatsRx = Receiver<std::result::Result<Arc<vr180_pipeline::clip_stats::ClipStats>, String>>;

/// What a raw clip's automatic exposure is doing, as its controls show it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum RawAutoShown {
    /// Not in use.
    Off,
    /// The clip is being metered: the part done, 0..1 (`None`: about to start).
    Metering(Option<f32>),
    /// The exposure of the frame on screen, and the least and the most over
    /// the clip, as the Exposure control would read them.
    Now { stops: f32, range: (f32, f32) },
}

pub(super) struct DisplayFrame {
    pub(super) texture: Arc<wgpu::Texture>,
    pub(super) egui_id: egui::TextureId,
    /// The same picture for `display_colour::PreviewPainter`: made the
    /// first time the frame is drawn with the display match on.
    pub(super) matched_picture: Option<wgpu::BindGroup>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frame_idx: u32,
    pub(super) timestamp_s: f64,
}

/// Largest rect of `aspect` (w/h) centred inside `outer`.
pub(super) fn fit_aspect(outer: egui::Rect, aspect: f32) -> egui::Rect {
    let (ow, oh) = (outer.width(), outer.height());
    let (w, h) = if ow / oh.max(1e-3) > aspect { (oh * aspect, oh) } else { (ow, ow / aspect.max(1e-3)) };
    egui::Rect::from_center_size(outer.center(), egui::vec2(w, h))
}

#[derive(Default)]
pub(super) struct FpsStats {
    pub(super) frames_in_window: u32,
    pub(super) last_window_start: Option<std::time::Instant>,
    pub(super) last_fps: f32,
}

impl App {
    /// `…/VR180SilverBullet2.0/preview_colour` — the display-match switch.
    pub(super) fn match_apple_playback_config_path() -> Option<PathBuf> {
        crate::decoder::Settings::config_path()
            .and_then(|p| p.parent().map(|d| d.join("preview_colour")))
    }

    /// Off unless the user turned it on, and off wherever it cannot work.
    pub(super) fn load_match_apple_playback() -> bool {
        crate::display_colour::available()
            && Self::match_apple_playback_config_path()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .is_some_and(|s| s.trim() == "apple")
    }

    pub(super) fn save_match_apple_playback(&self) {
        if let Some(p) = Self::match_apple_playback_config_path() {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, if self.match_apple_playback { "apple" } else { "plain" });
        }
    }

    /// The preview is drawn as Apple's players show the export right now.
    pub(super) fn display_match_on(&self) -> bool {
        self.match_apple_playback && crate::display_colour::available()
    }

    /// What the loaded clip's picture is made of: a developed photograph is
    /// sRGB, everything else is a frame of a video exported as Rec.709.
    pub(super) fn preview_encoding(&self) -> crate::display_colour::PictureEncoding {
        use crate::display_colour::PictureEncoding;
        if self.clip.as_ref().is_some_and(|c| c.source_kind.is_still()) {
            PictureEncoding::Srgb
        } else {
            PictureEncoding::Rec709Video
        }
    }

    /// The live frame (or the full-resolution still) as the display-matched
    /// painter takes it, with how to read its values. `None` with the match
    /// off or nothing to show: the caller then draws egui's image as before.
    pub(super) fn matched_picture(&mut self, full_res: bool)
        -> Option<(wgpu::BindGroup, crate::display_colour::PictureEncoding)>
    {
        if !self.display_match_on() { return None; }
        let encoding = self.preview_encoding();
        if self.preview_painter.is_none() {
            let painter = crate::display_colour::PreviewPainter::new(
                &self.pipeline.device, self.surface_format);
            // The paint callback finds it in the renderer's resources.
            self.egui_renderer.write().callback_resources.insert(painter.clone());
            self.preview_painter = Some(painter);
        }
        let painter = self.preview_painter.as_ref()?;
        let frame = if full_res { self.full_res_display.as_mut() } else { self.current_display.as_mut() }?;
        if frame.matched_picture.is_none() {
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.preview_view_format),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
            frame.matched_picture = Some(painter.bind_picture(&self.pipeline.device, &view));
        }
        frame.matched_picture.clone().map(|picture| (picture, encoding))
    }

    /// Drive the full-resolution still used by the zoom magnifier. While
    /// paused + zoomed on a fisheye clip, the current frame is decoded +
    /// re-projected at native source resolution and shown in the magnifier
    /// (the live preview decodes at a capped working size). No-op otherwise.
    ///
    /// The native render runs SYNCHRONOUSLY on this (main) thread — see
    /// `DetailCache` for why a background thread deadlocks against eframe's
    /// renderer on the shared Metal device. To hide the resulting one-frame
    /// hitch it is DEBOUNCED: it fires only once the desired (frame,
    /// settings) key has been stable for the settle window. During an active
    /// drag/scrub the key keeps changing, so the preview shows the live
    /// low-res frame and we skip the heavy render until you stop.
    pub(super) fn poll_full_res(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        // EAC (.360): the native-res zoom still is produced by the decoder
        // thread (it already holds the native cross) and ferried in via
        // `detail_frame_rx` — not the fisheye main-thread `DetailCache`.
        if self.clip.as_ref().map(|c| c.source_kind.is_eac()).unwrap_or(false) {
            self.poll_full_res_eac(ctx);
            return;
        }
        // Keep the detail cache alive whenever we're paused on a fisheye clip
        // — independent of zoom. Zoom only gates whether we RENDER a still,
        // not whether the cache (and its decoded native frame) exists. That
        // way zooming out and back in reuses the already-decoded frame
        // instead of re-decoding it (a GOP walk at native res), which showed
        // up as a fresh slowdown on every zoom-in.
        let keep_alive = self.decoder_alive
            && !self.playing
            && self.clip.as_ref().map(|c| c.source_kind.is_fisheye()).unwrap_or(false);

        if !keep_alive {
            if let Some(prev) = self.full_res_display.take() {
                self.egui_renderer.write().free_texture(&prev.egui_id);
            }
            self.full_res_key = 0;
            self.full_res_desired_key = 0;
            self.detail_last_key = 0;
            self.detail_cache = None; // close the native decoder
            return;
        }

        let (path, kind, fps) = match (&self.loaded_path, &self.clip) {
            (Some(p), Some(c)) => (p.clone(), c.source_kind, c.fps),
            _ => return,
        };
        // Spin up the native-res renderer on entering detail mode. It owns a
        // background decode thread; `ctx` lets that thread poke a repaint
        // when a frame finishes decoding so the still appears promptly.
        // Defer the full-res native decoder until the user actually zooms in.
        // Opening it eagerly on load adds a second 3840² decode of the same
        // file, contending with the preview worker (a slow first frame off an
        // external drive). The cache still persists on zoom-out (keep_alive),
        // so only the FIRST zoom-in pays the open.
        if self.detail_cache.is_none() && self.wants_full_res_still() {
            let segments = self.clip.as_ref()
                .map(|c| c.segments.clone())
                .unwrap_or_default();
            self.detail_cache = Some(crate::decoder::DetailCache::new(
                path, segments, kind, fps, self.settings.effective_swap_eyes(), ctx.clone()));
            self.full_res_key = 0;
            self.full_res_desired_key = 0;
            self.detail_last_key = 0;
        }

        // Only render a still while zoomed in. When zoomed out we keep the
        // cache + the last rendered still (the 1× view uses the live
        // preview), so returning here makes zoom-in instant when nothing
        // changed — and a cheap re-projection (no decode) if settings moved.
        if !self.wants_full_res_still() {
            return;
        }

        // Key identifies the (frame, settings) we want a still for. Zoom is
        // excluded — one still covers the whole frame; the UV sub-rect
        // samples it.
        let ts = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
        let gen = self.control.as_ref()
            .map(|c| c.settings_generation.load(Ordering::SeqCst)).unwrap_or(0);
        let ts_ms = (ts * 1000.0).round() as i64 as u64;
        // A pending snapshot renders at the export eye size — fold it into
        // the key so the still re-renders at that size, and back at native
        // once the snapshot is written.
        let snap_out = self.snapshot_pending.as_ref().map(|r| r.out);
        let snap_salt = snap_out.map(|(w, h)| ((w as u64) << 32) | h as u64).unwrap_or(0);
        let key = (ts_ms.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(gen) ^ snap_salt).max(1);

        self.full_res_desired_key = key;

        // Settle timer: reset whenever the wanted (frame+settings) changes.
        // The settle ONLY guards a frame change (scrub/seek) — that needs a
        // background re-decode, and the live preview is the right thing to
        // show meanwhile. A settings adjustment on a PAUSED frame is handled
        // separately below (no settle, no low-res drop).
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(150);
        if key != self.detail_last_key {
            self.detail_last_key = key;
            self.detail_key_changed_at = std::time::Instant::now();
        }

        // Already showing this exact key → done.
        if key == self.full_res_key {
            return;
        }

        // Is the wanted still for the SAME frame we're already showing at full
        // res? Then only the SETTINGS changed — an adjustment on a paused
        // frame. The DetailCache already has this frame's native pair decoded,
        // so re-projecting it is cheap (GPU project+compose on this thread, no
        // decode). Render it IN PLACE immediately instead of dropping to the
        // low-res live preview — so dragging a calib/view slider while zoomed
        // updates the full-res still live, with no low-res↔full-res swap.
        let same_frame = self.full_res_display.as_ref()
            .map(|d| (d.timestamp_s * 1000.0).round() as i64 as u64 == ts_ms)
            .unwrap_or(false);
        // Frame changed and not settled yet → leave the live preview up while
        // the background decode runs; wake when it settles.
        if !same_frame && self.detail_key_changed_at.elapsed() < SETTLE {
            ctx.request_repaint_after(SETTLE);
            return;
        }

        // Throttle the same-frame (slider-drag) re-render. It runs
        // synchronously on the UI thread at NATIVE res (project + RS quats +
        // compose), so doing it for every pointer event made the sliders feel
        // laggy — each drag tick paid a full-res render. ~8 Hz keeps the
        // still visibly tracking the drag while the slider stays fluid; the
        // trailing poll (key still != full_res_key once the interval elapses)
        // renders the FINAL value, so nothing is ever lost.
        const DRAG_RENDER_INTERVAL: std::time::Duration =
            std::time::Duration::from_millis(120);
        if same_frame && self.full_res_key != 0 {
            let since = self.full_res_rendered_at.elapsed();
            if since < DRAG_RENDER_INTERVAL {
                ctx.request_repaint_after(DRAG_RENDER_INTERVAL - since);
                return;
            }
        }

        // Settled. `render` now does the GPU project + compose on this thread
        // (~10 ms) IF the native frame is already decoded; otherwise it kicks
        // the background decode and returns None (cheap). So calling it every
        // poll while pending is fine — no heavy work on the main thread until
        // the frame is ready. On None we poll again (the worker also pokes a
        // repaint the instant it finishes decoding, so there's no needless
        // wait). This also means a transient decode failure simply retries
        // instead of latching "no still".
        let pipeline = self.pipeline.clone(); // Arc — avoids borrowing self twice
        let settings = self.worker_settings();
        let rendered = self.detail_cache.as_mut().unwrap()
            .render(&pipeline, ts, &settings, 0, snap_out);
        if rendered.is_none() {
            // Decoding (or a transient failure) — keep the live preview up
            // and poll again shortly.
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        if let Some((tex, w, h)) = rendered {
            let mut rend = self.egui_renderer.write();
            if let Some(prev) = self.full_res_display.take() {
                rend.free_texture(&prev.egui_id);
            }
            // Plain Unorm pass-through view (`preview_view_format`): egui-wgpu
            // 0.34 wants gamma/raw-byte samples, so NO sRGB decode here (an
            // sRGB view darkens the preview to sRGB⁻¹(v) vs the export).
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.preview_view_format),
                // wgpu 29 derives a view's usages from the texture (which
                // has STORAGE_BINDING from the compose pass) and rejects an
                // sRGB storage view. This view is only sampled for display,
                // so restrict it to TEXTURE_BINDING.
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
            let id = rend.register_native_texture(
                &self.pipeline.device, &view, wgpu::FilterMode::Linear);
            drop(rend);
            self.full_res_display = Some(DisplayFrame {
                texture: Arc::new(tex), egui_id: id, matched_picture: None,
                width: w, height: h,
                frame_idx: 0, timestamp_s: ts,
            });
            self.full_res_key = key;
            self.full_res_rendered_at = std::time::Instant::now();
        }
    }

    /// EAC (.360) zoom magnifier. While paused + zoomed we ask the decoder
    /// thread for a native-resolution still of the current frame (it already
    /// has the assembled native cross, so this is a re-projection, not a
    /// re-decode) and register the freshest one into `full_res_display`. The
    /// central panel's `use_full_res` check then prefers it over the capped
    /// live preview whenever its timestamp matches the shown frame.
    ///
    /// Mirrors the fisheye `DetailCache` UX (debounce/throttle live on the
    /// decoder side) but reuses the live render path as the single source of
    /// truth, so the still is pixel-identical to the preview at native res.
    pub(super) fn poll_full_res_eac(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        let want = self.decoder_alive && !self.playing && self.wants_full_res_still();
        if let Some(c) = &self.control {
            c.want_detail.store(want, Ordering::SeqCst);
        }
        if !want {
            // Keep the still across a zoom-OUT while still paused (zoom-in
            // reuses it instantly via the timestamp match). Only drop it when
            // playback resumes or the decoder is gone — then it's stale.
            if (self.playing || !self.decoder_alive) && self.full_res_display.is_some() {
                if let Some(prev) = self.full_res_display.take() {
                    self.egui_renderer.write().free_texture(&prev.egui_id);
                }
            }
            return;
        }
        // Pick up the freshest native-res still the decoder shipped (drop
        // any older ones queued behind it).
        if let Some(rx) = &self.detail_frame_rx {
            let mut newest: Option<DecodedFrame> = None;
            while let Ok(f) = rx.try_recv() { newest = Some(f); }
            if let Some(f) = newest {
                let mut rend = self.egui_renderer.write();
                if let Some(prev) = self.full_res_display.take() {
                    rend.free_texture(&prev.egui_id);
                }
                // Plain Unorm pass-through view (`preview_view_format`) — same
                // as the live preview, so no sRGB double-decode vs compose.
                let view = f.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.preview_view_format),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
                let id = rend.register_native_texture(
                    &self.pipeline.device, &view, wgpu::FilterMode::Linear);
                drop(rend);
                self.full_res_display = Some(DisplayFrame {
                    texture: f.texture, egui_id: id, matched_picture: None,
                    width: f.width, height: f.height,
                    frame_idx: f.frame_idx, timestamp_s: f.timestamp_s,
                });
            }
        }
        // Keep a slow poll alive while zoomed + paused so the TRAILING still
        // (the crisp one the decoder emits ~one throttle window after a slider
        // settles, when egui would otherwise have stopped repainting) is always
        // picked up. Cheap — this is a paused, hold-still inspection mode.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }

    /// Drain whatever's in the frame channel, register the newest
    /// frame's texture with egui (freeing the previous one).
    ///
    /// We keep at most ONE frame alive: every received frame replaces
    /// the previous display. Older frames received in the same poll
    /// are dropped — which is the right behavior for "real-time, no
    /// buffering" playback.
    pub(super) fn drain_frames(&mut self, ctx: &egui::Context) {
        // The picture went back to the in point at the trim's out point:
        // the sound goes there too.
        if let Some(t) = self.control.as_ref().and_then(|c| c.take_trim_loop()) {
            if let Some(player) = &self.audio_player {
                player.seek_to(t);
            }
        }
        let Some(rx) = &self.frame_rx else { return; };
        let mut newest: Option<DecodedFrame> = None;
        let mut drained = 0u32;
        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(f) => {
                    self.fps_stats.frames_in_window += 1;
                    newest = Some(f);
                    drained += 1;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                // The worker is gone. `Empty` and `Disconnected` used to look
                // identical here, so a dead decoder was indistinguishable from
                // a momentarily idle one.
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected && self.decoder_alive {
            tracing::warn!("decoder frame channel disconnected — worker is gone");
        }
        if drained > 0 && self.fps_stats.last_window_start.is_none() {
            tracing::info!("ui drain: first frame received (drained {} on first poll)", drained);
        }
        if let Some(f) = newest {
            self.replace_display(f);
            self.decoder_starting = false; // first frame in → load done, ungate respawns
            self.decoder_retried = false;  // a later start failure gets its one retry again
            // No extra repaint: this drain runs at the top of `update`, so
            // the frame is painted by this very pass. Requesting another
            // doubled the repaint rate to the display's, which parked the UI
            // thread in `nextDrawable` with wgpu's fence lock held and
            // stalled every decoder submit by a refresh (see `wake_ui`).
        }
        // FPS aggregation.
        let now = std::time::Instant::now();
        match self.fps_stats.last_window_start {
            None => self.fps_stats.last_window_start = Some(now),
            Some(start) => {
                let dt = now.duration_since(start).as_secs_f32();
                if dt > 0.5 {
                    self.fps_stats.last_fps = self.fps_stats.frames_in_window as f32 / dt;
                    self.fps_stats.frames_in_window = 0;
                    self.fps_stats.last_window_start = Some(now);
                }
            }
        }
    }

    pub(super) fn replace_display(&mut self, frame: DecodedFrame) {
        let is_first = self.current_display.is_none();
        let mut renderer = self.egui_renderer.write();
        // Free the previous texture id, if any.
        if let Some(prev) = self.current_display.take() {
            renderer.free_texture(&prev.egui_id);
        }
        // Register the new texture. Use a view of the SBS texture as
        // the egui-side handle. wgpu::Texture is Arc internally so
        // keeping our own Arc + handing eframe a view is fine.
        // Plain Unorm pass-through view (`preview_view_format`): egui-wgpu 0.34
        // samples display textures as gamma/raw bytes, so no sRGB decode here
        // (an sRGB view darkens the preview to sRGB⁻¹(v) vs the export).
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.preview_view_format),
            // wgpu 29: restrict to TEXTURE_BINDING so the sRGB view isn't
            // treated as a storage view (the SBS texture has STORAGE usage).
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let id = renderer.register_native_texture(
            &self.pipeline.device, &view, wgpu::FilterMode::Linear,
        );
        if is_first {
            tracing::info!("ui: first frame registered with egui (id={:?}, {}×{})",
                id, frame.width, frame.height);
        }
        self.current_display = Some(DisplayFrame {
            texture: frame.texture,
            egui_id: id,
            matched_picture: None,
            width: frame.width,
            height: frame.height,
            frame_idx: frame.frame_idx,
            timestamp_s: frame.timestamp_s,
        });
    }

    /// Toggle play/pause. If no decoder is alive yet, this kicks off
    /// a fresh one. Otherwise it just flips the paused flag — the
    /// decoder thread keeps its position so resume picks up
    /// exactly where pause left it.
    pub(super) fn toggle_play_pause(&mut self) {
        // Pressed during the async-load window (clip not ready, no decoder yet):
        // remember the intent instead of dropping it — `finish_load` starts the
        // decoder playing once the clip is ready. Toggling lets a second press
        // cancel it. (The on-screen Play button is gated to a loaded clip; this
        // is reached via the Space shortcut.)
        if self.load_pending.is_some() {
            self.play_when_ready = !self.play_when_ready;
            return;
        }
        // A photo has no playback: Space only revives a dead render thread,
        // paused (a "playing" still would spin the 8 ms repaint cadence).
        if self.clip.as_ref().map_or(false, |c| c.source_kind.is_still()) {
            if !self.decoder_alive { self.spawn_decoder(true); }
            return;
        }
        if !self.decoder_alive {
            self.spawn_decoder(false); // Play button → start playing
            return;
        }
        if let Some(ctl) = &self.control {
            self.playing = !self.playing;
            ctl.paused.store(!self.playing, Ordering::SeqCst);
            tracing::info!("toggle_play_pause: playing={}", self.playing);
        }
        if let Some(player) = &self.audio_player {
            player.set_playing(self.playing);
        }
    }

    /// Send a Seek command to the running decoder. If no decoder is
    /// alive, lazily spawns one (the seek then becomes the initial
    /// position via the regular pipeline).
    pub(super) fn seek_to(&mut self, target_s: f64) {
        if !self.decoder_alive {
            self.spawn_decoder(false);
        }
        // A jump back at the out point not yet followed is overtaken by this
        // seek: the sound goes where the picture is sent now.
        if let Some(ctl) = &self.control {
            let _ = ctl.take_trim_loop();
        }
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecoderCommand::Seek(target_s.max(0.0)));
        }
        if let Some(player) = &self.audio_player {
            player.seek_to(target_s.max(0.0));
        }
    }

    pub(super) fn set_trim_in(&mut self, t: Option<f64>) {
        // Auto-clear if it goes past out. Don't let in > out.
        let t = t.map(|v| v.max(0.0));
        if let (Some(ti), Some(to)) = (t, self.settings.trim_out_s) {
            if ti >= to { return; }
        }
        self.settings.trim_in_s = t;
    }
    pub(super) fn set_trim_out(&mut self, t: Option<f64>) {
        let t = t.map(|v| v.max(0.0));
        if let (Some(ti), Some(to)) = (self.settings.trim_in_s, t) {
            if to <= ti { return; }
        }
        self.settings.trim_out_s = t;
    }

    /// Render the timeline scrubber: a horizontal bar showing
    /// playhead position, with click/drag to seek and visual
    /// trim-in/out markers.
    pub(super) fn draw_timeline(&mut self, ui: &mut egui::Ui) {
        let total = self.clip.as_ref().map(|c| c.duration_sec).unwrap_or(0.0);
        let can_seek = self.clip.is_some() && total > 0.0;

        // Reserve a strip with room for the track + labels.
        let desired = egui::vec2(ui.available_width(), 22.0);
        let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());

        // Background track.
        let track_y0 = rect.top() + 8.0;
        let track_y1 = rect.bottom() - 4.0;
        let track = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 6.0, track_y0),
            egui::pos2(rect.right() - 6.0, track_y1),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(track, 3.0, Color32::from_rgb(38, 42, 50));

        // Position helpers.
        let track_w = track.width().max(1.0);
        let t_to_x = |t: f64| -> f32 {
            let r = (t / total).clamp(0.0, 1.0) as f32;
            track.left() + r * track_w
        };
        let x_to_t = |x: f32| -> f64 {
            let r = ((x - track.left()) / track_w).clamp(0.0, 1.0) as f64;
            r * total
        };

        // Trim shaded region.
        if can_seek {
            let in_t  = self.settings.trim_in_s.unwrap_or(0.0);
            let out_t = self.settings.trim_out_s.unwrap_or(total);
            let x_in  = t_to_x(in_t);
            let x_out = t_to_x(out_t);
            let shaded = egui::Rect::from_min_max(
                egui::pos2(x_in, track_y0),
                egui::pos2(x_out, track_y1),
            );
            painter.rect_filled(shaded, 3.0, Color32::from_rgb(56, 84, 128));
            // Marker lines.
            if self.settings.trim_in_s.is_some() {
                painter.line_segment(
                    [egui::pos2(x_in, track_y0 - 3.0), egui::pos2(x_in, track_y1 + 3.0)],
                    egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 255)),
                );
            }
            if self.settings.trim_out_s.is_some() {
                painter.line_segment(
                    [egui::pos2(x_out, track_y0 - 3.0), egui::pos2(x_out, track_y1 + 3.0)],
                    egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 255)),
                );
            }

            // Where the camera dropped frames (an X6 clip): a mark as wide as
            // the hole, two pixels at least, over the track's upper part.
            let holes = self.clip.as_ref().and_then(|c| c.x6_holes()).unwrap_or(&[]);
            let dt = self.clip.as_ref().map(|c| 1.0 / c.fps.max(1e-3) as f64).unwrap_or(0.0);
            for hole in holes {
                let x0 = t_to_x(hole.at_s);
                let x1 = t_to_x(hole.at_s + hole.frames as f64 * dt).max(x0 + 2.0);
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(x0, track_y0 - 4.0), egui::pos2(x1, track_y0 + 4.0)),
                    0.0, DROPPED_COLOR);
            }
            // The pointer on a mark says what is missing there.
            if let Some(pos) = resp.hover_pos() {
                let near = holes.iter()
                    .map(|h| (h, (t_to_x(h.at_s) - pos.x).abs().min((t_to_x(h.at_s + h.frames as f64 * dt) - pos.x).abs())))
                    .filter(|(_, d)| *d <= 4.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((hole, _)) = near {
                    resp.clone().on_hover_text_at_pointer(dropped_hole_text(hole, cur_lang() == Lang::Zh));
                }
            }

            // Playhead.
            let cur = self.current_display.as_ref().map(|d| d.timestamp_s).unwrap_or(0.0);
            let x_cur = t_to_x(cur);
            painter.line_segment(
                [egui::pos2(x_cur, track_y0 - 4.0), egui::pos2(x_cur, track_y1 + 4.0)],
                egui::Stroke::new(2.0, Color32::from_rgb(255, 220, 80)),
            );
            painter.circle_filled(
                egui::pos2(x_cur, (track_y0 + track_y1) * 0.5),
                4.0, Color32::from_rgb(255, 220, 80),
            );
        }

        // Click / drag → seek.
        if can_seek {
            if let Some(pos) = resp.interact_pointer_pos() {
                if resp.clicked() || resp.dragged() {
                    let t = x_to_t(pos.x);
                    self.seek_to(t);
                }
            }
        }
    }

    /// Tear down the decoder thread for good. Resets position so the
    /// next Play starts from frame 0.
    pub(super) fn stop_playback(&mut self) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(DecoderCommand::Stop);
        }
        // Make sure the decoder isn't waiting on a Pause flag right
        // before it sees the Stop — unpause so it processes the queue.
        if let Some(ctl) = &self.control {
            ctl.paused.store(false, Ordering::SeqCst);
        }
        self.frame_rx = None;
        self.detail_frame_rx = None;
        self.cmd_tx = None;
        self.control = None;
        self.playing = false;
        self.decoder_alive = false;
        // No live decoder → nothing is "starting". Left set, a Stop (or an
        // export of the loaded clip) before the first frame arrived kept
        // every later Play / Space / seek respawn gated off until a reload.
        self.decoder_starting = false;
        // Drop the audio player: cpal stream stops, worker thread
        // shuts down (Stop command).
        self.audio_player = None;
    }

    /// Re-seed the per-eye Override calibration whenever a NEW clip's in-file
    /// calibration arrives. Without this, loading a different clip while
    /// Override is on would keep the previous clip's factory values. Only
    /// re-seeds eyes currently in Override (Override-off eyes read the in-file
    /// calib directly via the resolver's auto path). The off→on toggle is
    /// handled separately in `draw_eye_lens`.
    pub(super) fn maybe_reseed_fisheye_override(&mut self) {
        let detected = self.control.as_ref().and_then(|c| *c.detected_calib.lock());
        // No control (stopped) or not seeded yet: KEEP the last seed. A
        // same-clip respawn (Restart preview, Play after Stop) passes through
        // `None` for ~100 ms and must not read as a new clip — that
        // overwrote hand-tuned Override values with the factory ones. A new
        // clip resets `fisheye_last_seeded` in `unload_clip`.
        let Some(d) = detected else { return; };
        if Some(d) == self.fisheye_last_seeded {
            return; // same clip's calibration — nothing to do
        }
        self.fisheye_last_seeded = Some(d);
        tracing::info!("RESEED-DBG: detected changed; suppress={} ov_l={} cx_l {:.4}->{:.4} fov_l {:.2}->{:.2}",
            self.suppress_reseed_once, self.settings.fisheye_override_left,
            self.settings.fisheye_cx_norm_left, d.left.cx_norm,
            self.settings.fisheye_fov_deg_left, d.left.fov_deg);
        // Clip-list restore in flight: the settings carry hand-tuned
        // per-clip Override values — keep them instead of re-seeding
        // from the file. One-shot (the next clip change reseeds again).
        if std::mem::take(&mut self.suppress_reseed_once) {
            return;
        }
        if self.settings.fisheye_override_left {
            self.settings.fisheye_fov_deg_left = d.left.fov_deg;
            self.settings.fisheye_cx_norm_left = d.left.cx_norm;
            self.settings.fisheye_cy_norm_left = d.left.cy_norm;
            self.settings.fisheye_k_left = d.left.k;
            self.settings.fisheye_k5_left = d.left.k5;
            self.settings.fisheye_p_left = d.left.p;
        }
        if self.settings.fisheye_override_right {
            self.settings.fisheye_fov_deg_right = d.right.fov_deg;
            self.settings.fisheye_cx_norm_right = d.right.cx_norm;
            self.settings.fisheye_cy_norm_right = d.right.cy_norm;
            self.settings.fisheye_k_right = d.right.k;
            self.settings.fisheye_k5_right = d.right.k5;
            self.settings.fisheye_p_right = d.right.p;
        }
    }

    /// Whether a paused frame should be shown as the native-resolution still:
    /// when the alignment magnifier is zoomed in, and always in the reframed
    /// mode (its viewport shows a small part of the lens, so the live
    /// working-res preview is soft and the magnifier is inert there).
    pub(super) fn wants_full_res_still(&self) -> bool {
        self.preview_zoom > 1.001
            || matches!(self.settings.fisheye_output_mode, crate::decoder::FisheyeOutputMode::Reframe)
            || self.snapshot_pending.is_some()
    }

    /// The loaded clip holds ONE eye (see `ClipInfo::single_eye`).
    pub(super) fn clip_single_eye(&self) -> bool {
        self.clip.as_ref().is_some_and(|c| c.single_eye())
    }

    /// The preview on screen is two eyes side by side (a single-eye clip
    /// always shows its one eye, whatever view mode is remembered).
    pub(super) fn preview_shows_sbs(&self) -> bool {
        self.settings.preview_mode == crate::decoder::PreviewMode::Sbs && !self.clip_single_eye()
    }

    /// The settings a renderer of the loaded clip works from (the preview
    /// decoder, the zoom still): the sidebar's, taken as a single-eye source
    /// takes them when the clip is one. `self.settings` itself never carries
    /// that — it is what gets saved and handed to the next clip.
    pub(super) fn worker_settings(&self) -> Settings {
        let mut s = if self.clip_single_eye() { self.settings.for_single_eye() } else { self.settings.clone() };
        s.ilog_source = self.clip.as_ref().is_some_and(|c| c.ilog());
        s.raw_source = self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video());
        s.raw_camera = s.raw_source.then(|| self.loaded_raw_camera());
        s.clip_stats = self.loaded_clip_stats();
        s
    }

    /// The camera the loaded raw clip is developed as: its take's own white
    /// balance where the take records one, the sensor's reference else (and
    /// for a clip that is no raw take).
    pub(super) fn loaded_raw_camera(&self) -> vr180_pipeline::raw_color::RawCamera {
        match (&self.clip, &self.loaded_path) {
            (Some(clip), Some(path)) if clip.source_kind.is_raw_video() => vr180_pipeline::x6_raw::camera_at(path),
            _ => vr180_pipeline::raw_color::RawCamera::X6,
        }
    }

    /// What the loaded clip's statistics are sampled for: the clip, and
    /// what makes its decoder give other pictures (a Blackmagic RAW clip's
    /// colour settings). `None`: nothing loaded, or a still, which is its
    /// own one frame.
    pub(super) fn clip_stats_for(&self) -> Option<ClipStatsFor> {
        let clip = self.clip.as_ref().filter(|c| !c.source_kind.is_still())?;
        let decode = if clip.source_kind == vr180_pipeline::SourceKind::BlackmagicRaw { format!("{:?}", self.settings.braw_color) } else { String::new() };
        Some((self.loaded_path.clone()?, decode))
    }

    /// The loaded clip's statistics, where they are in.
    pub(super) fn loaded_clip_stats(&self) -> Option<crate::decoder::ClipStatsRef> {
        let what = self.clip_stats_for()?;
        self.clip_stats.as_ref().filter(|(of, _)| *of == what).map(|(_, stats)| stats.clone())
    }

    /// The loaded clip is developed with an automatic exposure: it has to
    /// be metered ([`vr180_pipeline::clip_stats::wants_meter`]).
    pub(super) fn clip_wants_meter(&self) -> bool {
        self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) && self.settings.raw.auto_exposure
    }

    /// What the loaded raw clip's automatic exposure is doing, for its
    /// controls.
    pub(super) fn raw_auto_shown(&self) -> RawAutoShown {
        if !self.clip_wants_meter() {
            return RawAutoShown::Off;
        }
        if let Some(stats) = self.loaded_clip_stats().filter(|s| s.0.has_meter()) {
            let stage = vr180_pipeline::raw_tone::RawDevelop {
                camera: self.loaded_raw_camera(), settings: self.settings.raw,
                measure: vr180_pipeline::raw_tone::Measure::Wait,
            };
            if let Some(auto) = stats.0.auto_exposure(&stage) {
                let t = self.current_display.as_ref().map_or(0.0, |d| d.timestamp_s as f64);
                let (lo, hi) = auto.range();
                return RawAutoShown::Now {
                    stops: stage.exposure_control(auto.stops_at(t)),
                    range: (stage.exposure_control(lo), stage.exposure_control(hi)),
                };
            }
        }
        RawAutoShown::Metering(self.clip_stats_job.as_ref().map(|job| job.3.fraction()))
    }

    /// The loaded clip is rendered by something that goes by the picture's
    /// statistics: a raw clip always (its development), any other when a
    /// tone control that follows the levels is moved.
    pub(super) fn clip_wants_stats(&self) -> bool {
        match &self.clip {
            Some(clip) if clip.source_kind.is_raw_video() => true,
            Some(_) => self.settings.tone.needs_stats(),
            None => false,
        }
    }

    /// Keep the loaded clip's statistics coming: take in a sampling that
    /// has finished, call one off that is for a clip no longer loaded, and
    /// start one when the clip needs them and has none. They are sampled
    /// once per clip, on a thread of their own; until they are in, the
    /// preview measures each frame by itself, as a still does.
    pub(super) fn drive_clip_stats(&mut self, ctx: &egui::Context) {
        let what = self.clip_stats_for();
        if let Some((of, cancel, rx, _)) = &self.clip_stats_job {
            if Some(of) != what.as_ref() {
                cancel.store(true, Ordering::SeqCst);
                self.clip_stats_job = None;
            } else {
                match rx.try_recv() {
                    Ok(Ok(stats)) => {
                        self.clip_stats = Some((of.clone(), crate::decoder::ClipStatsRef(stats)));
                        self.clip_stats_job = None;
                        ctx.request_repaint();
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("clip statistics: {e}: each frame of this clip is measured by itself");
                        self.clip_stats_failed = Some(of.clone());
                        self.clip_stats_job = None;
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        ctx.request_repaint_after(std::time::Duration::from_millis(200));
                    }
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        self.clip_stats_failed = Some(of.clone());
                        self.clip_stats_job = None;
                    }
                }
            }
        }
        let Some(what) = what else {
            self.clip_stats = None;
            return;
        };
        if self.clip_stats.as_ref().is_some_and(|(of, _)| *of != what) {
            self.clip_stats = None;
        }
        // What is there serves unless the clip has to be metered and was
        // not: it is then sampled again, metered (its statistics stay in
        // use meanwhile: they come out the same).
        let metered = self.clip_wants_meter();
        if self.clip_stats.as_ref().is_some_and(|(_, stats)| !metered || stats.0.has_meter()) || self.clip_stats_job.is_some()
            || !self.clip_wants_stats() || self.clip_stats_failed.as_ref() == Some(&what)
        {
            return;
        }
        // Once what it is for has stood for a moment.
        const SETTLED: std::time::Duration = std::time::Duration::from_millis(400);
        let now = std::time::Instant::now();
        match &self.clip_stats_asked {
            Some((of, at)) if *of == what && now >= *at => {}
            Some((of, at)) if *of == what => {
                ctx.request_repaint_after(*at - now);
                return;
            }
            _ => {
                self.clip_stats_asked = Some((what, now + SETTLED));
                ctx.request_repaint_after(SETTLED);
                return;
            }
        }
        let Some(item) = self.batch.iter().find(|b| b.path == what.0) else { return };
        let cfg = Self::build_export_cfg(
            &item.path, std::path::Path::new("/tmp/vr180-clip-stats-unused.mp4"), item.fps, item.source_kind,
            item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height, item.sharp_eye(),
            &self.worker_settings(), &self.export_opts, self.clip_segments(&item.path));
        // An export has the GPU: nothing is sampled while it runs. An
        // export samples the clip it writes by itself, and what it sampled
        // is taken here as it is; else the clip is looked at again shortly.
        if self.batch_running {
            match vr180_pipeline::clip_stats::cached(&cfg).filter(|stats| !metered || stats.has_meter()) {
                Some(stats) => self.clip_stats = Some((what, crate::decoder::ClipStatsRef(stats))),
                None => {
                    const AGAIN: std::time::Duration = std::time::Duration::from_secs(2);
                    self.clip_stats_asked = Some((what, now + AGAIN));
                    ctx.request_repaint_after(AGAIN);
                }
            }
            return;
        }
        let pipeline = match &self.clip_stats_device {
            Some(device) => device.clone(),
            None => {
                let device = self.export_device();
                self.clip_stats_device = Some(device.clone());
                device
            }
        };
        let (tx, rx) = crossbeam_channel::bounded(1);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress = Arc::new(vr180_pipeline::clip_stats::Progress::default());
        let (stop, shown) = (cancel.clone(), progress.clone());
        std::thread::spawn(move || {
            let _ = tx.send(vr180_pipeline::clip_stats::for_clip_shown(&pipeline, &cfg, &stop, Some(&shown)).map_err(|e| e.to_string()));
        });
        self.clip_stats_job = Some((what, cancel, rx, progress));
    }

    /// Diff `self.settings` against `last_pushed_settings`; if changed,
    /// push to the shared control and bump the generation counter so
    /// the decoder rebuilds per-eye bundles on its next iteration.
    pub(super) fn maybe_push_settings(&mut self) {
        // What a raw clip's sharpening counts its radius in: the width of
        // one eye of the frame an export would write now.
        self.settings.export_eye_w = if self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) { self.raw_export_eye_w() } else { 0 };
        // The clip's statistics arriving is a change too: the frame on
        // screen is drawn again by them.
        let clip_stats = self.loaded_clip_stats();
        if self.settings == self.last_pushed_settings && clip_stats == self.last_pushed_clip_stats {
            return;
        }
        self.last_pushed_clip_stats = clip_stats;
        if let Some(ctl) = &self.control {
            *ctl.settings.write() = self.worker_settings();
            ctl.settings_generation.fetch_add(1, Ordering::SeqCst);
            tracing::debug!("settings changed → generation bumped");
        } else if !self.no_decoder_warned && self.loaded_path.is_some() && self.load_pending.is_none() {
            // Once per stop, for a LOADED clip (a load restores settings
            // before its decoder exists — not a lost preview): makes a
            // "preview stopped following the sliders" report diagnosable.
            self.no_decoder_warned = true;
            tracing::warn!("settings changed with no decoder alive — the preview will not follow until it is restarted");
        }
        self.last_pushed_settings = self.settings.clone();
    }

    /// Persist settings to disk when they change. Debounced: writes at most
    /// every ~1.2 s during continuous edits (a slider drag), and flushes
    /// IMMEDIATELY when the window is closing so a last-moment tweak isn't
    /// lost (eframe's own storage is disabled here).
    pub(super) fn persist_settings(&mut self, ctx: &egui::Context) {
        if self.settings == self.last_saved_settings {
            return;
        }
        let closing = ctx.input(|i| i.viewport().close_requested());
        if closing
            || self.last_settings_save_at.elapsed() >= std::time::Duration::from_millis(1200)
        {
            // Per-kind map is the source of truth; the legacy single
            // `settings.json` is kept in sync for the no-clip startup state.
            self.save_active_kind_template();
            Settings::save_kind_map(&self.kind_settings);
            self.settings.save_persisted();
            self.last_saved_settings = self.settings.clone();
            self.last_settings_save_at = std::time::Instant::now();
        }
    }
}

/// Format seconds as `mm:ss.ff` (frame-fraction-style, 2 decimals).
pub(super) fn format_time(sec: f64) -> String {
    if !sec.is_finite() || sec < 0.0 {
        return "--:--.--".to_string();
    }
    let total = sec.max(0.0);
    let m = (total / 60.0) as u32;
    let s = total - (m as f64) * 60.0;
    format!("{:02}:{:05.2}", m, s)
}
