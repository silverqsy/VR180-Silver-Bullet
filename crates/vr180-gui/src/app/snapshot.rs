//! A frame saved from the screen: the picture, and a raw take's frame as DNG.

use super::*;

/// Where a frame saved from the screen goes and what its files are called
/// before what tells them apart: `<source stem>_frame<N>` in the export
/// folder, beside the source when none is set.
pub(super) fn saved_frame_place(source: &std::path::Path, out_dir: Option<&std::path::Path>, frame_idx: u32) -> (PathBuf, String) {
    let dir = out_dir.map(|d| d.to_path_buf())
        .or_else(|| source.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = source.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "frame".into());
    (dir, format!("{stem}_frame{frame_idx}"))
}

/// The first of `name`, `name-2`, `name-3` … that is not `taken`: a frame
/// saved again leaves the files of the time before as they are.
pub(super) fn unused_frame_name(name: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(name) {
        return name.to_owned();
    }
    (2u32..).map(|n| format!("{name}-{n}")).find(|n| !taken(n)).unwrap_or_else(|| name.to_owned())
}

/// The name in `dir` of a frame written as the sensor recorded it: the
/// first of `name`, `name-2` … none of whose files is there.
pub(super) fn unused_original_name(dir: &std::path::Path, name: &str) -> String {
    unused_frame_name(name, |n| (0..2).any(|channel| vr180_pipeline::sensor_dng::lone_frame_file(dir, n, channel).exists()))
}

/// What the status line says of a frame written as the sensor recorded it
/// under `name`: its first file, and what the other one ends in
/// (`TAKE_frame5181_CAM0.dng + _CAM1.dng`).
pub(super) fn original_frame_saved(pair: &vr180_pipeline::sensor_dng::WrittenPair, name: &str) -> String {
    let files: Vec<String> = pair.files.iter()
        .map(|f| vr180_pipeline::sensor_dng::lone_frame_file(std::path::Path::new(""), name, f.0).to_string_lossy().into_owned())
        .collect();
    files.iter().enumerate()
        .map(|(i, file)| if i == 0 { file.as_str() } else { file.strip_prefix(name).unwrap_or(file) })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// A snapshot in flight: the still path renders the paused frame at the
/// export's per-eye size `out`; `poll_snapshot` writes it once it lands.
pub(super) struct SnapshotReq {
    pub(super) out: (u32, u32),
    pub(super) since: std::time::Instant,
}

impl App {
    /// Snapshot (📷 / `S`): save the frame on screen at the EXPORT's
    /// resolution — native / 8K for VR180, the chosen per-eye size for
    /// Reframed — as a 92 % JPEG in the export output folder, named
    /// `<source stem>_frame<N>.jpg`. The full-size render comes from the
    /// zoom-still path, which needs the decoder paused on the frame, so a
    /// snapshot mid-playback pauses there first; `poll_snapshot` writes the
    /// file when the still lands (or falls back to the preview texture if it
    /// never does).
    pub(super) fn request_snapshot(&mut self) {
        if self.current_display.is_none() || self.loaded_path.is_none() {
            self.snapshot_status = Some((tr("Nothing to snapshot yet").to_string(), std::time::Instant::now()));
            return;
        }
        // A still's full-size render IS the export; the zoom-still path that
        // serves snapshots is video-only (no DetailCache for a photo), so a
        // request here would wait out the 15 s timeout and save the
        // working-res preview.
        if self.clip.as_ref().map_or(false, |c| c.source_kind.is_still()) {
            self.snapshot_status = Some((tr("Still photo — use Export for the full-size file").to_string(), std::time::Instant::now()));
            return;
        }
        if self.playing { self.toggle_play_pause(); }
        let out = self.export_eye_dims();
        if let Some(c) = &self.control {
            // EAC: the decoder renders the still — ask for the export size and
            // bump the generation so it re-renders even on an unchanged frame.
            *c.detail_out.lock() = Some(out);
            c.settings_generation.fetch_add(1, Ordering::SeqCst);
        }
        self.snapshot_pending = Some(SnapshotReq { out, since: std::time::Instant::now() });
        self.snapshot_status = Some((format!("{}…", tr("Rendering snapshot")), std::time::Instant::now()));
    }

    /// Write the pending snapshot once the export-size still of the paused
    /// frame is showing; after 15 s without one, save the preview texture
    /// instead and say so.
    pub(super) fn poll_snapshot(&mut self, ctx: &egui::Context) {
        let Some((out, since)) = self.snapshot_pending.as_ref().map(|r| (r.out, r.since)) else { return; };
        let cur = self.current_display.as_ref().map(|d| (d.timestamp_s, d.frame_idx));
        // The still is two eyes side by side, or the one eye of a single-eye clip.
        let still_w = if self.clip_single_eye() { out.0 } else { out.0 * 2 };
        let ready = match (&self.full_res_display, cur) {
            (Some(f), Some((ts, _))) =>
                (f.timestamp_s - ts).abs() < 0.02 && f.width == still_w && f.height == out.1,
            _ => false,
        };
        let timed_out = since.elapsed().as_secs_f32() > 15.0;
        if !ready && !timed_out {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            return;
        }
        self.snapshot_pending = None;
        if let Some(c) = &self.control {
            *c.detail_out.lock() = None;
            c.settings_generation.fetch_add(1, Ordering::SeqCst);
        }
        let frame = if ready { self.full_res_display.as_ref() } else { self.current_display.as_ref() }
            .map(|d| (d.texture.clone(), d.width, d.height));
        let frame_idx = cur.map(|c| c.1).unwrap_or(0);
        if let Some((tex, w, h)) = frame {
            self.write_snapshot(tex, w, h, frame_idx, !ready);
        }
    }

    pub(super) fn write_snapshot(&mut self, tex: Arc<wgpu::Texture>, w: u32, h: u32, frame_idx: u32, preview_fallback: bool) {
        let Some(src) = self.loaded_path.clone() else { return; };
        let (dir, name) = saved_frame_place(&src, self.batch_out_dir.as_deref(), frame_idx);
        let name = unused_frame_name(&name, |n| dir.join(format!("{n}.jpg")).exists());
        let path = dir.join(format!("{name}.jpg"));
        let result = (|| -> anyhow::Result<()> {
            let rgb = self.pipeline.read_texture_rgb8(&tex, w, h)?;
            std::fs::create_dir_all(&dir)?;
            let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 92);
            enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
            Ok(())
        })();
        if result.is_ok() {
            self.spawn_frame_dng(path.with_extension("dng"));
        }
        let msg = match result {
            Ok(()) => {
                tracing::info!("snapshot: {} ({w}×{h}, preview_fallback={preview_fallback})", path.display());
                let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                if preview_fallback {
                    format!("{} {name} ({w}×{h}, {})", tr("Saved"), tr("preview size — full-size still never arrived"))
                } else {
                    format!("{} {name} ({w}×{h})", tr("Saved"))
                }
            }
            Err(e) => {
                tracing::warn!("snapshot failed: {e:#}");
                format!("{}: {e}", tr("Snapshot failed"))
            }
        };
        self.snapshot_status = Some((msg, std::time::Instant::now()));
    }

    /// A raw video clip's snapshot: the frame on screen also as a linear DNG
    /// at `out` (projected and stabilized as the export would, with the
    /// clip's raw development settings inside), written on its own thread.
    pub(super) fn spawn_frame_dng(&mut self, out: PathBuf) {
        let Some(src) = self.loaded_path.clone() else { return; };
        let Some(t_s) = self.current_display.as_ref().map(|d| d.timestamp_s) else { return; };
        let Some(item) = self.batch.iter().find(|b| b.path == src).filter(|b| b.source_kind.is_raw_video()) else { return; };
        if self.frame_dng_job.is_some() {
            return;
        }
        let mut cfg = Self::build_export_cfg(
            &item.path, &out, item.fps, item.source_kind,
            item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height, item.sharp_eye(),
            &self.worker_settings(), &self.export_opts, self.clip_segments(&item.path));
        // The snapshot's picture is the preview's, at the size picked without
        // enhanced frames: its DNG keeps that size (an enhanced frame, where
        // there is one, still gives it its pixels).
        cfg.raw_enhanced_eye = None;
        let pipeline = self.export_device();
        let (tx, rx) = crossbeam_channel::bounded(1);
        let path = out.clone();
        std::thread::spawn(move || {
            let result = vr180_pipeline::raw_export::export_frame_dng(pipeline, &cfg, t_s as f64, &path).map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.frame_dng_job = Some((out, rx));
    }

    /// Original DNG (the transport row's button / Shift+S): the frame on
    /// screen of a raw video clip as the sensor recorded it, one Bayer DNG
    /// per eye, where a snapshot of the frame goes and under its name:
    /// `<source stem>_frame<N>_CAM0.dng` and `…_CAM1.dng` in the export
    /// folder (beside the clip when none is set), `…_frame<N>-2_…` when
    /// the frame was written before. Playback pauses on the frame; the
    /// files are written on their own thread.
    pub(super) fn request_original_dng(&mut self) {
        if self.original_dng_job.is_some() {
            return;
        }
        let shown = self.current_display.as_ref().map(|d| (d.timestamp_s, d.frame_idx));
        let (Some(src), Some((t_s, frame_idx))) = (self.loaded_path.clone(), shown) else {
            self.snapshot_status = Some((tr("Nothing to snapshot yet").to_string(), std::time::Instant::now()));
            return;
        };
        if !self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) {
            return;
        }
        if self.playing { self.toggle_play_pause(); }
        let out_dir = self.batch_out_dir.clone();
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let (dir, name) = saved_frame_place(&src, out_dir.as_deref(), frame_idx);
            // Looked for on this thread: the folder may be on a slow share.
            let name = unused_original_name(&dir, &name);
            let result = vr180_pipeline::sensor_dng::export_sensor_dng_frame(&src, &dir, &name, t_s)
                .map(|pair| original_frame_saved(&pair, &name))
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.original_dng_job = Some(rx);
        self.snapshot_status = Some((format!("{}…", tr("Writing the camera's frame")), std::time::Instant::now()));
    }

    /// Report the camera's frame once it is written.
    pub(super) fn poll_original_dng(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.original_dng_job.as_ref() else { return; };
        match rx.try_recv() {
            Ok(Ok(saved)) => {
                tracing::info!("original DNG: {saved}");
                self.snapshot_status = Some((format!("{} {saved}", tr("Saved")), std::time::Instant::now()));
                self.original_dng_job = None;
            }
            Ok(Err(e)) => {
                tracing::warn!("original DNG failed: {e}");
                self.snapshot_status = Some((format!("{}: {e}", tr("Camera original frame failed")), std::time::Instant::now()));
                self.original_dng_job = None;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => self.original_dng_job = None,
        }
    }

    /// Report the frame DNG of a snapshot once it is written.
    pub(super) fn poll_frame_dng(&mut self, ctx: &egui::Context) {
        let Some((path, rx)) = self.frame_dng_job.as_ref() else { return; };
        let name = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        match rx.try_recv() {
            Ok(Ok(())) => {
                tracing::info!("snapshot: {}", path.display());
                self.snapshot_status = Some((format!("{} {name}", tr("Saved")), std::time::Instant::now()));
                self.frame_dng_job = None;
            }
            Ok(Err(e)) => {
                tracing::warn!("snapshot: frame DNG failed: {e}");
                self.snapshot_status = Some((format!("{name}: {e}"), std::time::Instant::now()));
                self.frame_dng_job = None;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => self.frame_dng_job = None,
        }
    }
}
