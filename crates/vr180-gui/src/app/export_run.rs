//! A run of the export queue: each clip's configuration and job, and the export bar.

use super::*;

pub(super) struct ExportJob {
    pub(super) output_path: PathBuf,
    pub(super) handle: std::thread::JoinHandle<vr180_pipeline::Result<()>>,
    pub(super) progress_rx: crossbeam_channel::Receiver<vr180_pipeline::fisheye_export::ExportProgress>,
    pub(super) cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Most recent progress update — refreshed by drain on every UI
    /// frame so the progress bar / counters don't lag.
    pub(super) last_progress: Option<vr180_pipeline::fisheye_export::ExportProgress>,
    pub(super) started_at: std::time::Instant,
}

/// Outcome of a finished export run, shown as a dismissable banner in the
/// export bar (and used to fire the completion notification).
pub(super) struct BatchSummary {
    pub(super) done: usize,
    pub(super) failed: usize,
    pub(super) elapsed: std::time::Duration,
    /// The one file the run's clips were joined into.
    pub(super) joined: Option<PathBuf>,
    /// What the run said of that file (enhanced frames it left out).
    pub(super) note: Option<String>,
    /// The run wrote the takes themselves, and what of them.
    pub(super) take: Option<TakeJob>,
}

/// What a finished run has done, in a few words: its clips exported, or
/// the takes themselves written (`take`).
pub(super) fn run_done_text(done: usize, take: Option<TakeJob>, zh: bool) -> String {
    match (take, zh) {
        (None, _) => format!("{done} exported"),
        (Some(TakeJob::Archive(_)), true) => format!("已制作 {done} 个存档副本"),
        (Some(TakeJob::Archive(_)), false) => format!("{done} archive {} made", if done == 1 { "copy" } else { "copies" }),
        (Some(TakeJob::Originals), true) => format!("已导出 {done} 条素材的相机原片 DNG"),
        (Some(TakeJob::Originals), false) => format!("camera original DNGs of {done} {} written", if done == 1 { "take" } else { "takes" }),
    }
}

/// `out`, or the first of `<stem> (2)`, `<stem> (3)`… that is not there
/// yet: nothing is written over, nothing silently skipped.
pub(super) fn unused_path(out: PathBuf) -> PathBuf {
    if !out.exists() {
        return out;
    }
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("output").to_string();
    // A folder of frames has no extension to keep.
    let ext = out.extension().and_then(|s| s.to_str()).map(|e| format!(".{e}")).unwrap_or_default();
    let ext = if ext.is_empty() && !out.is_dir() { ".mp4".to_string() } else { ext };
    let dir = out.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    (2..).map(|n| dir.join(format!("{stem} ({n}){ext}"))).find(|cand| !cand.exists()).unwrap_or(out)
}

impl App {
    pub(super) fn export_selected(&mut self) {
        if self.export_job.is_some() || self.batch_running { return; }
        self.sync_active_clip_settings();
        // The ticked clips; or, if none are ticked, the active clip alone.
        let mut run: Vec<usize> = self.batch.iter().enumerate()
            .filter(|(_, b)| b.selected && b.source_kind.is_exportable())
            .map(|(i, _)| i)
            .collect();
        if run.is_empty() {
            if let Some(idx) = self.active_clip {
                if self.batch.get(idx).map_or(false, |b| b.source_kind.is_exportable()) {
                    run.push(idx);
                }
            }
        }
        if run.is_empty() {
            tracing::warn!("export: nothing selected to export");
            return;
        }
        self.start_run(run);
    }

    /// Common run setup: freeze the output options, record the run-set +
    /// start time (for overall progress / ETA / completion), kick the first
    /// queued item.
    pub(super) fn begin_run(&mut self, run_indices: Vec<usize>) {
        if !self.batch.iter().any(|b| b.status == BatchStatus::Queued) { return; }
        self.batch_opts = Some(self.export_opts);
        self.batch_run_indices = run_indices;
        self.batch_started_at = Some(std::time::Instant::now());
        self.run_rate_anchor = None;
        self.last_batch_summary = None;
        self.batch_running = true;
        self.batch_skip_flag = false;
        self.start_next_batch_item();
    }

    /// macOS desktop notification when an export run finishes — people walk
    /// away during long batches (more so now that NR triples some clips).
    /// Fires only for multi-item or long (>20 s) runs so a quick single
    /// export doesn't nag. No-op off macOS.
    pub(super) fn notify_run_complete(
        &self,
        done: usize,
        failed: usize,
        elapsed: std::time::Duration,
        n_run: usize,
        take: Option<TakeJob>,
    ) {
        #[cfg(target_os = "macos")]
        {
            if n_run <= 1 && elapsed.as_secs() < 20 {
                return;
            }
            let when = {
                let (m, s) = (elapsed.as_secs() / 60, elapsed.as_secs() % 60);
                if m > 0 { format!("{m}m {s}s") } else { format!("{s}s") }
            };
            let what = run_done_text(done, take, false);
            let body = if failed > 0 {
                format!("{what}, {failed} failed · {when}")
            } else {
                format!("{what} · {when}")
            }
            .replace('"', "'");
            // osascript is always present on macOS; spawn detached so the UI
            // never blocks on it.
            let _ = std::process::Command::new("osascript")
                .arg("-e")
                .arg(format!(
                    "display notification \"{body}\" with title \"VR180 Silver Bullet\" \
                     sound name \"Glass\""
                ))
                .spawn();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (done, failed, elapsed, n_run, take);
        }
    }

    /// Frames one run item contributes to the totals: its TRIMMED length
    /// (every entry carries its own trim), or — for the item currently
    /// exporting — the pipeline's authoritative trim-aware count. Using the
    /// untrimmed clip length here was why a trimmed export's bar stalled
    /// short of 100% and its ETA ran long.
    pub(super) fn run_item_frames(&self, i: usize, running: bool) -> u64 {
        if running {
            if let Some(p) = self.export_job.as_ref().and_then(|j| j.last_progress.as_ref()) {
                if p.total_frames > 0 { return p.total_frames; }
            }
        }
        // (A run that writes the takes themselves may leave their trim out.)
        let whole = self.take_run.is_some_and(|run| run.whole);
        self.batch.get(i).map(|b| {
            let t_in = b.settings.trim_in_s.filter(|_| !whole).unwrap_or(0.0).max(0.0);
            let t_out = b.settings.trim_out_s.filter(|_| !whole).unwrap_or(b.duration_sec).min(b.duration_sec);
            ((t_out - t_in).max(0.0) * b.fps as f64).round().max(1.0) as u64
        }).unwrap_or(1)
    }

    /// `(done_frames, total_frames, running_item_pos)` for the current run;
    /// `None` when no run is in flight.
    pub(super) fn run_frame_tally(&self) -> Option<(u64, u64, usize)> {
        if !self.batch_running || self.batch_run_indices.is_empty() {
            return None;
        }
        let (mut done, mut total, mut item_pos) = (0u64, 0u64, 0usize);
        for (pos, &i) in self.batch_run_indices.iter().enumerate() {
            let status = self.batch.get(i).map(|b| &b.status);
            let running = matches!(status, Some(BatchStatus::Running));
            let frames = self.run_item_frames(i, running);
            total += frames;
            match status {
                Some(BatchStatus::Done | BatchStatus::Failed(_) | BatchStatus::Skipped) => {
                    done += frames;
                }
                Some(BatchStatus::Running) => {
                    item_pos = pos;
                    if let Some(p) = self.export_job.as_ref().and_then(|j| j.last_progress.as_ref()) {
                        done += p.frame_idx.min(frames);
                    }
                }
                _ => {}
            }
        }
        Some((done, total, item_pos))
    }

    /// Overall progress of the current run for the export bar:
    /// `(done_frames, total_frames, eta_secs, item_position, item_total)`.
    /// `None` when no run is in flight. The ETA rate is measured from the
    /// run's first written frame (see `run_rate_anchor`) across items, so
    /// slow-NR and fast plain clips self-correct and start-up time is never
    /// mistaken for encode time; before that anchor exists it falls back to
    /// the live encode rate.
    pub(super) fn overall_run_progress(&self) -> Option<(u64, u64, Option<f64>, usize, usize)> {
        let (done_frames, total_frames, item_pos) = self.run_frame_tally()?;
        let remaining = total_frames.saturating_sub(done_frames);
        let anchored = self.run_rate_anchor.and_then(|(t0, d0)| {
            let secs = t0.elapsed().as_secs_f64();
            (done_frames > d0 && secs > 0.5)
                .then(|| remaining as f64 * secs / (done_frames - d0) as f64)
        });
        let eta = anchored.or_else(|| {
            self.export_job.as_ref()
                .and_then(|j| j.last_progress.as_ref())
                .map(|p| p.fps_avg)
                .filter(|f| *f > 0.1)
                .map(|f| remaining as f64 / f as f64)
        });
        Some((done_frames, total_frames, eta, item_pos + 1, self.batch_run_indices.len()))
    }

    /// `<source_stem>_SBS.<ext-for-codec>` next to the source; a spatial
    /// video (one stream, not a side-by-side frame) is `<stem>_Spatial.mov`,
    /// and a lens frame `<stem>_Lens.<ext>` — the same clip may well be
    /// exported both ways, for two kinds of headset.
    pub(super) fn default_output_for(source: &std::path::Path, codec: ExportCodec, single_eye: bool, lens_frame: bool) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        // A single-eye source's frame is not side by side.
        let frame = if lens_frame { "Lens" } else if single_eye { "Mono" } else { "SBS" };
        let (tag, ext) = match codec {
            ExportCodec::H265 => (frame, "mp4"),
            ExportCodec::ProRes => (frame, "mov"), // ProRes is conventional in MOV
            ExportCodec::Spatial => ("Spatial", "mov"), // Apple's movie profile
        };
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_{tag}.{ext}"))
    }

    /// The folder of a raw video clip's DNG frames: `<source_stem>_DNG`
    /// next to the source (a take is a folder: next to that folder).
    pub(super) fn default_frames_dir_for(source: &std::path::Path, original: bool) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_{}", if original { "OriginalDNG" } else { "DNG" }))
    }

    /// `<source_stem>_SBS.<png|jpg|dng>` next to the source (stills).
    pub(super) fn default_still_output_for(source: &std::path::Path, format: StillFormatChoice) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_SBS.{}", format.extension()))
    }

    /// Build the full `FisheyeExportConfig` for one (source, output) pair
    /// from explicit clip facts + a Settings/ExportOptions pair. Pure —
    /// used by both the single export (current clip + live settings) and
    /// the batch (per-item probe + per-item settings snapshot).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_export_cfg(
        source_path: &std::path::Path,
        output_path: &std::path::Path,
        fps: f32,
        source_kind: vr180_pipeline::SourceKind,
        fisheye_eye_w: u32,
        fisheye_eye_h: u32,
        width: u32,
        height: u32,
        sharp_eye: Option<u32>,
        settings: &crate::decoder::Settings,
        opts: &ExportOptions,
        segments: Vec<PathBuf>,
    ) -> vr180_pipeline::fisheye_export::FisheyeExportConfig {
        // Pick the backend based on the chosen codec + platform. On
        // Windows/Linux, H.265 defaults to NVENC (hardware) — libx265 is
        // the VR180 export bottleneck — but the user can pick software in
        // the options; either way `open_h265_encoder` falls back to libx265
        // if NVENC can't be opened. macOS always uses VideoToolbox.
        use vr180_pipeline::encode::EncoderBackend;
        // Spatial video where this clip can be one; H.265 where it cannot.
        let codec = opts.codec_for(settings, source_kind.is_still());
        let backend = match (codec, cfg!(target_os = "macos")) {
            (ExportCodec::H265 | ExportCodec::Spatial, true) => EncoderBackend::VideoToolbox,
            (ExportCodec::H265 | ExportCodec::Spatial, false) =>
                if opts.h265_hardware {
                    EncoderBackend::HevcNvenc
                } else {
                    EncoderBackend::Libx265
                },
            (ExportCodec::ProRes, true) => EncoderBackend::ProResVideoToolbox,
            (ExportCodec::ProRes, false) => EncoderBackend::ProResKs,
        };

        // Output resolution depends on the projection target. For
        // half-equirect VR180 we use the source eye dims (or fallback
        // for non-fisheye sources). For fisheye pass-through we use
        // the raw source eye dims since the projection is bypassed.
        let native_eye = native_eye_dims(source_kind, fisheye_eye_w, fisheye_eye_h, width, height);
        let (mut eye_w, mut eye_h) = native_eye;
        // 8K target: render the projection itself at 4096² per eye
        // (8192×4096 SBS). The projection samples the native source
        // directly into the 4096 grid — full resolution end-to-end,
        // not a last-step upscale.
        // Native sharpness (`sharp_eye`: a clip whose frame is a crop of
        // the lens' view): the 180° canvas at the frame's own centre
        // density. Such a clip has its own choice; no other has that size.
        match (opts.resolution_for(SizeList::of(sharp_eye.is_some(), false)), sharp_eye) {
            (ExportResolution::Sharp, Some(side)) => { eye_w = side; eye_h = side; }
            (ExportResolution::Native, _) => {}
            _ => { eye_w = 4096; eye_h = 4096; }
        }

        // The Lens Space view through the camera's lens is a lens frame: the
        // fisheye projection, writing the lens's image instead of its disc,
        // at the size the eye was recorded (or the 8K copy).
        let lens_frame_on = settings.lens_frame_view(source_kind);
        let lens_frame = if lens_frame_on {
            (eye_w, eye_h) = opts.lens_frame_eye_dims(native_eye.0, native_eye.1, SizeList::of(sharp_eye.is_some(), false), settings.stabilize, settings.single_eye);
            vr180_pipeline::fisheye_export::LensFrameOut::FromSource
        } else {
            vr180_pipeline::fisheye_export::LensFrameOut::Off
        };
        let projection = settings.export_projection();
        if settings.fisheye_output_mode == crate::decoder::FisheyeOutputMode::Reframe {
            // The exact viewport at the chosen per-eye size and aspect
            // (overrides the VR180 resolution targets above); the
            // still-only sizes never reach a video export.
            let (w, h) = opts.reframe_size
                .clamped(source_kind.is_still(), settings.reframe_aspect)
                .eye_dims(settings.reframe_aspect);
            eye_w = w;
            eye_h = h;
        }
        // Fisheye output is a CIRCLE in a SQUARE frame — force eye_h = eye_w
        // so SBS becomes (2*side × side). For OSV (3840×3840) this is a
        // no-op; for non-square sources it picks the smaller side so
        // the fisheye disk fits.
        // (A lens frame is the recorded frame, of whatever shape.)
        if projection == vr180_pipeline::fisheye_export::FisheyeExportProjection::Fisheye && !lens_frame_on {
            let side = eye_w.min(eye_h);
            eye_w = side;
            eye_h = side;
        }
        // A raw take's enhanced frames (the folder of its camera-original
        // DNGs beside it, when it is there and the option is on) and the
        // size they are written at: the one picked for enhanced frames,
        // which the export takes where its stretch has some. A canvas for
        // the VR180 view and the disc, the frame itself for a lens frame; a
        // reframed view keeps the size it has.
        let raw_enhanced_dir = (source_kind.is_raw_video() && opts.raw_use_enhanced)
            .then(|| Self::originals_dir_for(source_path))
            .filter(|dir| dir.join("CAM0").is_dir());
        let raw_enhanced_eye = raw_enhanced_dir.as_ref()
            .filter(|_| !projection.is_reframe())
            .map(|_| if lens_frame_on {
                opts.lens_frame_eye_dims(native_eye.0, native_eye.1, SizeList::Enhanced, settings.stabilize, settings.single_eye)
            } else {
                let (w, h) = opts.enhanced_eye_dims(native_eye);
                if projection == vr180_pipeline::fisheye_export::FisheyeExportProjection::Fisheye { (w.min(h), w.min(h)) } else { (w, h) }
            });
        let mut color_stack = settings.build_color_stack();
        // Automatic eye-exposure matching is an X6 feature (independent
        // per-sensor AE); never let the persisted flag leak onto other sources.
        if !matches!(source_kind, vr180_pipeline::SourceKind::Insta360Insv) {
            color_stack.auto_eye_match = None;
        }
        // A still bound for a linear DNG: only what the DNG carries as
        // metadata. Decided from THIS export's options (a batch may use a
        // different still format than the live preview flag reflects).
        if source_kind.is_still() && opts.still_format == StillFormatChoice::LinearDng {
            crate::decoder::restrict_to_dng_metadata(&mut color_stack);
        }
        // "BeyondVR Hack" — final-stage per-eye scale. Half-equirect SBS
        // output only; the preview plan (built separately) never sets it.
        if opts.beyondvr_hack
            && !settings.single_eye
            && projection == vr180_pipeline::fisheye_export::FisheyeExportProjection::HalfEquirect
        {
            color_stack.eye_scale = (opts.beyondvr_scale_pct / 100.0).clamp(0.25, 1.0);
        }

        // Bitrate / bit-depth come from the options window. ProRes
        // ignores the bitrate field (the profile picks the rate)
        // and is always 10-bit / 12-bit per profile.
        let (bitrate_kbps, bit_depth) = match codec {
            ExportCodec::H265 | ExportCodec::Spatial => (
                // Reframed exports take the (much lower) reframed rate,
                // recommended per clip from its size / aspect / fps unless
                // the user pinned one; VR180 takes the VR180 slider.
                opts.effective_h265_mbps(settings.reframe_aspect_if_active(), fps, settings.single_eye)
                    .saturating_mul(1000),
                // A spatial video is always Main10; so is a raw video clip
                // (its eyes are 16-bit all the way through the grade).
                if codec == ExportCodec::Spatial || source_kind.is_raw_video() { 10 } else { opts.h265_bit_depth },
            ),
            ExportCodec::ProRes => {
                // The encoder ignores bitrate for ProRes but the config
                // field is non-optional — pass a sane default.
                let pix_is_12bit = matches!(
                    opts.prores_profile,
                    ProResProfile::P4444 | ProResProfile::P4444Xq,
                );
                (0, if pix_is_12bit { 10 } else { 10 })
            }
        };

        vr180_pipeline::fisheye_export::FisheyeExportConfig {
            source_path: source_path.to_path_buf(),
            segments,
            // Reuse the IMU the preview already loaded for this clip (if cached),
            // so exporting a just-previewed file skips the djmd re-read.
            preloaded_imu: crate::decoder::cached_dji_imu(source_path),
            output_path: output_path.to_path_buf(),
            source_kind,
            audio_track: match settings.audio_format {
                crate::decoder::AudioFormat::Stereo    => vr180_pipeline::fisheye_export::AudioTrack::Stereo,
                crate::decoder::AudioFormat::Ambisonic => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
                // No APAC encoder on Windows: a persisted/carried-over Apac
                // becomes Ambisonic here too, so the export does not depend
                // on the Audio panel having been drawn (which is where the
                // picker folds it). The pipeline's own fallback stays as the
                // last net.
                #[cfg(target_os = "macos")]
                crate::decoder::AudioFormat::Apac      => vr180_pipeline::fisheye_export::AudioTrack::Apac,
                #[cfg(not(target_os = "macos"))]
                crate::decoder::AudioFormat::Apac      => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
            },
            eye_w,
            eye_h,
            fps,
            bitrate_kbps,
            encoder: backend,
            bit_depth,
            prores_profile: opts.prores_profile as i32,
            inject_apmp: opts.inject_apmp,
            inject_youtube_vr180: opts.inject_youtube,
            spatial: (codec == ExportCodec::Spatial).then(|| vr180_pipeline::fisheye_export::SpatialVideo {
                baseline_mm: opts.spatial_baseline_mm,
                disparity: opts.spatial_disparity_pct / 100.0,
            }),
            stabilize: settings.stabilize,
            // The same cap and window the preview stabilizes with
            // (`Settings::stab_cap_and_window`); the export's own form of
            // "no cap" is 0.
            dji_max_corr_deg: Some(settings.stab_cap_and_window().0).filter(|cap| cap.is_finite()).unwrap_or(0.0),
            dji_smooth_ms: settings.stab_cap_and_window().1,
            dji_responsiveness: settings.dji_responsiveness,
            denoise_strength: settings.denoise_strength,
            view_adjust: settings.output_view_adjust(),
            fisheye_preset: settings.fisheye_preset.clone(),
            sbs_dewarp_fisheye: settings.sbs_dewarp_fisheye,
            fisheye_override_left: settings.fisheye_override_left,
            fisheye_override_right: settings.fisheye_override_right,
            fisheye_fov_deg_left: settings.fisheye_fov_deg_left,
            fisheye_fov_deg_right: settings.fisheye_fov_deg_right,
            fisheye_k_left: settings.fisheye_k_left,
            fisheye_k_right: settings.fisheye_k_right,
            fisheye_k5_left: settings.fisheye_k5_left,
            fisheye_k5_right: settings.fisheye_k5_right,
            fisheye_p_left: settings.fisheye_p_left,
            fisheye_p_right: settings.fisheye_p_right,
            fisheye_cx_norm_left: settings.fisheye_cx_norm_left,
            fisheye_cy_norm_left: settings.fisheye_cy_norm_left,
            fisheye_cx_norm_right: settings.fisheye_cx_norm_right,
            fisheye_cy_norm_right: settings.fisheye_cy_norm_right,
            // Effective swap: manual toggle XOR upside-down mount.
            fisheye_swap_eyes: settings.effective_swap_eyes(),
            insv_view: settings.insv_view,
            crop_mask: settings.crop_mask,
            braw_color: settings.braw_color.clone(),
            trim_in_s: settings.trim_in_s,
            trim_out_s: settings.trim_out_s,
            // A raw take's enhanced frames: the folder of its camera-original
            // DNGs beside it, when it is there and the option is on.
            raw_enhanced_dir,
            raw_enhanced_eye,
            color_stack,
            projection,
            lens_frame,
        }
    }

    /// The device a background export runs on.
    pub(super) fn export_device(&self) -> Arc<vr180_pipeline::gpu::Device> {
        // The export worker runs GPU work + RGB48 readback (`Maintain::Wait`)
        // on a background thread. Lesson #1: doing that on the device eframe
        // presents with can wedge the present queue. Export never hands a
        // texture to egui (it reads back to CPU for the encoder), so on
        // Windows we give it a DEDICATED logical device on the same adapter —
        // fully isolated from eframe's present queue. Falls back to the shared
        // device if the dedicated one can't be created.
        #[cfg(target_os = "windows")]
        let pipeline = match vr180_pipeline::gpu::Device::new_dedicated_from_adapter(
            &self.pipeline.adapter,
        ) {
            Ok(d) => {
                tracing::info!(
                    "export: dedicated wgpu device (isolated from eframe present queue)"
                );
                Arc::new(d)
            }
            Err(e) => {
                tracing::warn!("export: dedicated device unavailable ({e}); sharing eframe device");
                self.pipeline.clone()
            }
        };
        #[cfg(not(target_os = "windows"))]
        let pipeline = self.pipeline.clone();
        pipeline
    }

    /// Spawn the export worker thread for a prepared config and install
    /// it as the (single) running `export_job`.
    /// `eac_stab`: for a GoPro `.360` (EAC) export, the `(Settings,
    /// n_frames)` needed to compute the per-eye stabilization (the
    /// preview's `build_per_eye_frames`); `None` for fisheye sources,
    /// which compute stab inside `export_fisheye`.
    pub(super) fn spawn_export_job(
        &mut self,
        cfg: vr180_pipeline::fisheye_export::FisheyeExportConfig,
        eac_stab: Option<(crate::decoder::Settings, usize)>,
    ) {
        let output_path = cfg.output_path.clone();
        let still_format = self.batch_opts.unwrap_or(self.export_opts).still_format;
        // (A run that writes the takes themselves has no view to export.)
        let writes = run_output(self.take_run, cfg.source_kind, &self.batch_opts.unwrap_or(self.export_opts));
        let take_job = match writes { RunOutput::Take(job) => Some(job), _ => None };
        let raw_frames = writes == RunOutput::DngFrames;

        // Stop preview playback when it reads the SAME file as the export
        // (they'd race on the decoder / file). Previewing a DIFFERENT clip
        // is fine — that's what lets you keep tuning clips mid-batch.
        // A STILL is the exception: its preview thread read the DNG once
        // and now holds only textures, and the export re-reads the file
        // itself, so there is nothing to race on. Stopping it left the
        // photo frozen on the last render with no transport to restart it
        // (the still has no Play button) — the "UI stopped responding
        // after exporting" report.
        if self.loaded_path.as_ref() == Some(&cfg.source_path) && !cfg.source_kind.is_still() {
            self.stop_playback();
        }

        let (progress_tx, progress_rx) = crossbeam_channel::bounded(256);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        let pipeline = self.export_device();
        let progress_tx2 = progress_tx.clone();
        let handle = std::thread::spawn(move || {
            if let Some((settings, n_frames)) = eac_stab {
                // GoPro EAC: compute the same per-eye stab the preview uses,
                // then run the EAC export. build_per_eye_frames_multi is
                // empty when stabilization + RS are both off → identity.
                // Multi-segment: span ALL segments (gyro aggregated as one).
                let total_n = if cfg.segments.len() > 1 {
                    crate::decoder::segments_total_frames(&cfg.segments)
                } else { n_frames };
                let per_eye = crate::decoder::build_per_eye_frames_multi(
                    &cfg.segments, &settings, cfg.fps, total_n,
                ).unwrap_or_default();
                vr180_pipeline::fisheye_export::export_eac(
                    pipeline, cfg, per_eye,
                    move |p| { let _ = progress_tx.try_send(p); },
                    cancel_for_thread,
                )
            } else if cfg.source_kind.is_still() {
                // Still photo: one frame through the raw stage + projection,
                // written as PNG / JPEG / linear DNG. Progress is reported as a
                // 0..100 "frame" count so the export bar needs no special case.
                let still_cfg = vr180_pipeline::still_export::StillExportConfig {
                    source_path: cfg.source_path.clone(),
                    output_path: cfg.output_path.clone(),
                    format: still_format.to_pipeline(),
                    eye_w: cfg.eye_w, eye_h: cfg.eye_h,
                    view_adjust: cfg.view_adjust,
                    swap_eyes: cfg.fisheye_swap_eyes,
                    develop: vr180_pipeline::raw_develop::DevelopParams::default(),
                    color_stack: cfg.color_stack.clone(),
                    projection: cfg.projection,
                    lens_override: vr180_pipeline::still_export::StillLensOverride::from_export_cfg(&cfg),
                };
                let t0 = std::time::Instant::now();
                vr180_pipeline::still_export::export_still(pipeline, &still_cfg, &|_stage, f| {
                    let done = (f * 100.0).round() as u64;
                    let _ = progress_tx2.try_send(vr180_pipeline::fisheye_export::ExportProgress {
                        frame_idx: done.min(100), total_frames: 100,
                        fps_avg: done as f32 / t0.elapsed().as_secs_f32().max(1e-3),
                    });
                })
            } else if let Some(TakeJob::Archive(size)) = take_job {
                // An archive copy of the take.
                vr180_pipeline::raw_archive::export_archive(
                    &cfg.source_path, &cfg.output_path, cfg.trim_in_s, cfg.trim_out_s, size,
                    move |p| { let _ = progress_tx2.try_send(p); },
                    cancel_for_thread,
                ).map(|_| ())
            } else if take_job == Some(TakeJob::Originals) {
                // The take's frames as the sensor recorded them.
                vr180_pipeline::sensor_dng::export_sensor_dng_sequence(
                    &cfg.source_path, &cfg.output_path, cfg.trim_in_s, cfg.trim_out_s,
                    move |p| { let _ = progress_tx2.try_send(p); },
                    cancel_for_thread,
                ).map(|_| ())
            } else if raw_frames {
                // A raw video clip as a folder of linear DNG frames.
                vr180_pipeline::raw_export::export_dng_sequence(
                    pipeline, cfg,
                    move |p| { let _ = progress_tx2.try_send(p); },
                    cancel_for_thread,
                ).map(|_| ())
            } else {
                vr180_pipeline::fisheye_export::export_fisheye(
                    pipeline, cfg,
                    move |p| { let _ = progress_tx2.try_send(p); },
                    cancel_for_thread,
                )
            }
        });
        tracing::info!("export: spawned worker → {}", output_path.display());

        self.export_job = Some(ExportJob {
            output_path,
            handle,
            progress_rx,
            cancel,
            last_progress: None,
            started_at: std::time::Instant::now(),
        });
    }

    pub(super) fn poll_export_job(&mut self) {
        let Some(job) = self.export_job.as_mut() else { return; };
        // Drain the channel.
        while let Ok(p) = job.progress_rx.try_recv() {
            job.last_progress = Some(p);
        }
        let has_frames = job.last_progress.as_ref().map_or(false, |p| p.frame_idx > 0);
        // The run's first written frame anchors the ETA rate: everything
        // before it (IMU load, decoder/encoder init, ProRes Vulkan warm-up)
        // is start-up, not encode throughput.
        if has_frames && self.run_rate_anchor.is_none() {
            let done = self.run_frame_tally().map(|t| t.0).unwrap_or(0);
            self.run_rate_anchor = Some((std::time::Instant::now(), done));
        }
        let Some(job) = self.export_job.as_mut() else { return; };
        // Check completion. JoinHandle::is_finished is stable in 1.61+.
        if job.handle.is_finished() {
            let job = self.export_job.take().unwrap();
            let result = job.handle.join();
            // (A joined run's look for enhanced frames writes nothing.)
            let looked = self.batch_current.is_none() && self.join_run.as_ref().is_some_and(|j| j.enhanced.is_none());
            match &result {
                Ok(Ok(())) if looked => tracing::info!("join: the takes' folders looked through in {:.2?}", job.started_at.elapsed()),
                Ok(Ok(())) => tracing::info!(
                    "export: done → {} (took {:.2?})",
                    job.output_path.display(), job.started_at.elapsed()
                ),
                Ok(Err(e)) => tracing::error!("export: failed: {e}"),
                Err(_) => tracing::error!("export: worker panicked"),
            }
            // Batch bookkeeping: record this item's outcome, then chain
            // the next queued item. A failed item never stops the batch
            // (that's the unattended-run contract) — it records the error
            // and moves on.
            if let Some(idx) = self.batch_current.take() {
                if let Some(item) = self.batch.get_mut(idx) {
                    item.status = if self.batch_skip_flag {
                        BatchStatus::Skipped
                    } else {
                        match result {
                            Ok(Ok(())) => BatchStatus::Done,
                            Ok(Err(e)) => BatchStatus::Failed(e.to_string()),
                            Err(_) => BatchStatus::Failed("worker panicked".into()),
                        }
                    };
                }
                self.batch_skip_flag = false;
                // An archive copy that came out whole is known to its take's entry.
                if matches!(self.take_run, Some(TakeRun { job: TakeJob::Archive(_), .. }))
                    && self.batch.get(idx).is_some_and(|b| b.status == BatchStatus::Done)
                {
                    self.note_copy(idx, &job.output_path);
                }
                // One file or none: a joined run whose clip did not come
                // out whole ends here, and the clips still waiting are not
                // exported for a file that will not be made.
                if self.join_run.is_some() && !self.batch.get(idx).is_some_and(|b| b.status == BatchStatus::Done) {
                    for b in &mut self.batch {
                        if b.status == BatchStatus::Queued {
                            b.status = BatchStatus::Idle;
                        }
                    }
                }
                if self.batch_running {
                    self.start_next_batch_item();
                } else {
                    // Stopped by the user.
                    self.end_join(Err(tr("stopped").to_string()));
                    self.end_take_run();
                }
            } else if self.join_run.as_ref().is_some_and(|j| j.enhanced.is_none()) {
                // The run's first job: its takes' folders looked through
                // for enhanced frames.
                self.end_join_check();
            } else if self.join_run.as_ref().is_some_and(|j| j.joining) {
                // The run's last job: its parts put together.
                self.end_join(match result {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(_) => Err("worker panicked".into()),
                });
                self.finish_run();
            }
        }
    }

    /// Output path for a batch item: `<stem>_SBS.<ext>` in the override
    /// folder (or next to the source), auto-renamed ` (2)`, ` (3)`… on
    /// collision rather than overwriting or silently skipping.
    pub(super) fn batch_output_for(&self, item: &BatchItem, opts: &ExportOptions) -> PathBuf {
        let mut out = match run_output(self.take_run, item.source_kind, opts) {
            RunOutput::Still => Self::default_still_output_for(&item.path, opts.still_format),
            RunOutput::Take(job) => Self::take_job_dir_for(&item.path, job),
            RunOutput::DngFrames => Self::default_frames_dir_for(&item.path, false),
            RunOutput::Video => Self::default_output_for(&item.path, opts.codec_for(&item.export_settings(), false), item.single_eye(),
                item.settings.lens_frame_view(item.source_kind)),
        };
        if let Some(dir) = &self.batch_out_dir {
            if let Some(name) = out.file_name() {
                out = dir.join(name);
            }
        }
        unused_path(out)
    }

    /// The export of list entry `idx` to `output` under `opts`, and what a
    /// `.360` export needs for its stabilization.
    pub(super) fn export_cfg_for(
        &self,
        idx: usize,
        output: &std::path::Path,
        opts: &ExportOptions,
        segments: Vec<PathBuf>,
    ) -> (vr180_pipeline::fisheye_export::FisheyeExportConfig, Option<(crate::decoder::Settings, usize)>) {
        let item = &self.batch[idx];
        let eac_stab = item.source_kind.is_eac().then(|| {
            // Total across the merged recording (== single-file count when
            // `segments` is one file) so EAC stab spans the whole timeline.
            (item.settings.clone(), crate::decoder::segments_total_frames(&segments).max(1))
        });
        (
            Self::build_export_cfg(
                &item.path, output,
                item.fps, item.source_kind,
                item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height,
                item.sharp_eye(),
                &item.export_settings(), opts,
                segments,
            ),
            eac_stab,
        )
    }

    /// Start a run of `run` (list entries, in order): as ONE file when that
    /// is chosen and there are several clips, else each to its own.
    pub(super) fn start_run(&mut self, run: Vec<usize>) {
        self.end_take_run();
        // Each clip is read from its take, or from the take's copy while the take is not at hand.
        for &i in &run {
            self.settle_source(i);
        }
        self.join_notice = None;
        self.joined_output = None;
        self.joined_note = None;
        if self.join_clips && run.len() >= 2 {
            match self.plan_join(&run, &self.export_opts) {
                Ok(plan) => {
                    tracing::info!("join: {} clips → {} ({:?} sound{})", run.len(), plan.output.display(), plan.sound,
                        if plan.immersive { ", Apple Immersive Video: a lens for each clip" } else { "" });
                    self.join_run = Some(plan);
                }
                Err(why) => {
                    tracing::warn!("join: not started: {why}");
                    self.join_notice = Some(why);
                    return;
                }
            }
        }
        for &i in &run {
            self.batch[i].status = BatchStatus::Queued;
        }
        self.begin_run(run);
    }

    /// Hold the system's exemption while a run is in flight (`at_work`) and
    /// give it back when none is.
    pub(super) fn tell_the_system(&mut self, at_work: bool) {
        if at_work && self.busy.is_none() {
            self.busy = crate::activity::Busy::begin("Exporting");
            if self.busy.is_some() {
                tracing::info!("export run: the system is told the app is at work (full speed behind other windows, no idle sleep)");
            }
        } else if !at_work && self.busy.take().is_some() {
            tracing::info!("export run: over, the system may rest the app again");
        }
    }

    /// The run is over: its summary (banner + notification), and the run's
    /// state back to idle.
    pub(super) fn finish_run(&mut self) {
        let done = self.batch_run_indices.iter()
            .filter(|&&i| self.batch.get(i).map_or(false, |b| b.status == BatchStatus::Done))
            .count();
        let failed = self.batch_run_indices.iter()
            .filter(|&&i| self.batch.get(i).map_or(false,
                |b| matches!(b.status, BatchStatus::Failed(_))))
            .count();
        let elapsed = self.batch_started_at.map(|t| t.elapsed())
            .unwrap_or_default();
        tracing::info!("export run: complete — {done} done, {failed} failed in {:.1?}", elapsed);
        let n_run = self.batch_run_indices.len();
        self.batch_running = false;
        self.batch_opts = None;
        self.batch_started_at = None;
        self.batch_run_indices.clear();
        let take = self.take_run.map(|run| run.job);
        if done + failed > 0 {
            self.notify_run_complete(done, failed, elapsed, n_run, take);
            self.last_batch_summary = Some(BatchSummary { done, failed, elapsed, joined: self.joined_output.take(), note: self.joined_note.take(), take });
        }
        self.end_take_run();
    }

    /// Export every not-yet-done clip in the list (Idle / Failed / Skipped).
    /// Done clips stay done — re-arm one with its Re-run button. As ONE
    /// file (`join_clips`) the run is the whole list, whatever was exported
    /// before: the file is the list.
    pub(super) fn start_batch(&mut self) {
        if self.batch_running || self.export_job.is_some() { return; }
        self.sync_active_clip_settings();
        let run: Vec<usize> = self.batch.iter().enumerate()
            .filter(|(_, item)| item.ready() && (self.joins_all() || matches!(item.status,
                BatchStatus::Idle | BatchStatus::Failed(_) | BatchStatus::Skipped)))
            .map(|(i, _)| i)
            .collect();
        self.start_run(run);
    }

    pub(super) fn start_next_batch_item(&mut self) {
        // A joined run that has yet to learn whether its parts read
        // enhanced frames looks for them before its first clip.
        if self.join_run.as_ref().is_some_and(|j| j.enhanced.is_none()) {
            self.start_join_check();
            return;
        }
        let Some(idx) = self.batch.iter().position(|b| b.status == BatchStatus::Queued) else {
            // Queue drained. A run that writes one file puts its parts
            // together now; any other is complete.
            if !self.start_join() {
                self.finish_run();
            }
            return;
        };
        let opts = self.batch_opts.unwrap_or(self.export_opts);
        // A clip of a joined run is written to its part.
        let part = self.join_run.as_ref().and_then(|j| {
            let k = j.parts.iter().position(|(clip, _)| *clip == self.batch[idx].path)?;
            Some((k, j.parts[k].1.clone(), j.parts_dir.clone(), j.sound, j.enhanced.unwrap_or(false)))
        });
        let output = match &part {
            Some((_, file, _, _, _)) => file.clone(),
            None => self.batch_output_for(&self.batch[idx], &opts),
        };
        let item_path = self.batch[idx].path.clone();
        let segments = self.clip_segments(&item_path);
        let (mut cfg, eac_stab) = self.export_cfg_for(idx, &output, &opts, segments);
        if self.take_run.is_some_and(|run| run.whole) {
            cfg.trim_in_s = None;
            cfg.trim_out_s = None;
        }
        if let Some((k, _, dir, sound, enhanced)) = &part {
            // Enhanced frames: every part of the file reads them, or none.
            if !enhanced {
                cfg = cfg.without_enhanced_frames();
            }
            // Every part carries what the joined file's one track is made
            // of: stereo, or the clip's four-channel sound field.
            cfg.audio_track = match sound {
                vr180_pipeline::join::JoinSound::Stereo => vr180_pipeline::fisheye_export::AudioTrack::Stereo,
                _ => vr180_pipeline::fisheye_export::AudioTrack::Ambisonic,
            };
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!("join: {}: {e}", dir.display());
            }
            if *k == 0 {
                if let Some(join) = self.join_run.as_mut() {
                    join.first_cfg = Some(Box::new(cfg.clone()));
                }
            }
        }
        self.batch[idx].status = BatchStatus::Running;
        self.batch_current = Some(idx);
        tracing::info!("batch: item {}/{} → {}", idx + 1, self.batch.len(), output.display());
        self.spawn_export_job(cfg, eac_stab);
    }

    /// Cancel the running item but keep the batch going (poll marks it
    /// Skipped and chains the next).
    pub(super) fn skip_current_batch_item(&mut self) {
        if let (Some(_), Some(job)) = (self.batch_current, self.export_job.as_ref()) {
            self.batch_skip_flag = true;
            job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Stop the whole batch: cancel the running item and don't chain.
    pub(super) fn stop_batch(&mut self) {
        self.batch_running = false;
        self.batch_opts = None;
        // A joined run's own jobs: its first (the look for enhanced frames)
        // and its last (the parts put together).
        let joining = self.join_run.as_ref().is_some_and(|j| j.joining || j.enhanced.is_none());
        if let Some(job) = self.export_job.as_ref() {
            if self.batch_current.is_some() {
                self.batch_skip_flag = true; // mark the cut-short item Skipped
                job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            } else if joining {
                job.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    /// Draw the floating export-options window when `export_opts_visible`
    /// is set. The window is dismissable; "Start Export…" hands off to
    /// `commit_export()` which pops the save-file dialog and spawns the
    /// encode worker.
    /// Persistent bottom export bar — the single home for output settings +
    /// running the queue (replaces the old Batch window AND the Export-options
    /// window). Idle: Output / Format dropdowns + Export selected / Export all.
    /// Running: overall progress bar + ETA + Skip / Stop. A finished run leaves
    /// a dismissable summary chip.
    pub(super) fn draw_export_bar(&mut self, ctx: &egui::Context) {
        // Snapshot everything the panel needs up front so the closure only
        // mutably borrows `self.export_opts` (egui disjoint-field capture).
        let running = self.batch_running;
        let run_progress = self.overall_run_progress();
        let run_opts = self.batch_opts.unwrap_or(self.export_opts);
        // (A run that writes the takes themselves filters nothing.)
        let cur_nr = self.take_run.is_none() && self.batch_current
            .and_then(|i| self.batch.get(i))
            .map_or(false, |b| export_runs_noise_filter(b.source_kind, b.settings.denoise_strength, &run_opts));
        let n_all = self.batch.iter().filter(|b| b.ready() && matches!(b.status,
            BatchStatus::Idle | BatchStatus::Queued
            | BatchStatus::Failed(_) | BatchStatus::Skipped)).count();
        let n_sel = self.batch.iter().filter(|b| b.selected && b.source_kind.is_exportable()).count();
        // As one file: "all" is the whole list, and a selection joins when
        // it holds several clips.
        let n_ready = self.batch.iter().filter(|b| b.ready()).count();
        let join_all = self.joins_all();
        let join_sel = self.join_clips && n_sel >= 2;
        let join_run = self.join_run.is_some();
        // The run's last job: its parts being put together.
        let join_progress = self.join_run.as_ref().filter(|j| j.joining)
            .map(|_| self.export_job.as_ref().and_then(|j| j.last_progress));
        // Its first: the takes' folders looked through for enhanced frames.
        let join_checking = self.join_run.as_ref().is_some_and(|j| j.enhanced.is_none());
        // Enhanced frames left out of a joined run, and why.
        let join_note = self.join_run.as_ref().and_then(|j| j.note.clone());
        let can_export = self.export_job.is_none() && !running;
        let has_active = self.active_clip.is_some();
        let out_label = match &self.batch_out_dir {
            Some(d) => d.file_name().map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| d.to_string_lossy().to_string()),
            None => tr("next to source").to_string(),
        };
        let has_out_dir = self.batch_out_dir.is_some();
        let clip_fps = self.clip.as_ref().map(|c| c.fps).unwrap_or(30.0);
        let is_still = self.clip.as_ref().map(|c| c.source_kind.is_still()).unwrap_or(false);
        // A single-eye clip's frame is its one eye, not two side by side.
        let single_eye = self.clip_single_eye();
        // The loaded clip's own choice (a Crop clip has one apart, and so
        // has a raw take whose enhanced frames were found).
        let sharp_eye = self.clip.as_ref().and_then(|c| c.sharp_eye(self.settings.insv_view));
        let lens_frame = self.clip.as_ref()
            .is_some_and(|c| self.settings.lens_frame_view(c.source_kind));
        let enhanced = self.enhanced_frames(ctx);
        let enhanced_sizes = matches!(enhanced, Some(EnhancedFrames::Found(_))) && self.export_opts.raw_use_enhanced;
        let size_list = SizeList::of(sharp_eye.is_some(), enhanced_sizes);
        let resolution = self.export_opts.resolution_for(size_list);
        let res_label = match (self.settings.reframe_aspect_if_active(), resolution, sharp_eye) {
            // Lens Space through the camera's lens: the eye as recorded (or
            // its 8K copy, or a size of its enhanced frames), named as such.
            _ if lens_frame => {
                // The clip's own eye: its native frame is that eye, or two side by side.
                let (ew, eh) = self.native_sbs_dims()
                    .map(|(w, h)| (if single_eye { w } else { w / 2 }, h)).unwrap_or((0, 0));
                // A stabilized frame carries the stabilizer's margin; a
                // reduced copy and a size of the enhanced frames keep theirs.
                let (w, h) = self.export_opts.lens_frame_out_dims(ew, eh, size_list, self.settings.stabilize, single_eye);
                format!("{} {} × {h}", tr("Lens Space"), if single_eye { w } else { w * 2 })
            }
            (Some(aspect), _, _) => self.export_opts.reframe_size.frame_label(aspect, single_eye),
            // Native = the clip's own size (a 12K BRAW: 12288 × 6144).
            (None, ExportResolution::Native, _) => match self.native_sbs_dims() {
                Some((w, h)) => format!("{w} × {h}"),
                None => resolution.frame_label(single_eye).to_string(),
            },
            // A Crop clip's native-sharpness canvas.
            (None, ExportResolution::Sharp, Some(side)) =>
                format!("{} × {side}", if single_eye { side } else { side * 2 }),
            // The canvases of a raw take's enhanced frames.
            (None, r @ (ExportResolution::R13k | ExportResolution::R16k), _) => r.frame_label(single_eye).to_string(),
            (None, _, _) => ExportResolution::R8k.frame_label(single_eye).to_string(),
        };
        let h265_mbps = self.export_opts.effective_h265_mbps(
            self.settings.reframe_aspect_if_active(), clip_fps, single_eye);
        // What this clip would really export as (spatial video only where
        // it can be one).
        let bar_codec = self.export_opts.codec_for(&self.worker_settings(), is_still);
        // A spatial video is one stream of the eye's own size, not a
        // side-by-side frame: label it by the eye.
        let res_label = match (bar_codec, self.settings.reframe_aspect_if_active()) {
            (ExportCodec::Spatial, Some(aspect)) => {
                let (w, h) = self.export_opts.reframe_size.eye_dims(aspect);
                format!("{w} × {h} {}", tr("per eye"))
            }
            _ => res_label,
        };
        let raw_frames = self.export_opts.raw_dng_sequence
            && self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video());
        let fmt_label = if is_still {
            // A still has no codec / bitrate: container + size only.
            format!("{} · {}", self.export_opts.still_format.label(), res_label)
        } else if raw_frames {
            format!("{} · {}", tr("DNG sequence"), res_label)
        } else { format!("{} · {}{}{}",
            bar_codec.label(),
            res_label,
            match bar_codec {
                ExportCodec::H265 => format!(" · {} Mbps · {}-bit",
                    h265_mbps, self.export_opts.h265_bit_depth),
                ExportCodec::ProRes => format!(" · {}", self.export_opts.prores_profile.label()),
                ExportCodec::Spatial => format!(" · {} Mbps · {:.0}°",
                    h265_mbps, self.settings.reframe_hfov_deg),
            },
            if self.export_opts.beyondvr_hack && !single_eye && self.settings.reframe_aspect_if_active().is_none() {
                format!(" · BeyondVR {:.0}%", self.export_opts.beyondvr_scale_pct)
            } else { String::new() }) };
        let summary = self.last_batch_summary.as_ref().map(|s| {
            let col = if s.failed > 0 { Color32::from_rgb(230, 160, 90) }
                      else { Color32::from_rgb(120, 200, 120) };
            let took = format_eta(s.elapsed.as_secs_f64());
            let txt = if s.failed > 0 {
                format!("{DONE_MARK} {} done · {} failed · {took}", s.done, s.failed)
            } else if s.joined.is_some() {
                if cur_lang() == Lang::Zh { format!("{DONE_MARK} 已合成 {} 个片段 · {took}", s.done) } else { format!("{DONE_MARK} {} clips joined · {took}", s.done) }
            } else {
                format!("{DONE_MARK} {} · {took}", run_done_text(s.done, s.take, cur_lang() == Lang::Zh))
            };
            // The joined file's name goes with the hint.
            let hover = match (s.joined.as_deref().map(clip_label), &s.note) {
                (Some(name), Some(note)) => format!("{name}\n{note}\n{}", tr("Click to dismiss")),
                (Some(name), None) => format!("{name}\n{}", tr("Click to dismiss")),
                (None, _) => tr("Click to dismiss").to_string(),
            };
            (txt, col, hover)
        });

        let mut do_export_selected = false;
        let mut do_export_all = false;
        let mut do_skip = false;
        let mut do_stop = false;
        let mut do_choose_dir = false;
        let mut do_clear_dir = false;
        let mut dismiss_summary = false;
        let mut dismiss_gpu_error = false;
        let mut toggle_settings = false;

        egui::TopBottomPanel::bottom("export_bar").exact_height(40.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                if running {
                    if let Some(progress) = join_progress {
                        // The clips are written; the parts are copied into
                        // the one file.
                        let frac = progress.filter(|p| p.total_frames > 0)
                            .map_or(0.0, |p| (p.frame_idx as f32 / p.total_frames as f32).clamp(0.0, 1.0));
                        ui.add(egui::ProgressBar::new(frac).desired_width(240.0)
                            .text(format!("{:.0}%", frac * 100.0)));
                        ui.label(RichText::new(tr("joining the clips…")).color(Color32::from_rgb(120, 180, 255)));
                    } else if join_checking {
                        ui.spinner();
                        ui.label(RichText::new(tr("Looking for enhanced frames…")).color(Color32::from_rgb(120, 180, 255)));
                    } else if let Some((done, total, eta, pos, n)) = run_progress {
                        let frac = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        ui.add(egui::ProgressBar::new(frac).desired_width(240.0)
                            .text(format!("{:.0}%", frac * 100.0)));
                        let eta_txt = eta.map(|e| if cur_lang() == Lang::Zh {
                            format!(" · 约剩 {}", format_eta(e))
                        } else {
                            format!(" · ~{} left", format_eta(e))
                        }).unwrap_or_default();
                        let nr_txt = if cur_nr {
                            if cur_lang() == Lang::Zh { " · 降噪" } else { " · NR" }
                        } else { "" };
                        let item_txt = if cur_lang() == Lang::Zh {
                            format!("第 {pos}/{n} 项{eta_txt}{nr_txt}")
                        } else {
                            format!("item {pos}/{n}{eta_txt}{nr_txt}")
                        };
                        ui.label(RichText::new(item_txt).color(Color32::from_rgb(120, 180, 255)));
                    } else {
                        ui.label(RichText::new(tr("starting…")).color(Color32::GRAY));
                    }
                    // Slow-path note from the pipeline (portable CPU loop, or
                    // a CPU encoder fallback) — users shouldn't need the log
                    // to know why an export is slow.
                    if let Some(note) = vr180_pipeline::fisheye_export::export_path_note() {
                        ui.label(RichText::new(format!("⚠ {note}"))
                            .small().color(Color32::from_rgb(230, 160, 90)));
                    }
                    // A joined run that leaves enhanced frames out says so
                    // (the bar has room for a few words: why is on hover).
                    if let Some(note) = &join_note {
                        ui.label(RichText::new(format!("⚠ {}", tr("enhanced frames left out")))
                            .small().color(Color32::from_rgb(230, 160, 90)))
                            .on_hover_text(note);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(tr("Stop")).clicked() { do_stop = true; }
                        // (One file or none: a joined run has no clip to skip.)
                        if !join_run && ui.button(tr("Skip current")).clicked() { do_skip = true; }
                    });
                } else {
                    ui.menu_button(format!("📁  {out_label}"), |ui| {
                        if ui.button(tr("Choose folder…")).clicked() { do_choose_dir = true; ui.close_menu(); }
                        ui.add_enabled_ui(has_out_dir, |ui| {
                            if ui.button(tr("Next to each source")).clicked() { do_clear_dir = true; ui.close_menu(); }
                        });
                    }).response.on_hover_text(tr("Output folder for exports"));
                    // A toggle (not a menu) — a settings menu closes the moment
                    // you touch a ComboBox/slider inside it. Opens a small
                    // non-modal window instead (drawn after this panel).
                    if ui.selectable_label(self.show_export_settings, format!("⚙  {fmt_label}"))
                        .on_hover_text(tr("Output format, resolution & metadata")).clicked()
                    {
                        toggle_settings = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let all_label = if join_all {
                            format!("{} ({n_ready})", tr("Join all"))
                        } else {
                            format!("{} ({n_all})", tr("Export all"))
                        };
                        ui.add_enabled_ui(can_export && (join_all || n_all > 0), |ui| {
                            if ui.button(RichText::new(all_label).strong()).clicked() {
                                do_export_all = true;
                            }
                        });
                        let sel_label = if join_sel {
                            format!("{} ({n_sel})", tr("Join selected"))
                        } else if n_sel > 0 {
                            format!("{} ({n_sel})", tr("Export selected"))
                        } else {
                            tr("Export selected").to_string()
                        };
                        ui.add_enabled_ui(can_export && (n_sel > 0 || has_active), |ui| {
                            if ui.button(sel_label).on_hover_text(tr(if join_sel {
                                "Export the ticked clips as one file"
                            } else if n_sel > 0 {
                                "Export the ticked clips"
                            } else {
                                "Export the active clip"
                            })).clicked() { do_export_selected = true; }
                        });
                        // Several clips can be written as one file.
                        if n_ready >= 2 {
                            ui.add_enabled_ui(can_export, |ui| {
                                ui.checkbox(&mut self.join_clips, tr("Join into one file"))
                                    .on_hover_text(tr(JOIN_HOVER));
                            });
                        }
                        if let Some((txt, col, hover)) = &summary {
                            if ui.add(egui::Label::new(RichText::new(txt).color(*col))
                                .sense(egui::Sense::click()))
                                .on_hover_text(hover).clicked() {
                                dismiss_summary = true;
                            }
                        }
                        // An uncaptured GPU error used to be a panic; now it
                        // is recorded and shown here, so a GPU fault is
                        // visible instead of a silent wedge. Click to dismiss.
                        if let Some(err) = vr180_pipeline::gpu::last_gpu_error() {
                            if ui.add(egui::Label::new(RichText::new(
                                    format!("⚠ {}", tr("GPU error — see the log")))
                                    .small().color(Color32::from_rgb(230, 120, 120)))
                                .sense(egui::Sense::click()))
                                .on_hover_text(format!("{err}

{}", tr("Click to dismiss")))
                                .clicked() {
                                dismiss_gpu_error = true;
                            }
                        }
                    });
                }
            });
        });

        if do_choose_dir {
            self.spawn_dialog(ctx, || {
                rfd::FileDialog::new().pick_folder().map(DialogResult::BatchOutDir)
            });
        }
        if do_clear_dir { self.batch_out_dir = None; }
        if dismiss_summary { self.last_batch_summary = None; }
        if dismiss_gpu_error { vr180_pipeline::gpu::clear_gpu_error(); }
        if toggle_settings { self.show_export_settings = !self.show_export_settings; }
        if do_export_selected { self.export_selected(); }
        if do_export_all { self.start_batch(); }
        if do_skip { self.skip_current_batch_item(); }
        if do_stop { self.stop_batch(); }
        self.draw_join_notice(ctx);

        // Format settings popover — a real (non-modal) window so ComboBoxes
        // and sliders stay open while you edit them.
        if self.show_export_settings {
            let mut open = true;
            let clip_fps = self.clip.as_ref().map(|c| c.fps).unwrap_or(30.0);
            let is_still = self.clip.as_ref().map(|c| c.source_kind.is_still()).unwrap_or(false);
            // A raw video clip, and what is known of its enhanced frames.
            let raw_video = enhanced;
            let native_dims = self.native_sbs_dims();
            let single_eye = self.clip_single_eye();
            // A Crop clip's native-sharpness canvas (side by side, or the one
            // eye of a single-eye clip), if it has one.
            let sharp_dims = self.clip.as_ref()
                .and_then(|c| c.sharp_eye(self.settings.insv_view))
                .map(|side| (if single_eye { side } else { side * 2 }, side));
            // The Lens Space view: a lens frame, or the equidistant disc.
            let lens_space = self.lens_space_out();
            egui::Window::new(tr("Export format"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -150.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(360.0);
                    Self::export_options_ui(ui, &mut self.export_opts, "export",
                        self.settings.reframe_aspect_if_active(),
                        (self.settings.reframe_hfov_deg, self.settings.reframe_defish),
                        clip_fps, is_still, native_dims, sharp_dims, lens_space,
                        self.settings.stabilize, single_eye, raw_video);
                });
            self.show_export_settings = open;
        }
    }
}

/// Human ETA: `1h 5m` / `18m 3s` / `45s`. For the export bar's time-left.
pub(super) fn format_eta(secs: f64) -> String {
    if !secs.is_finite() {
        return "—".to_string();
    }
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{}s", s)
    }
}
