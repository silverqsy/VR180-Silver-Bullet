//! Several clips written as one file: the plan, the look through their folders, and the join after the last clip.

use super::*;

/// The sound of a joined file: the first clip's Audio choice (`wanted`).
/// Ambisonic and APAC are for a run whose clips all have four-channel sound
/// (`four_channel`, one entry a clip, in the run's order): `Err` is the
/// place in the run of the first clip that has not.
pub(super) fn joined_sound(wanted: crate::decoder::AudioFormat, four_channel: &[bool]) -> Result<vr180_pipeline::join::JoinSound, usize> {
    use crate::decoder::AudioFormat as AF;
    use vr180_pipeline::join::JoinSound;
    // A first clip without such sound has no say: its choice is not even
    // shown, and may be one left over from another clip.
    if wanted == AF::Stereo || !four_channel.first().copied().unwrap_or(false) {
        return Ok(JoinSound::Stereo);
    }
    if let Some(odd) = four_channel.iter().position(|has| !has) {
        return Err(odd);
    }
    Ok(if wanted == AF::Apac { JoinSound::Apac } else { JoinSound::Ambisonic })
}

/// A run of the export queue whose clips are written as ONE file. Each
/// clip is exported the usual way, to a part of its own, and the parts are
/// put together when the last one is written (`vr180_pipeline::join`):
/// nothing is encoded twice, a cut falls on a keyframe, and the sound is
/// encoded once for the whole length. One file or none: a clip that does
/// not come out whole ends the run, and what was exported by then stays in
/// the parts' folder.
pub(super) struct JoinRun {
    /// The joined file.
    pub(super) output: PathBuf,
    /// The folder the parts are written in: made beside the joined file,
    /// and taken away with the parts once that file is whole.
    pub(super) parts_dir: PathBuf,
    /// The clips in the order they are joined, each with its part's file.
    pub(super) parts: Vec<(PathBuf, PathBuf)>,
    /// The first clip's export, as it ran: the joined file is described to
    /// a player as that export described its own.
    pub(super) first_cfg: Option<Box<vr180_pipeline::fisheye_export::FisheyeExportConfig>>,
    /// The file's sound ([`joined_sound`]): every part is exported with the
    /// track it is made of.
    pub(super) sound: vr180_pipeline::join::JoinSound,
    /// The parts are lens frames, each with its own lens: the joined file
    /// is an Apple Immersive Video file, which holds a lens for every part
    /// (`vr180_pipeline::join::join_immersive`).
    pub(super) immersive: bool,
    /// The parts are written; they are being put together.
    pub(super) joining: bool,
    /// The parts read their takes' enhanced frames: all of them, or none
    /// (the user's rule for a joined file, 2026-11-26: only when every clip
    /// to be joined uses them, else they are left out). `None` while the
    /// takes' folders are looked through, as the run's first job.
    pub(super) enhanced: Option<bool>,
    /// That job ([`JoinCheck`]), for a run with a clip that could read
    /// enhanced frames.
    pub(super) check: Option<JoinCheck>,
    /// Enhanced frames were left out although some clips have them
    /// ([`join_enhanced_note`]): said in the export bar while the run is in
    /// flight, and with its summary.
    pub(super) note: Option<String>,
}

/// The first job of a joined run with raw takes that could read enhanced
/// frames: each such take's folder is looked through for its clip's
/// stretch, on a thread (thousands of files, often on a share). The parts
/// of one file read enhanced frames all or none
/// (`vr180_pipeline::join::parts_read_enhanced`), so the run has to know
/// before its first clip is written.
pub(super) struct JoinCheck {
    /// The parts' exports as planned, in the run's order, for those that
    /// could read enhanced frames (`None`: no raw take, no folder beside
    /// it, or the option off). Taken when the job starts.
    pub(super) exports: Vec<Option<vr180_pipeline::fisheye_export::FisheyeExportConfig>>,
    /// Whether the parts come out as one shape read from the capture, and
    /// read from enhanced frames (`Err`: why not, for the user). A size
    /// picked for enhanced frames is not the size picked without them.
    pub(super) agree_plain: Result<(), String>,
    pub(super) agree_enhanced: Result<(), String>,
    /// The lookup's answer when the job is done: a part a place, `None`
    /// for a part that was not asked.
    pub(super) found: Option<crossbeam_channel::Receiver<Vec<Option<bool>>>>,
}

/// The look itself ([`JoinCheck`]): for each part that could read enhanced
/// frames (`Some`), whether its take's folder holds some for the part's
/// stretch. `None`: stopped (`stop`) before it was through. Lists folders:
/// for a thread of its own.
pub(super) fn join_look(
    exports: &[Option<vr180_pipeline::fisheye_export::FisheyeExportConfig>],
    stop: &std::sync::atomic::AtomicBool,
) -> Option<Vec<Option<bool>>> {
    let mut found = Vec::with_capacity(exports.len());
    for export in exports {
        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            return None;
        }
        found.push(export.as_ref().map(|export| export.reads_enhanced_frames()));
    }
    Some(found)
}

/// What a joined run does once its takes' folders are looked through
/// (`found`: a part a place, `None` for a clip that could read no enhanced
/// frames; `parts`: the run's clips): whether its parts read enhanced
/// frames, which they do when every one of them has some; and, where they
/// are left out although some clips have them, the place of the first clip
/// that has none, to be named. An answer that does not cover the run (the
/// look was stopped) reads the capture.
pub(super) fn join_check_outcome(found: &[Option<bool>], parts: usize) -> (bool, Option<usize>) {
    if found.len() != parts {
        return (false, None);
    }
    let has: Vec<bool> = found.iter().map(|f| *f == Some(true)).collect();
    let enhanced = vr180_pipeline::join::parts_read_enhanced(&has);
    (enhanced, if enhanced || !has.contains(&true) { None } else { has.iter().position(|h| !h) })
}

/// What a refusal adds for two Lens Space clips whose frames differ because
/// one is stabilized and the other is not.
pub(super) const JOIN_LENS_MARGIN_HINT: &str = "In the Lens Space view a stabilized clip's frame is larger than that of a clip that is not stabilized: turn stabilization on for both clips, or off for both.";

/// What a joined run says when it leaves enhanced frames out although some
/// of its clips have them: `name` is the first clip that has none.
pub(super) fn join_enhanced_note(name: &str, zh: bool) -> String {
    if zh {
        format!("未使用增强帧：合成文件只有在每个片段都有增强帧时才读取，{name} 没有。")
    } else {
        format!("Enhanced frames left out: a joined file reads them only when every clip has some, and {name} has none.")
    }
}

/// Why two clips cannot go into one joined file, for the user (`a` is the
/// run's first clip), in Chinese (`zh`) or English.
pub(super) fn join_mismatch_text(a: &str, b: &str, why: vr180_pipeline::join::Mismatch, zh: bool) -> String {
    use vr180_pipeline::join::Mismatch as M;
    match why {
        M::Frame((w, h), (w2, h2)) => if zh {
            format!("{a} 的输出尺寸是 {w} × {h}，{b} 是 {w2} × {h2}。合成需要同一尺寸：请在“格式”里选一个所有片段都适用的尺寸。")
        } else {
            format!("{a} exports at {w} × {h} and {b} at {w2} × {h2}. A joined file has one frame size: pick a size in Format that every clip can use.")
        },
        M::FrameRate(x, y) => if zh {
            format!("{a} 是 {x:.2} fps，{b} 是 {y:.2} fps：帧率不同的片段无法合成。")
        } else {
            format!("{a} is {x:.2} fps and {b} is {y:.2} fps: clips of different frame rates cannot be joined.")
        },
        M::Encoding => if zh {
            format!("{a} 和 {b} 的编码不同（编码器或位深）。raw 视频片段始终按 10 位写出：请在“格式”里把 H.265 设为 10-bit。")
        } else {
            format!("{a} and {b} come out in different encodings (codec or bit depth). A raw video clip is always written at 10 bits: set H.265 to 10-bit in Format.")
        },
        M::View => if zh {
            format!("{a} 和 {b} 的输出视图不同：合成文件里的片段必须是同一种视图（VR180、镜头空间或重新取景），单眼和双眼片段也不能混合。")
        } else {
            format!("{a} and {b} are exported in different views: the clips of a joined file share one (VR180, Lens Space or Reframed), and all hold one eye or all two.")
        },
    }
}

/// The name of a joined file, from the one its first clip would export
/// under by itself: `<stem>_Joined_<tag>.<ext>` for `<stem>_<tag>.<ext>`.
/// `immersive`: lens frames, joined as an Apple Immersive Video file
/// (`<stem>_Joined_Lens.aivu`, whatever the codec).
pub(super) fn joined_output_for(first_source: &std::path::Path, codec: ExportCodec, single_eye: bool, immersive: bool) -> PathBuf {
    let stem = first_source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let out = App::default_output_for(&first_source.with_file_name(format!("{stem}_Joined.x")), codec, single_eye, immersive);
    if immersive { out.with_extension("aivu") } else { out }
}

impl App {
    /// The clips `run` (list entries, in order) as one joined file under
    /// `opts`: where it and its parts go. `Err`: why these clips cannot be
    /// one file, for the user. Nothing is written yet.
    pub(super) fn plan_join(&self, run: &[usize], opts: &ExportOptions) -> Result<JoinRun, String> {
        let mut exports = Vec::with_capacity(run.len());
        let mut parts = Vec::with_capacity(run.len());
        for (k, &i) in run.iter().enumerate() {
            let item = &self.batch[i];
            let name = clip_label(&item.path);
            if item.source_kind.is_still() {
                return Err(format!("{name} {}", tr("is a still photo: a joined file is made of video clips.")));
            }
            if item.source_kind.is_raw_video() && opts.raw_dng_sequence {
                return Err(tr("A DNG sequence is a folder of frames for each clip: it cannot be joined. Pick a video format in Format.").to_string());
            }
            // (No segments: what the export writes does not depend on them.)
            exports.push(self.export_cfg_for(i, std::path::Path::new(""), opts, Vec::new()).0);
            // A part is named as the clip's own export would be, with its
            // place in front: what is left behind when no file is joined
            // can be used as it is.
            let own = Self::default_output_for(&item.path, opts.codec_for(&item.export_settings(), false), item.single_eye(),
                item.settings.lens_frame_view(item.source_kind));
            parts.push((item.path.clone(), format!("{:02}_{}", k + 1, clip_label(&own))));
        }
        let (immersive, enhanced, check) = Self::join_enhanced_plan(&self.batch, run, exports)?;
        let head = &self.batch[run[0]];
        let four_channel: Vec<bool> = run.iter().map(|&i| self.batch[i].four_channel_sound).collect();
        let sound = joined_sound(head.settings.audio_format, &four_channel).map_err(|odd| {
            let name = clip_label(&self.batch[run[odd]].path);
            format!("{name} {}", tr("has no four-channel sound: a joined file with Ambisonic or APAC sound needs it in every clip. Set the first clip's Audio to Stereo, or leave this clip out."))
        })?;
        // (The parts agree on it: a lens frame beside anything else is another view.)
        let mut output = joined_output_for(&head.path, opts.codec_for(&head.export_settings(), false), head.single_eye(), immersive);
        if let Some(dir) = &self.batch_out_dir {
            if let Some(name) = output.file_name() {
                output = dir.join(name);
            }
        }
        let output = unused_path(output);
        let parts_dir = unused_path(output.with_extension("parts"));
        Ok(JoinRun {
            parts: parts.into_iter().map(|(clip, name)| (clip, parts_dir.join(name))).collect(),
            output,
            parts_dir,
            first_cfg: None,
            sound,
            immersive,
            joining: false,
            enhanced,
            check,
            note: None,
        })
    }

    /// What a joined run of `run` has to settle about enhanced frames, from
    /// its parts' exports as planned (`exports`, in the run's order): every
    /// part of the file reads them, or none does. A clip that cannot read
    /// any (no raw take, no folder beside it, the option off) settles that
    /// for the run; where a clip could, its folder is looked through first,
    /// as the run's first job ([`JoinCheck`]). Returns whether the parts
    /// are lens frames, what is settled already (`None`: the look decides)
    /// and that job. `Err`: why these clips cannot be one file either way.
    pub(super) fn join_enhanced_plan(
        batch: &[BatchItem],
        run: &[usize],
        exports: Vec<vr180_pipeline::fisheye_export::FisheyeExportConfig>,
    ) -> Result<(bool, Option<bool>, Option<JoinCheck>), String> {
        let could: Vec<bool> = exports.iter().map(|c| c.source_kind.is_raw_video() && c.raw_enhanced_dir.is_some()).collect();
        let plain = Self::join_parts_agree(batch, run, &exports, false);
        let (immersive, agree) = if !could.contains(&true) {
            return Ok((plain?.immersive(), Some(false), None));
        } else if could.contains(&false) {
            // Read from the capture whatever the folders hold: they are
            // looked through only to say so where frames are left out.
            (plain?.immersive(), (Ok(()), Ok(())))
        } else {
            // A size picked for enhanced frames is not the size picked
            // without them: which of the two the run's parts have to agree
            // at is known once the folders are looked through. A refusal
            // that is the same either way needs no look.
            let with = Self::join_parts_agree(batch, run, &exports, true);
            if let (Err(a), Err(b)) = (&plain, &with) {
                if a == b {
                    return Err(a.clone());
                }
            }
            let immersive = [&plain, &with].into_iter().find_map(|p| p.as_ref().ok()).is_some_and(|p| p.immersive());
            (immersive, (plain.map(|_| ()), with.map(|_| ())))
        };
        Ok((immersive, None, Some(JoinCheck {
            exports: exports.into_iter().zip(&could).map(|(export, could)| could.then_some(export)).collect(),
            agree_plain: agree.0,
            agree_enhanced: agree.1,
            found: None,
        })))
    }

    /// The parts a joined run of `run` would be made of, from their exports
    /// as planned (`exports`, in the run's order), read from their takes'
    /// enhanced frames (`enhanced`) or from the capture: the first part
    /// when they come out as one shape, or why they do not, for the user.
    /// Each part's frame is the one its export writes
    /// (`vr180_pipeline::join::planned_part`).
    pub(super) fn join_parts_agree(
        batch: &[BatchItem],
        run: &[usize],
        exports: &[vr180_pipeline::fisheye_export::FisheyeExportConfig],
        enhanced: bool,
    ) -> Result<vr180_pipeline::join::PlannedPart, String> {
        use vr180_pipeline::join::{mismatch, planned_part, Mismatch, Unjoinable};
        let mut first: Option<(String, vr180_pipeline::join::PlannedPart, bool)> = None;
        for (&i, cfg) in run.iter().zip(exports) {
            let item = &batch[i];
            let name = clip_label(&item.path);
            let part = planned_part(cfg, item.part_source(enhanced)).map_err(|why| match why {
                Unjoinable::Spatial => tr("A spatial video cannot be joined: export the clips one by one, or pick H.265 or ProRes in Format.").to_string(),
                Unjoinable::LensFrame => format!("{name} {}", tr("is in the Lens Space view through the camera's lens. Clips in that view are joined as an Apple Immersive Video file, which needs macOS 26 or later and two eyes in every clip: here they cannot be joined. Pick the Equidistant lens, or another view.")),
            })?;
            match &first {
                None => first = Some((name, part, cfg.stabilize)),
                Some((first_name, a, stabilized)) => {
                    if let Some(why) = mismatch(a, &part) {
                        let mut text = join_mismatch_text(first_name, &name, why, cur_lang() == Lang::Zh);
                        // Two lens frames of one size apart by the
                        // stabilizer's margin: no size in Format mends that.
                        if matches!(why, Mismatch::Frame(..)) && a.immersive() && part.immersive() && *stabilized != cfg.stabilize {
                            text.push(' ');
                            text.push_str(tr(JOIN_LENS_MARGIN_HINT));
                        }
                        return Err(text);
                    }
                }
            }
        }
        first.map(|(_, part, _)| part).ok_or_else(|| tr("no clip was exported").to_string())
    }

    /// The clips of a joined run are all written: put them together, as
    /// the run's last job (so its progress, Stop, and the hand-over while
    /// the window is covered are those of any export). `false`: this is no
    /// joined run, or one that ends here without a file.
    pub(super) fn start_join(&mut self) -> bool {
        let Some(join) = self.join_run.as_mut() else { return false };
        if join.joining {
            return false;
        }
        // One file or none: every clip of the run has to be there.
        let missing = join.parts.iter()
            .find(|(clip, _)| !self.batch.iter().any(|b| b.path == *clip && b.status == BatchStatus::Done))
            .map(|(clip, _)| clip_label(clip));
        if let Some(name) = missing {
            self.end_join(Err(format!("{name} {}", tr("was not exported"))));
            return false;
        }
        let Some(cfg) = join.first_cfg.clone() else {
            self.end_join(Err(tr("no clip was exported").to_string()));
            return false;
        };
        join.joining = true;
        let parts: Vec<PathBuf> = join.parts.iter().map(|(_, part)| part.clone()).collect();
        let output = join.output.clone();
        let (sound, immersive) = (join.sound, join.immersive);
        tracing::info!("join: {} parts → {}", parts.len(), output.display());
        let (progress_tx, progress_rx) = crossbeam_channel::bounded(256);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        let out = output.clone();
        let handle = std::thread::spawn(move || {
            if immersive {
                // Each part states its own lens: the joined file gets them all.
                return vr180_pipeline::join::join_immersive(&parts, &out, sound, move |p| { let _ = progress_tx.try_send(p); }, cancel_for_thread);
            }
            vr180_pipeline::join::join_exports(&parts, &out, sound, move |p| { let _ = progress_tx.try_send(p); }, cancel_for_thread)?;
            let mut cfg = *cfg;
            cfg.output_path = out.clone();
            vr180_pipeline::fisheye_export::state_export_metadata(cfg, &out)
        });
        self.export_job = Some(ExportJob {
            output_path: output,
            handle,
            progress_rx,
            cancel,
            last_progress: None,
            started_at: std::time::Instant::now(),
        });
        true
    }

    /// A joined run ends: with its file (the parts go), or without one
    /// (`Err`: why; what was exported stays where it was written).
    pub(super) fn end_join(&mut self, outcome: Result<(), String>) {
        let Some(join) = self.join_run.take() else { return };
        match outcome {
            Ok(()) => {
                // The parts are this run's own files, in a folder it made:
                // they go, and the folder with them unless something else
                // is in it.
                for (_, part) in &join.parts {
                    let _ = std::fs::remove_file(part);
                }
                let _ = std::fs::remove_dir(&join.parts_dir);
                tracing::info!("join: done → {}", join.output.display());
                self.joined_output = Some(join.output);
                self.joined_note = join.note;
            }
            Err(why) => {
                let why = why.trim_start_matches("ffmpeg: ").trim_start_matches("join: ").to_string();
                tracing::warn!("join: no joined file: {why}");
                let kept = join.parts.iter().filter(|(_, part)| part.exists()).count();
                let mut text = format!("{}: {why}", tr("Not joined"));
                if kept > 0 {
                    text.push_str(&format!(" · {} {}", tr("the clips exported so far are in"), join.parts_dir.display()));
                } else {
                    let _ = std::fs::remove_dir(&join.parts_dir);
                }
                self.join_notice = Some(text);
            }
        }
    }

    /// "Export all" writes the whole list as one file: that is chosen, and
    /// the list holds several clips to join.
    pub(super) fn joins_all(&self) -> bool {
        self.join_clips && self.batch.iter().filter(|b| b.ready()).count() >= 2
    }

    /// A joined run's first job, where a clip could read enhanced frames
    /// ([`JoinCheck`]): each such take's folder is looked through for its
    /// clip's stretch, on a thread. The run goes on from
    /// [`Self::end_join_check`].
    pub(super) fn start_join_check(&mut self) {
        let Some(join) = self.join_run.as_mut() else { return };
        let Some(check) = join.check.as_mut() else {
            // Nothing to look up: the parts read the capture.
            join.enhanced = Some(false);
            self.start_next_batch_item();
            return;
        };
        let exports = std::mem::take(&mut check.exports);
        let (tx, rx) = crossbeam_channel::bounded(1);
        check.found = Some(rx);
        let output = join.output.clone();
        // (The bar shows the last export's note until an export clears it.)
        vr180_pipeline::fisheye_export::set_export_path_note(None);
        tracing::info!("join: looking for enhanced frames in {} of {} takes", exports.iter().flatten().count(), exports.len());
        let (_progress_tx, progress_rx) = crossbeam_channel::bounded(1);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = cancel.clone();
        let handle = std::thread::spawn(move || -> vr180_pipeline::Result<()> {
            if let Some(found) = join_look(&exports, &stop) {
                let _ = tx.send(found);
            }
            Ok(())
        });
        self.export_job = Some(ExportJob {
            output_path: output,
            handle,
            progress_rx,
            cancel,
            last_progress: None,
            started_at: std::time::Instant::now(),
        });
    }

    /// The look for enhanced frames is over: the run's parts read them, all
    /// or none, and are checked against each other at the size that goes
    /// with it. Then the first clip is written, or the run ends here with
    /// the reason and nothing exported.
    pub(super) fn end_join_check(&mut self) {
        let Some(join) = self.join_run.as_mut() else { return };
        let check = join.check.take();
        let found: Vec<Option<bool>> = check.as_ref()
            .and_then(|c| c.found.as_ref())
            .and_then(|rx| rx.try_recv().ok())
            .unwrap_or_default();
        let unqueue = |batch: &mut Vec<BatchItem>| {
            for b in batch.iter_mut() {
                if b.status == BatchStatus::Queued {
                    b.status = BatchStatus::Idle;
                }
            }
        };
        if !self.batch_running {
            // Stopped by the user while it looked.
            unqueue(&mut self.batch);
            self.end_join(Err(tr("stopped").to_string()));
            return;
        }
        let (enhanced, odd) = join_check_outcome(&found, join.parts.len());
        // Left out although some clips have them: say which clip has none.
        let note = odd.map(|odd| join_enhanced_note(&join.parts.get(odd).map(|(clip, _)| clip_label(clip)).unwrap_or_default(), cur_lang() == Lang::Zh));
        let agree = match check {
            Some(c) => if enhanced { c.agree_enhanced } else { c.agree_plain },
            None => Ok(()),
        };
        tracing::info!("join: enhanced frames found {found:?}: the parts read {}", if enhanced { "them" } else { "the capture" });
        match agree {
            Ok(()) => {
                if let Some(note) = &note {
                    tracing::warn!("join: {note}");
                }
                join.enhanced = Some(enhanced);
                join.note = note;
                self.start_next_batch_item();
            }
            Err(why) => {
                // These clips cannot be one file after all: none is exported.
                tracing::warn!("join: not started: {why}");
                unqueue(&mut self.batch);
                self.join_run = None;
                self.join_notice = Some(why);
                self.finish_run();
            }
        }
    }

    /// Why a joined export did not start, or ended without its file: said
    /// in a window of its own, until the user has read it.
    pub(super) fn draw_join_notice(&mut self, ctx: &egui::Context) {
        let Some(note) = self.join_notice.clone() else { return };
        let mut close = false;
        egui::Window::new(tr("Join into one file"))
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(460.0)
            .show(ctx, |ui| {
                ui.label(note);
                ui.add_space(10.0);
                if ui.button("OK").clicked() {
                    close = true;
                }
            });
        if close {
            self.join_notice = None;
        }
    }
}
