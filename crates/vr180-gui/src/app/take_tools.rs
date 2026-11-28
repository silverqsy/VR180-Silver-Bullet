//! The Source panel's tools for a raw take: its archive copy and its camera original DNGs.

use super::*;

/// How small an archive copy of a raw clip is to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ArchiveSize {
    /// Every sample kept: as small as that allows.
    Lossless,
    /// One bit a pixel: the largest lossy copy worth making.
    #[default]
    HighQuality,
    /// The bits a pixel picked by hand.
    Custom,
}

/// What the Source panel writes of a raw take ITSELF, and not of the view
/// of it: no view, no size, no grade, no stabilization and no noise filter
/// applies, and what is written is read again as a source (a copy opens
/// like the take; the camera's own frames are where enhanced frames and a
/// raw converter's settings are looked for). Not an export choice: each is
/// started by its own button, for that one run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TakeJob {
    /// An archive copy of the take, at this many bits a pixel (`None`:
    /// every sample kept).
    Archive(Option<f64>),
    /// The frames as the sensor recorded them: one Bayer DNG an eye.
    Originals,
}

/// A run of the queue that writes [`TakeJob`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TakeRun {
    pub(super) job: TakeJob,
    /// The whole take, whatever its trim (`false`: from In to Out).
    pub(super) whole: bool,
}

/// What a run of the queue writes of a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum RunOutput {
    /// A still photo's file.
    Still,
    /// The take itself ([`TakeJob`]).
    Take(TakeJob),
    /// A raw clip's developed view as a folder of DNG frames.
    DngFrames,
    /// The view as a video file.
    Video,
}

/// What a run writes of a clip of `kind`: the take itself in a run started
/// for that (`take`), else the view as `opts` say.
pub(super) fn run_output(take: Option<TakeRun>, kind: vr180_pipeline::SourceKind, opts: &ExportOptions) -> RunOutput {
    if kind.is_still() {
        RunOutput::Still
    } else if let Some(run) = take.filter(|_| kind.is_raw_video()) {
        RunOutput::Take(run.job)
    } else if kind.is_raw_video() && opts.raw_dng_sequence {
        RunOutput::DngFrames
    } else {
        RunOutput::Video
    }
}

/// Whose takes a run of the Source panel's tools writes. Each has a start
/// button of its own: the tools stand under the facts of the clip on
/// screen, so a run is of that clip, and a tick in the list sends no run
/// elsewhere unless the button of the ticked clips is the one clicked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TakeSet {
    /// The clip on screen, ticked or not.
    OnScreen,
    /// The clips ticked in the list.
    Ticked,
}

/// The list entries a run of `job` for `of` started now would write: the
/// clip on screen (`active`), or the ticked clips, where the run is for
/// them. An archive copy is made of a camera's take; the camera's own
/// frames are written from any raw take.
pub(super) fn take_clips(batch: &[BatchItem], active: Option<usize>, job: TakeJob, of: TakeSet) -> Vec<usize> {
    let fits = |b: &BatchItem| match job {
        TakeJob::Archive(_) => b.camera_take(),
        TakeJob::Originals => b.ready() && b.source_kind.is_raw_video(),
    };
    match of {
        TakeSet::OnScreen => active.filter(|&i| batch.get(i).is_some_and(fits)).into_iter().collect(),
        TakeSet::Ticked => batch.iter().enumerate().filter(|(_, b)| b.selected && fits(b)).map(|(i, _)| i).collect(),
    }
}

/// The entries behind a tool's two start buttons: the clip on screen, and
/// the ticked clips. The second has none where the only clip ticked is the
/// one on screen: the first button is that run.
pub(super) fn take_buttons(batch: &[BatchItem], active: Option<usize>, job: TakeJob) -> (Vec<usize>, Vec<usize>) {
    let on_screen = take_clips(batch, active, job, TakeSet::OnScreen);
    let ticked = take_clips(batch, active, job, TakeSet::Ticked);
    let ticked = if ticked == on_screen { Vec::new() } else { ticked };
    (on_screen, ticked)
}

/// A run that writes the takes themselves begins: the export queue's state
/// is put aside, and the run's `clips` alone are armed.
pub(super) fn queue_put_aside(batch: &mut [BatchItem], clips: &[usize]) {
    for item in batch.iter_mut() {
        item.before_take = Some(std::mem::replace(&mut item.status, BatchStatus::Idle));
    }
    for &i in clips {
        batch[i].status = BatchStatus::Queued;
        batch[i].take_failed = None;
    }
}

/// Such a run (of `job`) is over: the export queue is as the run found it,
/// and a take the run could not write keeps the reason.
pub(super) fn queue_brought_back(batch: &mut [BatchItem], job: Option<TakeJob>) {
    for item in batch.iter_mut() {
        let Some(before) = item.before_take.take() else { continue };
        if let (Some(job), BatchStatus::Failed(why)) = (job, std::mem::replace(&mut item.status, before)) {
            item.take_failed = Some((job, why.trim_start_matches("ffmpeg: ").to_string()));
        }
    }
}

/// Which of the two has its choices open in the Source panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TakeTool { Archive, Originals }

/// The Source panel's choices for the two.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TakeTools {
    pub(super) open: Option<TakeTool>,
    /// The size of an archive copy, and the bits a pixel of one picked by
    /// hand (kept across launches, see `App::save_take_copy`).
    pub(super) size: ArchiveSize,
    pub(super) bits: f32,
    /// The whole take is written, whatever its trim.
    pub(super) archive_whole: bool,
    pub(super) originals_whole: bool,
}

impl Default for TakeTools {
    fn default() -> Self {
        Self { open: None, size: ArchiveSize::default(), bits: 0.75, archive_whole: false, originals_whole: false }
    }
}

impl TakeTools {
    /// The bits a pixel of the archive copy picked (`None`: every sample kept).
    pub(super) fn archive_bits(&self) -> Option<f64> {
        use vr180_pipeline::raw_archive::{BITS_LEAST, BITS_MOST};
        match self.size {
            ArchiveSize::Lossless => None,
            ArchiveSize::HighQuality => Some(BITS_MOST),
            ArchiveSize::Custom => Some((self.bits as f64).clamp(BITS_LEAST, BITS_MOST)),
        }
    }

    /// The choice as the file that keeps it says it, and back.
    pub(super) fn kept(&self) -> String {
        match self.size {
            ArchiveSize::Lossless => format!("lossless {:.2}", self.bits),
            ArchiveSize::HighQuality => format!("high {:.2}", self.bits),
            ArchiveSize::Custom => format!("custom {:.2}", self.bits),
        }
    }

    /// The run the archive button starts for `takes`, and the one of the
    /// camera's own frames: a run of takes without a trim writes them whole
    /// whatever is picked.
    pub(super) fn archive_run(&self, takes: &[TakeView]) -> TakeRun {
        TakeRun { job: TakeJob::Archive(self.archive_bits()), whole: self.archive_whole || !takes.iter().any(|t| t.written(false).2) }
    }

    pub(super) fn originals_run(&self, takes: &[TakeView]) -> TakeRun {
        TakeRun { job: TakeJob::Originals, whole: self.originals_whole || !takes.iter().any(|t| t.written(false).2) }
    }

    pub(super) fn from_kept(text: &str) -> Self {
        let mut words = text.split_whitespace();
        let size = match words.next() {
            Some("lossless") => ArchiveSize::Lossless,
            Some("custom") => ArchiveSize::Custom,
            _ => ArchiveSize::HighQuality,
        };
        let bits = words.next().and_then(|w| w.parse::<f32>().ok()).filter(|b| b.is_finite()).map_or(0.75, |b| b.clamp(0.5, 1.0));
        Self { size, bits, ..Default::default() }
    }
}

/// What the list keeps of a raw take beyond its frame.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TakeFacts {
    /// The length of its capture.
    pub(super) capture_bytes: u64,
    /// What it says of itself as an archive copy (`None`: a camera's take).
    pub(super) archive: Option<vr180_fisheye::x6b::ArchiveNote>,
    /// The share of a frame's samples that lie outside its lens's image
    /// circle (`vr180_pipeline::x6_raw::PictureArea`): an archive copy
    /// stores them as black.
    pub(super) outside: f64,
}

impl TakeFacts {
    /// Read them from the take's small files; `None` for any other source.
    pub(super) fn read(kind: vr180_pipeline::SourceKind, path: &std::path::Path) -> Option<Self> {
        if !kind.is_raw_video() {
            return None;
        }
        let take = vr180_pipeline::x6_raw::describe(path).ok()?;
        Some(Self {
            capture_bytes: take.capture_bytes,
            archive: vr180_pipeline::x6_raw::archive_note(path),
            outside: vr180_pipeline::x6_raw::PictureArea::of(&take).share_outside(),
        })
    }
}

/// An archive copy a run made of a list entry's take.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TakeCopy {
    /// The copy's folder.
    pub(super) dir: PathBuf,
    /// Its length, and the length of the same frames in the take.
    pub(super) bytes: u64,
    pub(super) source_bytes: u64,
}

/// The stretch of its take an archive copy holds, in the take's seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CopyExtent {
    /// When the copy's first frame lies in the take, and how long the copy is.
    pub(super) begins_s: f64,
    pub(super) duration_s: f64,
    /// It begins with the take's first frame, and ends with its last.
    pub(super) from_start: bool,
    pub(super) to_end: bool,
}

impl CopyExtent {
    /// From what the copy says of itself (one that does not say is taken
    /// to hold its whole take), and the copy's own length.
    pub(super) fn of(note: &vr180_fisheye::x6b::ArchiveNote, duration_s: f64) -> Self {
        let first = note.first_pair.unwrap_or(0);
        let to_end = match (note.pairs, note.pairs_of_source) {
            (Some(pairs), Some(of)) => first + pairs >= of,
            _ => true,
        };
        Self { begins_s: note.first_pair_s.unwrap_or(0.0).max(0.0), duration_s, from_start: first == 0, to_end }
    }

    /// A trim set on the take, as it lies in the copy: an end that falls on
    /// the copy's own end, or past it, is no trim there.
    pub(super) fn trim_in_copy(&self, trim: (Option<f64>, Option<f64>), fps: f32) -> (Option<f64>, Option<f64>) {
        let half = 0.5 / fps.max(1.0) as f64;
        (
            trim.0.map(|t| t - self.begins_s).filter(|t| *t > half),
            trim.1.map(|t| t - self.begins_s).filter(|t| *t < self.duration_s - half).map(|t| t.max(0.0)),
        )
    }

    /// A trim set on the copy, as it lies in the take. Where the copy is a
    /// stretch of the take its own ends are a trim there: the clip is that
    /// stretch, whichever of the two files it is read from. (The copy's
    /// end as a trim: half a frame before where its last frame ends, so
    /// that the take's next frame, a little early as frames can be, stays
    /// out.)
    pub(super) fn trim_in_take(&self, trim: (Option<f64>, Option<f64>), fps: f32) -> (Option<f64>, Option<f64>) {
        let half = 0.5 / fps.max(1.0) as f64;
        (
            match trim.0 { Some(t) => Some(self.begins_s + t), None => (!self.from_start).then_some(self.begins_s) },
            match trim.1 { Some(t) => Some(self.begins_s + t), None => (!self.to_end).then_some(self.begins_s + self.duration_s - half) },
        )
    }
}

/// A list entry that reads an archive copy in the place of its take, which
/// is not at hand. The entry stays the take's: a project names the take,
/// and the take is read again once it is back.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StandIn {
    pub(super) take: PathBuf,
    /// Where the copy lies in that take.
    pub(super) extent: CopyExtent,
    /// The entry's trim on the take when the copy took its place, and what
    /// that became on the copy: a trim not touched since goes back to the
    /// take as it was, to the digit.
    pub(super) trim_on_take: (Option<f64>, Option<f64>),
    pub(super) trim_on_copy: (Option<f64>, Option<f64>),
}

impl StandIn {
    /// The copy `extent` describes is read in the place of `take`, whose
    /// entry has the trim `trim_on_take`.
    pub(super) fn of(take: PathBuf, extent: CopyExtent, trim_on_take: (Option<f64>, Option<f64>), fps: f32) -> Self {
        Self { take, extent, trim_on_take, trim_on_copy: extent.trim_in_copy(trim_on_take, fps) }
    }

    /// The entry's trim `now`, set on the copy, as it lies in the take.
    pub(super) fn trim_in_take(&self, now: (Option<f64>, Option<f64>), fps: f32) -> (Option<f64>, Option<f64>) {
        if now == self.trim_on_copy { self.trim_on_take } else { self.extent.trim_in_take(now, fps) }
    }
}

/// The archive copy to read in the place of the take that was at `take`: a
/// folder beside where the take was, or in `also` (the output folder),
/// whose name begins with the take's and whose note says it was made of
/// that take. Of several: one that keeps every sample before one that does
/// not, then the one that holds the most of the take, then the one of the
/// most bits a pixel. Touches the disk.
pub(super) fn copy_standing_in(take: &std::path::Path, also: Option<&std::path::Path>) -> Option<(PathBuf, vr180_fisheye::x6b::ArchiveNote)> {
    let name = take.file_name()?.to_str()?;
    let mut found: Vec<(PathBuf, vr180_fisheye::x6b::ArchiveNote)> = Vec::new();
    for dir in [take.parent(), also].into_iter().flatten() {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let file_name = entry.file_name();
            let Some(folder) = file_name.to_str() else { continue };
            if folder == name || !folder.starts_with(name) || found.iter().any(|(path, _)| *path == entry.path()) {
                continue;
            }
            if let Some(note) = vr180_fisheye::x6b::read_archive_note(&entry.path()).filter(|n| n.source.as_deref() == Some(name)) {
                found.push((entry.path(), note));
            }
        }
    }
    found.sort_by(|(a_path, a), (b_path, b)| {
        let rank = |n: &vr180_fisheye::x6b::ArchiveNote| (n.exact, n.pairs.unwrap_or(u64::MAX), n.bits_per_pixel.unwrap_or(0.0));
        let (ra, rb) = (rank(a), rank(b));
        rb.0.cmp(&ra.0).then(rb.1.cmp(&ra.1)).then(rb.2.total_cmp(&ra.2)).then(a_path.cmp(b_path))
    });
    found.into_iter().next()
}

/// Which file a list entry is to read.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SourceChoice {
    /// The one it reads.
    Same,
    /// Its take, which is back.
    Take(PathBuf),
    /// This archive copy, in the place of its take that is not there.
    Copy(PathBuf, vr180_fisheye::x6b::ArchiveNote),
}

/// The file an entry that reads `path` is to read: the take first, a copy
/// of it only while the take is not at hand (`stand_in`: the entry reads a
/// copy in its take's place already; `also`: the output folder, where a
/// copy may have been written). Touches the disk.
pub(super) fn source_choice(path: &std::path::Path, stand_in: Option<&StandIn>, also: Option<&std::path::Path>) -> SourceChoice {
    match stand_in {
        Some(stand_in) if vr180_pipeline::x6_raw::is_take(&stand_in.take) => SourceChoice::Take(stand_in.take.clone()),
        Some(_) => SourceChoice::Same,
        None if path.exists() => SourceChoice::Same,
        None => copy_standing_in(path, also).map_or(SourceChoice::Same, |(dir, note)| SourceChoice::Copy(dir, note)),
    }
}

/// Of what is opened together, without the archive copies whose take is
/// opened too: a take and its copy are one recording, and the take is the
/// one to work on. `source_of`: the take a folder says it is a copy of.
/// Returns what is opened and the copies left out.
pub(super) fn without_copies_of_opened_takes(paths: Vec<PathBuf>, source_of: impl Fn(&std::path::Path) -> Option<String>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let names: std::collections::HashSet<String> = paths.iter().filter_map(|p| Some(p.file_name()?.to_str()?.to_owned())).collect();
    paths.into_iter().partition(|p| !source_of(p).is_some_and(|take| names.contains(&take)))
}

/// What the list says when several of a project's takes are read from
/// their copies.
pub(super) fn stand_ins_notice(takes: usize, zh: bool) -> String {
    if zh { format!("{takes} 条原素材不在：改读它们的存档副本。") } else { format!("{takes} takes are not at hand: their archive copies are read in their place.") }
}

/// What the list says when an entry changes the file it reads.
pub(super) fn stand_in_notice(take: &std::path::Path, copy: &std::path::Path, to_copy: bool, zh: bool) -> String {
    let (take, copy) = (clip_label(take), clip_label(copy));
    match (to_copy, zh) {
        (true, true) => format!("{take} 不在：改读它的存档副本 {copy}。"),
        (true, false) => format!("{take} is not at hand: its archive copy {copy} is read in its place."),
        (false, true) => format!("{take} 回来了：重新读原素材，不再读副本 {copy}。"),
        (false, false) => format!("{take} is back: it is read again, and its archive copy {copy} no longer."),
    }
}

/// A take as the Source panel's two operations show it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TakeView {
    pub(super) path: PathBuf,
    pub(super) fps: f32,
    pub(super) duration_s: f64,
    /// Its trim as set (the clip on screen: as it is set right now).
    pub(super) trim: (Option<f64>, Option<f64>),
    pub(super) frame_samples: u64,
    pub(super) capture_bytes: u64,
    /// The share of a frame's samples outside its lens's image circle.
    pub(super) outside: f64,
    /// It is an archive copy, and one that does not hold every sample.
    pub(super) copy: bool,
    pub(super) lossy_copy: bool,
}

impl TakeView {
    /// The seconds a run writes of it, from and to, and whether that is
    /// less than the take: its stretch from In to Out, or all of it
    /// (`whole`, and a take without a trim).
    pub(super) fn written(&self, whole: bool) -> (f64, f64, bool) {
        let all = self.duration_s.max(0.0);
        let from = self.trim.0.unwrap_or(0.0).clamp(0.0, all);
        let to = self.trim.1.unwrap_or(all).clamp(from, all);
        let half = 0.5 / self.fps.max(1.0) as f64;
        if whole || (from <= half && to >= all - half) { (0.0, all, false) } else { (from, to, true) }
    }

    /// The length in its capture of `seconds` of it, near enough.
    pub(super) fn bytes_of(&self, seconds: f64) -> u64 {
        if self.duration_s <= 0.0 {
            return 0;
        }
        (self.capture_bytes as f64 * (seconds / self.duration_s).clamp(0.0, 1.0)).round() as u64
    }

    pub(super) fn name(&self) -> String {
        clip_label(&self.path)
    }
}

/// What the Source panel's take tools are shown for.
#[derive(Default)]
pub(super) struct TakeToolsView<'a> {
    /// The take on screen, where a run of archive copies started now would
    /// write it, and where a run of the camera's own frames would.
    pub(super) archive: &'a [TakeView],
    pub(super) originals: &'a [TakeView],
    /// The ticked takes such runs would write, behind a start button of
    /// their own ([`take_buttons`]).
    pub(super) archive_ticked: &'a [TakeView],
    pub(super) originals_ticked: &'a [TakeView],
    /// The copies the last runs made of takes in the list.
    pub(super) copies: &'a [TakeCopy],
    /// The takes the last run could not write: what it was to write, the
    /// take's name and the reason.
    pub(super) failed: &'a [(TakeJob, String, String)],
    /// The folder exports go to when it is not the source's own (its name).
    pub(super) out_dir: Option<&'a str>,
    /// No run is in flight.
    pub(super) idle: bool,
}

/// What was asked for there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TakeAsk {
    /// A run, and whose takes it is of.
    Start(TakeRun, TakeSet),
    /// The size picked is to be kept for the next launch.
    Keep,
}

/// The heading over the two buttons, and what the buttons say.
pub(super) const TAKE_TOOLS_TITLE: &str = "RAW Footage Archiving";
pub(super) const TAKE_ARCHIVE_LABEL: &str = "Compressed archive";
pub(super) const TAKE_ORIGINALS_LABEL: &str = "Camera original DNGs";
pub(super) const TAKE_ARCHIVE_HOVER: &str = "A smaller copy of the take for keeping, which opens like the take.";
pub(super) const TAKE_ARCHIVE_NOTE: &str = "The take's folder again with a smaller capture, which opens like the take: every frame, its timing and the files beside it are kept. The take itself is left as it is.";
pub(super) const TAKE_EXACT_LABEL: &str = "Lossless: every sample of the picture kept";
pub(super) const TAKE_EXACT_NOTE: &str = "Every sample of the picture is kept as the camera recorded it, and each frame is read back and compared before it is written. Most takes come out a quarter to two fifths smaller. A take that is noise for the most part gains little: where its copy would not be smaller, none is made.";
pub(super) const TAKE_LOSSY_NOTE: &str = "The copy is lossy: at 1 bit a pixel a frame is off by less than its own noise; below that the shadows lose their grain when they are lifted.";
pub(super) const TAKE_STRETCH_NOTE: &str = "A copy of a stretch is a shorter clip. What the app takes from a whole clip, such as the tone statistics and the stabilizer's reference, is then taken from the stretch, so its picture can differ slightly from the same stretch of the take.";
pub(super) const TAKE_COPY_SOURCE_NOTE: &str = "This clip is an archive copy of a take: a copy is made from the camera's take, not from another copy.";
pub(super) const TAKE_ORIGINALS_NOTE: &str = "Every frame as the sensor recorded it: one Bayer DNG per eye, named by the camera's frame number. No projection, no stabilization, no grading — for running the frames through a raw converter first.";
pub(super) const TAKE_ORIGINALS_PLACE_NOTE: &str = "The folder of that name beside the take is also where the app looks for enhanced frames and for Camera Raw settings.";
pub(super) const TAKE_LOSSY_SOURCE_NOTE: &str = "This clip is a lossy archive copy: the frames written are the copy's, close to what the sensor recorded but not the same.";
pub(super) const TAKE_LOSSY_TICKED_NOTE: &str = "Lossy archive copies are among the ticked clips: the frames written of those are the copies', close to what the sensor recorded but not the same.";
/// Over the takes of each start button, where a tool shows both.
pub(super) const TAKE_ON_SCREEN_TITLE: &str = "This take";
pub(super) const TAKE_TICKED_TITLE: &str = "Ticked takes";
pub(super) const TAKE_COPIED_NOTE: &str = "The take can now be moved away or deleted. That is yours to do: the app moves and deletes nothing. Once the take is gone the app reads its copy in its place, as long as the copy stays beside where the take was, or in the output folder.";
pub(super) const STAND_IN_HOVER: &str = "The take this clip belongs to is not at hand: its archive copy is read in its place. When the take is back, it is read again.";
pub(super) const TAKE_BUSY_HOVER: &str = "A run is in flight: this can start when it is over.";

/// Takes listed one by one under a run's choices; a longer run names the rest by their number.
pub(super) const TAKES_LISTED: usize = 6;

/// A length of time in a note.
pub(super) fn span_text(seconds: f64, zh: bool) -> String {
    let s = seconds.max(0.0);
    if s < 59.95 {
        return if zh { format!("{s:.1} 秒") } else { format!("{s:.1} s") };
    }
    let total = s.round() as u64;
    if zh { format!("{} 分 {} 秒", total / 60, total % 60) } else { format!("{} min {} s", total / 60, total % 60) }
}

/// A size on disk in a note.
pub(super) fn size_text(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 100.0 { format!("{gb:.0} GB") } else if gb >= 1.0 { format!("{gb:.1} GB") } else { format!("{:.0} MB", bytes as f64 / 1e6) }
}

/// One take of a run: its name and what the run writes of it.
pub(super) fn take_line(take: &TakeView, whole: bool, zh: bool) -> String {
    let (from, to, cut) = take.written(whole);
    let (name, all) = (take.name(), span_text(take.duration_s, zh));
    if !cut {
        return if zh { format!("{name} · 整条 · {all}") } else { format!("{name} · whole take · {all}") };
    }
    let kept = span_text(to - from, zh);
    if zh {
        format!("{name} · {} 到 {} · {all}中的 {kept}", format_time(from), format_time(to))
    } else {
        format!("{name} · {} to {} · {kept} of {all}", format_time(from), format_time(to))
    }
}

/// What copies of the takes' stretches from In to Out leave out, said
/// before they are made; `None` when no take of the run is trimmed.
pub(super) fn take_cut_warning(takes: &[TakeView], zh: bool) -> Option<String> {
    let cut = takes.iter().filter(|t| t.written(false).2).count();
    if cut == 0 {
        return None;
    }
    if let [take] = takes {
        let (from, to, _) = take.written(false);
        let (before, after) = (from, (take.duration_s - to).max(0.0));
        let half = 0.5 / take.fps.max(1.0) as f64;
        let (kept, before_text, after_text) = (span_text(to - from, zh), span_text(before, zh), span_text(after, zh));
        return Some(if zh {
            let left = match (before > half, after > half) {
                (true, true) => format!("入点之前的 {before_text}和出点之后的 {after_text}"),
                (true, false) => format!("入点之前的 {before_text}"),
                _ => format!("出点之后的 {after_text}"),
            };
            format!("副本只包含入点到出点之间的 {kept}：{left}不会进副本。确认这条素材别的部分都不要了，再删除原素材。")
        } else {
            let left = match (before > half, after > half) {
                (true, true) => format!("the {before_text} before In and the {after_text} after Out are"),
                (true, false) => format!("the {before_text} before In are"),
                _ => format!("the {after_text} after Out are"),
            };
            format!("Only the {kept} between In and Out go into the copy: {left} left out. Keep the take until you are sure nothing else of it is wanted.")
        });
    }
    Some(if zh {
        format!("{cut} 条素材设了入出点：副本只包含各自入点到出点之间的部分，之外的不会进副本。确认别的部分都不要了，再删除原素材。")
    } else {
        format!("{cut} of these takes are trimmed: each copy holds only the stretch between In and Out, and what lies outside is left out. Keep the takes until you are sure nothing else of them is wanted.")
    })
}

/// What an archive copy does with the corners of a frame that lie outside
/// the lens's image circle, said before the copy is made; `None` when no
/// take of the run has any (a frame that is the middle of the sensor).
pub(super) fn take_corners_note(takes: &[TakeView], zh: bool) -> Option<String> {
    if !takes.iter().any(|t| t.outside > 0.0) {
        return None;
    }
    Some(match (takes, zh) {
        ([take], true) => format!("画幅四角在镜头成像圈之外，占样本的 {:.0}%，没有画面，只有杂散光和噪声：存成黑。", take.outside * 100.0),
        ([take], false) => format!("The corners of the frame outside the lens's image circle, {:.0}% of its samples, hold no picture, only stray light and noise: they are stored as black.", take.outside * 100.0),
        (_, true) => "画幅四角在镜头成像圈之外的部分没有画面，只有杂散光和噪声：存成黑。".to_string(),
        (_, false) => "The corners of a frame outside the lens's image circle hold no picture, only stray light and noise: they are stored as black.".to_string(),
    })
}

/// The sizes a run of archive copies comes to, before it is made: the
/// takes', the part of them that is copied where that is less, and the
/// copies' where it is known beforehand (a lossy copy's `bits` a pixel).
pub(super) fn take_sizes_text(takes: &[TakeView], whole: bool, bits: Option<f64>, zh: bool) -> String {
    let (mut all, mut part, mut copy, mut cut) = (0u64, 0u64, 0f64, false);
    for take in takes {
        let (from, to, less) = take.written(whole);
        all += take.capture_bytes;
        part += if less { take.bytes_of(to - from) } else { take.capture_bytes };
        copy += bits.unwrap_or(0.0) * take.frame_samples as f64 * ((to - from) * take.fps as f64).round() / 8.0;
        cut |= less;
    }
    let n = takes.len();
    let mut text = match (n, zh) {
        (1, true) => format!("原素材 {}", size_text(all)),
        (1, false) => format!("Take {}", size_text(all)),
        (_, true) => format!("{n} 条素材共 {}", size_text(all)),
        (_, false) => format!("{n} takes {}", size_text(all)),
    };
    if cut {
        text.push_str(&if zh { format!(" · 要复制的部分 {}", size_text(part)) } else { format!(" · the part copied {}", size_text(part)) });
    }
    if bits.is_some() {
        text.push_str(&match (n, zh) {
            (1, true) => format!(" · 副本约 {}", size_text(copy as u64)),
            (1, false) => format!(" · the copy about {}", size_text(copy as u64)),
            (_, true) => format!(" · 副本共约 {}", size_text(copy as u64)),
            (_, false) => format!(" · the copies about {}", size_text(copy as u64)),
        });
    }
    text
}

/// Where a run's folders go: beside each take, or in the output folder
/// picked in the export bar (`out_dir`: its name), named after the take.
pub(super) fn take_place_text(takes: &[TakeView], job: TakeJob, out_dir: Option<&str>, zh: bool) -> String {
    // (Several takes: each folder is named after its own.)
    let name = clip_label(&App::take_job_dir_for(match takes {
        [take] => &take.path,
        _ => std::path::Path::new(if zh { "<素材名>" } else { "<take>" }),
    }, job));
    match (out_dir, zh) {
        (None, true) => format!("写在素材旁边：{name}"),
        (None, false) => format!("Written beside the take: {name}"),
        (Some(dir), true) => format!("写到输出文件夹 {dir}：{name}"),
        (Some(dir), false) => format!("Written into the output folder {dir}: {name}"),
    }
}

/// What the button that starts a run says: what the run writes, and whose
/// takes it is of (`n` of them, where they are the ticked ones).
pub(super) fn take_start_label(job: TakeJob, of: TakeSet, n: usize, zh: bool) -> String {
    let several = of == TakeSet::Ticked && n != 1;
    let what = match (job, zh) {
        (TakeJob::Archive(_), true) => "制作存档副本",
        (TakeJob::Archive(_), false) if several => "Make the archive copies",
        (TakeJob::Archive(_), false) => "Make the archive copy",
        (TakeJob::Originals, true) => "导出相机原片 DNG",
        (TakeJob::Originals, false) => "Write the camera original DNGs",
    };
    match (of, zh) {
        (TakeSet::OnScreen, true) => format!("{what}：当前素材"),
        (TakeSet::OnScreen, false) => format!("{what}: this take"),
        (TakeSet::Ticked, true) => format!("{what}：勾选的 {n} 条"),
        (TakeSet::Ticked, false) => format!("{what}: {n} ticked take{}", if several { "s" } else { "" }),
    }
}

/// The mark of something done (one the built-in fonts hold: a test checks.
/// The lighter check mark is in a Mac's system fonts and in none of a
/// Windows system's that the app loads, where it was drawn as an empty box).
pub(super) const DONE_MARK: &str = "✔";

/// A copy a run has made: its folder, its size, and that size beside the
/// same frames in the take.
pub(super) fn take_copy_line(copy: &TakeCopy, zh: bool) -> String {
    let name = clip_label(&copy.dir);
    if copy.bytes == 0 || copy.source_bytes == 0 {
        return format!("{DONE_MARK} {name}");
    }
    let (size, share) = (size_text(copy.bytes), copy.bytes as f64 * 100.0 / copy.source_bytes as f64);
    if zh { format!("{DONE_MARK} {name} · {size}，是原素材同样这些帧的 {share:.0}%") } else { format!("{DONE_MARK} {name} · {size}, {share:.0}% of the same frames in the take") }
}

/// Why a take was not written, as the panel says it: the pipeline's own
/// words, and in Chinese the one reason a user is likely to meet (a copy
/// that keeps every sample would not be smaller than a take of noise).
pub(super) fn take_failed_text(why: &str, zh: bool) -> String {
    if !zh || !why.contains(vr180_pipeline::raw_archive::NOT_SMALLER) {
        return why.to_string();
    }
    // "(8.14 against 7.65 bits a pixel on 3 of its frames)"
    match why.split_once('(').and_then(|(_, rest)| rest.split_once(" bits a pixel")).and_then(|(figures, _)| figures.split_once(" against ")) {
        Some((copy, take)) => format!("无损副本不会比这条素材更小：每像素 {copy} bit，原素材 {take} bit。它的帧以噪声为主，相机已经压得差不多了。可改用有损档。"),
        None => "无损副本不会比这条素材更小：它的帧以噪声为主，相机已经压得差不多了。可改用有损档。".to_string(),
    }
}

/// What the Source panel says of a clip that is an archive copy: how it
/// was made, and which stretch of its take it holds when it is not all of
/// it.
pub(super) fn archive_note_text(note: &vr180_fisheye::x6b::ArchiveNote, zh: bool) -> String {
    // (A copy that stores the corners outside the image circle as black keeps every sample of the picture.)
    let corners = note.outside_black.is_some();
    let kind = match (note.bits_per_pixel, corners, zh) {
        (Some(bits), _, _) if !note.exact => format!("{bits:.2} {}", if zh { tr_zh("bit a pixel") } else { "bit a pixel" }),
        (_, false, false) => "every sample kept".to_string(),
        (_, false, true) => tr_zh("every sample kept").to_string(),
        (_, true, false) => "every sample of the picture kept".to_string(),
        (_, true, true) => "画面内每个样本都保留".to_string(),
    };
    let kind = match (corners, zh) {
        (false, _) => kind,
        (true, false) => format!("{kind} · corners black"),
        (true, true) => format!("{kind} · 四角为黑"),
    };
    match (note.stretch(), zh) {
        (None, _) => kind,
        (Some((first, pairs, of)), true) => format!("{kind} · 原素材 {of} 帧中的第 {} 到 {} 帧", first + 1, first + pairs),
        (Some((first, pairs, of)), false) => format!("{kind} · frames {} to {} of the take's {of}", first + 1, first + pairs),
    }
}

impl App {
    /// `…/VR180SilverBullet2.0/take_copy`: the size of an archive copy, as
    /// last picked in the Source panel.
    pub(super) fn take_copy_config_path() -> Option<PathBuf> {
        crate::decoder::Settings::config_path()
            .and_then(|p| p.parent().map(|d| d.join("take_copy")))
    }

    pub(super) fn load_take_copy() -> TakeTools {
        Self::take_copy_config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map_or_else(TakeTools::default, |text| TakeTools::from_kept(&text))
    }

    pub(super) fn save_take_copy(&self) {
        if let Some(p) = Self::take_copy_config_path() {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, self.take_tools.kept());
        }
    }

    /// The folder of a raw video clip's archive copy: `<source_stem>_Archive`
    /// next to the take's folder.
    pub(super) fn default_archive_dir_for(source: &std::path::Path) -> PathBuf {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        source.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join(format!("{stem}_Archive"))
    }

    /// The folder a raw take's camera-original DNGs are READ from (enhanced
    /// frames, a raw converter's settings): the one named after it, beside
    /// it. An archive copy that has none of its own reads its take's, where
    /// that is beside the copy: the frames there are named by the camera's
    /// frame numbers, which a copy keeps. Touches the disk for a copy only.
    pub(super) fn originals_dir_for(source: &std::path::Path) -> PathBuf {
        let own = Self::default_frames_dir_for(source, true);
        let Some(take) = vr180_pipeline::x6_raw::archive_note(source).and_then(|note| note.source) else { return own };
        if own.is_dir() {
            return own;
        }
        let of_take = Self::default_frames_dir_for(&source.with_file_name(take), true);
        if of_take.is_dir() { of_take } else { own }
    }

    /// The folder a run that writes the take itself puts `job`'s output in,
    /// next to the take's folder.
    pub(super) fn take_job_dir_for(source: &std::path::Path, job: TakeJob) -> PathBuf {
        match job {
            TakeJob::Archive(_) => Self::default_archive_dir_for(source),
            TakeJob::Originals => Self::default_frames_dir_for(source, true),
        }
    }

    /// Start a run that writes the takes themselves ([`TakeRun`]), of the
    /// clip on screen or of the ticked clips (`of`, the button clicked): through
    /// the export queue, so its progress, Skip and Stop, and the hand-over
    /// while the window is covered are those of any export. The queue's own
    /// state is put aside for the run and comes back when it is over
    /// ([`Self::end_take_run`]): what was exported, or waits to be, is not
    /// touched by it. Never joined: each take is written to a folder of its
    /// own.
    pub(super) fn start_take_run(&mut self, run: TakeRun, of: TakeSet) {
        if self.export_job.is_some() || self.batch_running {
            return;
        }
        self.sync_active_clip_settings();
        self.end_take_run();
        for i in 0..self.batch.len() {
            let asked = match of {
                TakeSet::OnScreen => self.active_clip == Some(i),
                TakeSet::Ticked => self.batch[i].selected,
            };
            if asked {
                self.settle_source(i);
            }
        }
        let clips = take_clips(&self.batch, self.active_clip, run.job, of);
        if clips.is_empty() {
            tracing::warn!("take run: no raw take to write");
            return;
        }
        self.join_notice = None;
        self.joined_output = None;
        self.joined_note = None;
        self.join_run = None;
        queue_put_aside(&mut self.batch, &clips);
        if matches!(run.job, TakeJob::Archive(_)) {
            for &i in &clips {
                self.batch[i].copy = None;
            }
        }
        tracing::info!(
            "take run: {:?} of {} takes ({}), {}",
            run.job, clips.len(),
            match of { TakeSet::OnScreen => "the clip on screen", TakeSet::Ticked => "the ticked clips" },
            if run.whole { "whole" } else { "from In to Out" },
        );
        self.take_run = Some(run);
        self.begin_run(clips);
    }

    /// A run that wrote the takes themselves is over, ended or stopped: the
    /// export queue is as that run found it.
    pub(super) fn end_take_run(&mut self) {
        let job = self.take_run.take().map(|run| run.job);
        queue_brought_back(&mut self.batch, job);
    }

    /// The archive copy a run has just written into `dir` for list entry
    /// `idx`: what its folder says of it (two small reads).
    pub(super) fn note_copy(&mut self, idx: usize, dir: &std::path::Path) {
        let Some(item) = self.batch.get_mut(idx) else { return };
        let note = vr180_fisheye::x6b::read_archive_note(dir);
        let bytes = std::fs::metadata(dir.join(vr180_fisheye::x6b::CAPTURE_FILE)).map(|m| m.len()).unwrap_or(0);
        let pairs = note.as_ref().and_then(|n| n.pairs).unwrap_or(0);
        let source_bytes = note.as_ref().and_then(|n| n.source_bits_per_pixel)
            .map_or(0, |bits| (bits * (pairs * item.frame_samples()) as f64 / 8.0).round() as u64);
        item.copy = Some(TakeCopy { dir: dir.to_path_buf(), bytes, source_bytes });
    }

    /// A list entry as the Source panel's take tools show it (the clip on
    /// screen with its trim as it is set right now).
    pub(super) fn take_view(&self, idx: usize) -> TakeView {
        let b = &self.batch[idx];
        let live = self.active_clip == Some(idx) && self.loaded_path.as_deref() == Some(b.path.as_path());
        let settings = if live { &self.settings } else { &b.settings };
        TakeView {
            path: b.path.clone(),
            fps: b.fps,
            duration_s: b.duration_sec,
            trim: (settings.trim_in_s, settings.trim_out_s),
            frame_samples: b.frame_samples(),
            capture_bytes: b.take.as_ref().map_or(0, |t| t.capture_bytes),
            outside: b.take.as_ref().map_or(0.0, |t| t.outside),
            copy: b.take.as_ref().is_some_and(|t| t.archive.is_some()),
            lossy_copy: b.take.as_ref().and_then(|t| t.archive.as_ref()).is_some_and(|n| !n.exact),
        }
    }

    /// Under the facts of a raw take on screen, under a heading of their
    /// own: the two things written of the take itself ([`TakeJob`]), each
    /// behind its button. A run is of the take on screen; the ticked takes
    /// have a start button of their own beside it ([`take_buttons`]).
    pub(super) fn draw_take_tools(&mut self, ui: &mut egui::Ui) {
        if !self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video()) {
            return;
        }
        let views = |clips: Vec<usize>| -> Vec<TakeView> { clips.into_iter().map(|i| self.take_view(i)).collect() };
        let (archive, archive_ticked) = take_buttons(&self.batch, self.active_clip, TakeJob::Archive(None));
        let (originals, originals_ticked) = take_buttons(&self.batch, self.active_clip, TakeJob::Originals);
        let (archive, archive_ticked, originals, originals_ticked) = (views(archive), views(archive_ticked), views(originals), views(originals_ticked));
        let copies: Vec<TakeCopy> = self.batch.iter().filter_map(|b| b.copy.clone()).collect();
        let failed: Vec<(TakeJob, String, String)> = self.batch.iter()
            .filter_map(|b| b.take_failed.as_ref().map(|(job, why)| (*job, clip_label(&b.path), why.clone())))
            .collect();
        let out_dir = self.batch_out_dir.as_deref().map(clip_label);
        let view = TakeToolsView {
            archive: &archive, originals: &originals, archive_ticked: &archive_ticked, originals_ticked: &originals_ticked,
            copies: &copies, failed: &failed, out_dir: out_dir.as_deref(),
            idle: self.export_job.is_none() && !self.batch_running,
        };
        match Self::take_tools_ui(ui, &mut self.take_tools, &view) {
            Some(TakeAsk::Start(run, of)) => self.start_take_run(run, of),
            Some(TakeAsk::Keep) => self.save_take_copy(),
            None => {}
        }
    }

    /// The take tools themselves: their heading, a button each, and under
    /// the one that is open its choices, and then for the take on screen,
    /// and for the ticked takes where there are any besides it: what a run
    /// would write of which take and what it would leave out, where that
    /// goes, and the button that starts that run, which says whose takes
    /// it is of. Nothing here is an export choice: a run is started by its
    /// button, for that run. Returns what was asked for.
    pub(super) fn take_tools_ui(ui: &mut egui::Ui, tools: &mut TakeTools, view: &TakeToolsView) -> Option<TakeAsk> {
        let zh = cur_lang() == Lang::Zh;
        let note = |ui: &mut egui::Ui, text: &str| { ui.label(RichText::new(text).small().color(Color32::GRAY)); };
        let warn = |ui: &mut egui::Ui, text: &str| { ui.label(RichText::new(text).small().color(Color32::from_rgb(255, 200, 120))); };
        let mut ask = None;
        ui.add_space(6.0);
        ui.label(RichText::new(tr(TAKE_TOOLS_TITLE)).strong());
        ui.horizontal_wrapped(|ui| {
            for (tool, label, hover) in [
                (TakeTool::Archive, TAKE_ARCHIVE_LABEL, TAKE_ARCHIVE_HOVER),
                (TakeTool::Originals, TAKE_ORIGINALS_LABEL, TAKE_ORIGINALS_NOTE),
            ] {
                let mut open = tools.open == Some(tool);
                if ui.toggle_value(&mut open, tr(label)).on_hover_text(tr(hover)).changed() {
                    tools.open = open.then_some(tool);
                }
            }
        });
        // The choice between In to Out and the whole take, where a trim
        // makes them differ for a take of either start button.
        let range = |ui: &mut egui::Ui, whole: &mut bool, sets: [&[TakeView]; 2]| {
            if sets.iter().any(|takes| takes.iter().any(|t| t.written(false).2)) {
                ui.label(RichText::new(tr("Range")).strong());
                ui.radio_value(whole, false, tr("In to Out"));
                ui.radio_value(whole, true, tr("Whole take"));
            }
        };
        // Over the takes of each start button, where both are there: whose they are.
        let heading = |ui: &mut egui::Ui, of: TakeSet| {
            ui.add_space(4.0);
            ui.label(RichText::new(tr(match of { TakeSet::OnScreen => TAKE_ON_SCREEN_TITLE, TakeSet::Ticked => TAKE_TICKED_TITLE })).strong());
        };
        // The takes of a run, one by one, with what is written of each.
        let listed = |ui: &mut egui::Ui, whole: bool, takes: &[TakeView]| {
            for take in takes.iter().take(TAKES_LISTED) {
                ui.label(RichText::new(take_line(take, whole, zh)).small());
            }
            if takes.len() > TAKES_LISTED {
                let more = takes.len() - TAKES_LISTED;
                ui.label(RichText::new(if zh { format!("另有 {more} 条") } else { format!("and {more} more") }).small());
            }
        };
        // The takes the last run of this kind could not write, and why.
        let failed = |ui: &mut egui::Ui, archive: bool| {
            for (_, name, why) in view.failed.iter().filter(|(job, _, _)| matches!(job, TakeJob::Archive(_)) == archive) {
                ui.label(RichText::new(format!("✗ {name}: {}", take_failed_text(why, zh))).small().color(Color32::from_rgb(230, 110, 110)));
            }
        };
        let start = |ui: &mut egui::Ui, job: TakeJob, of: TakeSet, n: usize| -> bool {
            ui.add_enabled(view.idle, egui::Button::new(RichText::new(take_start_label(job, of, n, zh)).strong()).wrap())
                .on_disabled_hover_text(tr(TAKE_BUSY_HOVER))
                .clicked()
        };
        match tools.open {
            Some(TakeTool::Archive) => {
                ui.add_space(2.0);
                let sets = [(TakeSet::OnScreen, view.archive), (TakeSet::Ticked, view.archive_ticked)];
                // (No copy of the clip on screen: it is a copy itself, or a take whose description did not read.)
                if view.archive.is_empty() && view.originals.iter().any(|t| t.copy) {
                    note(ui, tr(TAKE_COPY_SOURCE_NOTE));
                }
                if sets.iter().any(|(_, takes)| !takes.is_empty()) {
                    let before = (tools.size, tools.bits);
                    ui.radio_value(&mut tools.size, ArchiveSize::Lossless, tr(TAKE_EXACT_LABEL));
                    ui.radio_value(&mut tools.size, ArchiveSize::HighQuality, tr("High quality: 1 bit a pixel"));
                    ui.radio_value(&mut tools.size, ArchiveSize::Custom, tr("Custom"));
                    if tools.size == ArchiveSize::Custom {
                        ui.add(egui::Slider::new(&mut tools.bits, 0.5..=1.0).step_by(0.05).fixed_decimals(2).suffix(format!(" {}", tr("bit a pixel"))));
                    }
                    if before != (tools.size, tools.bits) {
                        ask = Some(TakeAsk::Keep);
                    }
                    note(ui, tr(TAKE_ARCHIVE_NOTE));
                    note(ui, tr(if tools.archive_bits().is_none() { TAKE_EXACT_NOTE } else { TAKE_LOSSY_NOTE }));
                    let all: Vec<TakeView> = view.archive.iter().chain(view.archive_ticked).cloned().collect();
                    if let Some(text) = take_corners_note(&all, zh) {
                        note(ui, &text);
                    }
                    ui.add_space(4.0);
                    range(ui, &mut tools.archive_whole, [view.archive, view.archive_ticked]);
                    let (both, bits) = (sets.iter().all(|(_, takes)| !takes.is_empty()), tools.archive_bits());
                    // (What a copy of a stretch is: said once.)
                    let mut stretch_said = false;
                    for (of, takes) in sets {
                        if takes.is_empty() {
                            continue;
                        }
                        if both {
                            heading(ui, of);
                        }
                        let run = tools.archive_run(takes);
                        listed(ui, run.whole, takes);
                        if !run.whole {
                            if let Some(text) = take_cut_warning(takes, zh) {
                                warn(ui, &text);
                                if !std::mem::replace(&mut stretch_said, true) {
                                    note(ui, tr(TAKE_STRETCH_NOTE));
                                }
                            }
                        }
                        note(ui, &take_sizes_text(takes, run.whole, bits, zh));
                        note(ui, &take_place_text(takes, run.job, view.out_dir, zh));
                        ui.add_space(2.0);
                        if start(ui, run.job, of, takes.len()) {
                            ask = Some(TakeAsk::Start(run, of));
                        }
                    }
                }
                failed(ui, true);
                // The copies made, and what may be done with their takes now: by the user, never by the app.
                if !view.copies.is_empty() {
                    ui.add_space(4.0);
                    for copy in view.copies {
                        ui.label(RichText::new(take_copy_line(copy, zh)).small().color(Color32::from_rgb(120, 200, 120)));
                    }
                    note(ui, tr(TAKE_COPIED_NOTE));
                }
                ui.add_space(4.0);
            }
            Some(TakeTool::Originals) if !view.originals.is_empty() || !view.originals_ticked.is_empty() => {
                ui.add_space(2.0);
                note(ui, tr(TAKE_ORIGINALS_NOTE));
                let sets = [(TakeSet::OnScreen, view.originals, TAKE_LOSSY_SOURCE_NOTE), (TakeSet::Ticked, view.originals_ticked, TAKE_LOSSY_TICKED_NOTE)];
                for (_, takes, lossy) in sets {
                    if takes.iter().any(|t| t.lossy_copy) {
                        warn(ui, tr(lossy));
                    }
                }
                ui.add_space(4.0);
                range(ui, &mut tools.originals_whole, [view.originals, view.originals_ticked]);
                let both = sets.iter().all(|(_, takes, _)| !takes.is_empty());
                // (What else the folder beside a take is for: said once.)
                let mut place_said = view.out_dir.is_some();
                for (of, takes, _) in sets {
                    if takes.is_empty() {
                        continue;
                    }
                    if both {
                        heading(ui, of);
                    }
                    let run = tools.originals_run(takes);
                    listed(ui, run.whole, takes);
                    let frames: u64 = takes.iter().map(|t| { let (from, to, _) = t.written(run.whole); ((to - from) * t.fps as f64).round() as u64 }).sum();
                    note(ui, &if zh { format!("共 {frames} 帧") } else { format!("{frames} frames") });
                    note(ui, &take_place_text(takes, run.job, view.out_dir, zh));
                    if !std::mem::replace(&mut place_said, true) {
                        note(ui, tr(TAKE_ORIGINALS_PLACE_NOTE));
                    }
                    ui.add_space(2.0);
                    if start(ui, run.job, of, takes.len()) {
                        ask = Some(TakeAsk::Start(run, of));
                    }
                }
                failed(ui, false);
                ui.add_space(4.0);
            }
            _ => {}
        }
        ask
    }

    /// List entry `idx` reads the file it is to read ([`source_choice`]):
    /// its take first, an archive copy of the take only while the take is
    /// not at hand. Where that changes the file, the entry's trim moves
    /// with it (a copy of a stretch begins where the stretch did), the list
    /// says so, and a clip on screen is loaded again. Nothing on disk is
    /// touched. Returns whether the file changed.
    pub(super) fn settle_source(&mut self, idx: usize) -> bool {
        let Some(item) = self.batch.get(idx).filter(|b| b.ready() && b.status != BatchStatus::Running) else { return false };
        let (to, note) = match source_choice(&item.path, item.stand_in.as_ref(), self.batch_out_dir.as_deref()) {
            SourceChoice::Same => return false,
            SourceChoice::Take(take) => (take, None),
            SourceChoice::Copy(dir, note) => (dir, Some(note)),
        };
        let Some(facts) = ClipFacts::read(&to) else {
            tracing::warn!("list: {} does not open: {} is read on", to.display(), item.path.display());
            return false;
        };
        // (The clip on screen is edited in `self.settings`.)
        self.sync_active_clip_settings();
        let on_screen = self.active_clip == Some(idx);
        let zh = cur_lang() == Lang::Zh;
        let item = &mut self.batch[idx];
        let trim = (item.settings.trim_in_s, item.settings.trim_out_s);
        let (trim, stand_in, notice) = match (note, item.stand_in.take()) {
            (Some(note), _) => {
                let stand_in = StandIn::of(item.path.clone(), CopyExtent::of(&note, facts.duration_sec), trim, facts.fps);
                (stand_in.trim_on_copy, Some(stand_in), stand_in_notice(&item.path, &to, true, zh))
            }
            (None, Some(was)) => (was.trim_in_take(trim, item.fps), None, stand_in_notice(&to, &item.path, false, zh)),
            (None, None) => return false,
        };
        tracing::info!("list: {} is read in the place of {}", to.display(), item.path.display());
        (item.settings.trim_in_s, item.settings.trim_out_s) = trim;
        item.take_facts(to.clone(), facts);
        item.stand_in = stand_in;
        item.copy = None;
        item.take_failed = None;
        self.import_notice = Some(notice);
        if on_screen {
            self.settings = self.batch[idx].settings.clone();
            self.suppress_reseed_once = true;
            self.load_file_inner(to, true);
        }
        true
    }
}
