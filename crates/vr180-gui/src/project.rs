//! A project: the clip list, every clip's own settings and the output
//! choices, as one JSON file.
//!
//! Two files are written in this form. The SESSION, next to
//! `settings.json`, is kept up to date as the user works and read back at
//! the next launch, so the list is there again. A PROJECT file is written
//! where the user says and opened when they ask. Neither holds any media:
//! a clip is named by where its file is, and (in a project file) by where
//! that is from the project file's own folder, so a project moved together
//! with its media still finds it. A clip whose file is not found stays in
//! the list, marked, with its settings.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::decoder::Settings;

/// The format's version. A file of a later one is refused: what it holds
/// may not mean what this build would take it to mean.
pub const VERSION: u32 = 1;

/// A project file's extension.
pub const EXTENSION: &str = "vr180proj";

/// A file a project names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileRef {
    /// Where it was when the project was written.
    pub path: PathBuf,
    /// The same from the project file's folder, `/` between the parts
    /// (`None` in the session, and where the two share no folder).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative: Option<String>,
}

impl FileRef {
    /// `path`, as a file written in `base` (the project file's folder)
    /// names it.
    pub fn new(path: &Path, base: Option<&Path>) -> Self {
        Self { path: path.to_path_buf(), relative: base.and_then(|b| relative_to(path, b)) }
    }

    /// Where the file is now: where it was, else where the relative path
    /// leads from `base`. `None`: at neither. Touches the disk.
    pub fn resolve(&self, base: Option<&Path>) -> Option<PathBuf> {
        if self.path.exists() {
            return Some(self.path.clone());
        }
        let moved = join_relative(base?, self.relative.as_deref()?);
        moved.exists().then_some(moved)
    }
}

/// `path` from the folder `base`, `/` between the parts: `None` unless both
/// are absolute and share a folder below the root (two volumes do not). On
/// Windows the volume is the drive or the network share the two begin with,
/// and being on one is enough: a disk that comes up under another letter
/// still holds a project and its clips at the same places.
pub fn relative_to(path: &Path, base: &Path) -> Option<String> {
    if !path.is_absolute() || !base.is_absolute() {
        return None;
    }
    let (p, b): (Vec<Component>, Vec<Component>) = (path.components().collect(), base.components().collect());
    let common = p.iter().zip(&b).take_while(|(x, y)| x == y).count();
    // The root alone is no shared folder; a drive or a share is one.
    if !p[..common].iter().any(|c| matches!(c, Component::Normal(_) | Component::Prefix(_))) {
        return None;
    }
    let mut parts: Vec<String> = vec!["..".to_string(); b.len() - common];
    for c in &p[common..] {
        match c {
            Component::Normal(name) => parts.push(name.to_str()?.to_string()),
            _ => return None,
        }
    }
    Some(parts.join("/"))
}

/// The file a relative path of [`relative_to`] names from `base`.
pub fn join_relative(base: &Path, relative: &str) -> PathBuf {
    let mut out = base.to_path_buf();
    for part in relative.split('/').filter(|p| !p.is_empty() && *p != ".") {
        if part == ".." {
            out.pop();
        } else {
            out.push(part);
        }
    }
    out
}

/// One clip of the list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectClip {
    pub file: FileRef,
    /// The files that make up the clip, in order, where the user chose them
    /// at import (a recording merged from several files, or one file kept
    /// apart from its chapters). `None`: the app looks for itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<FileRef>>,
    /// Ticked in the list.
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub settings: Settings,
}

/// The list and the output choices. `E` is the app's export options (they
/// live with the export bar; this module only carries them).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "E: Deserialize<'de> + Default"))]
pub struct Project<E> {
    pub version: u32,
    /// The build that wrote the file (for a reader of the file; not read).
    #[serde(default)]
    pub written_by: String,
    #[serde(default)]
    pub clips: Vec<ProjectClip>,
    /// The clip on screen, by its place in `clips`.
    #[serde(default)]
    pub active: Option<usize>,
    /// The clips of an export run are written as one file.
    #[serde(default)]
    pub join_clips: bool,
    /// Where exports go (`None`: next to each source).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_folder: Option<PathBuf>,
    #[serde(default)]
    pub export: E,
    /// In the session only: the project file the list belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_file: Option<PathBuf>,
}

impl<E: Default> Default for Project<E> {
    fn default() -> Self {
        Self {
            version: VERSION,
            written_by: String::new(),
            clips: Vec::new(),
            active: None,
            join_clips: false,
            output_folder: None,
            export: E::default(),
            project_file: None,
        }
    }
}

/// Read a project (or the session). The error is for the user.
pub fn load<E: for<'de> Deserialize<'de> + Default>(path: &Path) -> Result<Project<E>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text)
}

/// [`load`] from the file's text.
pub fn parse<E: for<'de> Deserialize<'de> + Default>(text: &str) -> Result<Project<E>, String> {
    // The version first, by itself: a later file's other fields may not
    // even parse here.
    #[derive(Deserialize)]
    struct Head {
        version: u32,
    }
    let head: Head = serde_json::from_str(text).map_err(|e| format!("not a project file ({e})"))?;
    if head.version > VERSION {
        return Err(format!(
            "this project was written by a later version of the app (format {}, this build reads up to {VERSION})",
            head.version));
    }
    serde_json::from_str(text).map_err(|e| format!("the project file could not be read ({e})"))
}

/// Write a project (or the session): to a file beside `path` first, moved
/// into place once it is whole, so that a crash never leaves half a file.
/// What is written is read back first: a file this build could not open
/// again (a value JSON has no form for) never takes the place of one it
/// can.
pub fn save<E: Serialize + for<'de> Deserialize<'de> + Default>(project: &Project<E>, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(project).map_err(|e| format!("the project could not be written ({e})"))?;
    parse::<E>(&json).map_err(|e| format!("the project was not written: it would not read back ({e})"))?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".writing");
    let scratch = path.with_file_name(name);
    std::fs::write(&scratch, json).map_err(|e| format!("{}: {e}", scratch.display()))?;
    std::fs::rename(&scratch, path).map_err(|e| {
        let _ = std::fs::remove_file(&scratch);
        format!("{}: {e}", path.display())
    })
}

/// The app's own folder: where `settings.json` is.
pub fn config_dir() -> Option<PathBuf> {
    Settings::config_path().and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// The session's file in the app's folder.
const SESSION_FILE: &str = "session.json";
/// A session that was not taken, kept for the user (one copy).
const SESSION_ASIDE: &str = "session.unreadable.json";
/// The mark of [`GuardStage`].
const GUARD_FILE: &str = "session.loading";

/// Where the session is kept.
pub fn session_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join(SESSION_FILE))
}

/// A mark left on disk (in the app's folder) while a launch does something
/// that the next launch would do again by itself, and taken away once that
/// has gone well, or the app closes in order. A launch that finds one knows
/// the last launch died on the way and does not walk into the same thing:
/// one bad session, or one bad file, must not stop the app from starting
/// ever again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardStage {
    /// Putting the session's list back. The next launch starts with an
    /// empty list and puts the session file aside.
    Restoring,
    /// Loading a clip that had shown no frame yet. The next launch puts
    /// the list back without a clip on screen.
    Loading,
}

impl GuardStage {
    fn text(self) -> &'static str {
        match self {
            GuardStage::Restoring => "restoring",
            GuardStage::Loading => "loading",
        }
    }

    /// Leave the mark (best effort: without it the app only loses the
    /// protection).
    pub fn leave(self) {
        if let Some(dir) = config_dir() {
            self.leave_in(&dir);
        }
    }

    pub fn leave_in(self, dir: &Path) {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(GUARD_FILE), self.text());
    }

    /// Take the mark away.
    pub fn clear() {
        if let Some(dir) = config_dir() {
            let _ = std::fs::remove_file(dir.join(GUARD_FILE));
        }
    }

    /// The mark the last launch left in `dir`, taken away as it is read.
    /// Anything that is not the first stage's own word counts as the
    /// second: the milder reading of a mark that cannot be told.
    pub fn take_in(dir: &Path) -> Option<Self> {
        let guard = dir.join(GUARD_FILE);
        let text = std::fs::read_to_string(&guard).ok()?;
        let _ = std::fs::remove_file(&guard);
        Some(if text.trim() == GuardStage::Restoring.text() { GuardStage::Restoring } else { GuardStage::Loading })
    }
}

/// The last session's list as a launch takes it from the app's folder
/// `dir`, and whether its clip stays off the screen (the last launch ended
/// while a clip was loading). A session the last launch died putting back,
/// or one that does not read, is put aside (never written over) and the
/// launch starts with an empty list.
pub fn take_session<E: for<'de> Deserialize<'de> + Default>(dir: &Path) -> (Option<Project<E>>, bool) {
    let died = GuardStage::take_in(dir);
    let died_loading = died == Some(GuardStage::Loading);
    if died_loading {
        tracing::warn!("session: the last launch ended while a clip was loading: the list comes back without one on screen");
    }
    let path = dir.join(SESSION_FILE);
    if !path.is_file() {
        return (None, died_loading);
    }
    let put_aside = |why: &str| {
        tracing::warn!("session: {why}: kept as {SESSION_ASIDE}, starting with an empty list");
        let _ = std::fs::rename(&path, dir.join(SESSION_ASIDE));
    };
    if died == Some(GuardStage::Restoring) {
        put_aside("the last launch ended while the list was put back");
        return (None, false);
    }
    match load::<E>(&path) {
        Ok(session) => {
            tracing::info!("session: {} clips from {}", session.clips.len(), path.display());
            (Some(session), died_loading)
        }
        Err(e) => {
            put_aside(&e);
            (None, died_loading)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    struct Options {
        size: u32,
        fast: bool,
    }

    impl Default for Options {
        fn default() -> Self {
            Self { size: 8, fast: true }
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vr180_project_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn a_file_is_named_from_the_project_folder() {
        let rel = |p: &str, b: &str| relative_to(Path::new(p), Path::new(b));
        // Beside the project, below it, and in a sibling folder.
        assert_eq!(rel("/Volumes/A/trip/a.insv", "/Volumes/A/trip").as_deref(), Some("a.insv"));
        assert_eq!(rel("/Volumes/A/trip/day 1/a.insv", "/Volumes/A/trip").as_deref(), Some("day 1/a.insv"));
        assert_eq!(rel("/Volumes/A/clips/a.insv", "/Volumes/A/trip/projects").as_deref(), Some("../../clips/a.insv"));
        // Nothing shared below the root: no relative name.
        assert_eq!(rel("/Users/x/a.insv", "/Volumes/A/trip"), None);
        assert_eq!(rel("a.insv", "/Volumes/A/trip"), None);
        // And back.
        assert_eq!(join_relative(Path::new("/mnt/B/trip/projects"), "../../clips/a.insv"), PathBuf::from("/mnt/B/clips/a.insv"));
        assert_eq!(join_relative(Path::new("/mnt/B/trip"), "day 1/a.insv"), PathBuf::from("/mnt/B/trip/day 1/a.insv"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn a_file_is_named_from_the_project_folder() {
        let rel = |p: &str, b: &str| relative_to(Path::new(p), Path::new(b));
        // Beside the project, below it, and in a sibling folder.
        assert_eq!(rel(r"E:\trip\a.insv", r"E:\trip").as_deref(), Some("a.insv"));
        assert_eq!(rel(r"E:\trip\day 1\a.insv", r"E:\trip").as_deref(), Some("day 1/a.insv"));
        assert_eq!(rel(r"E:\media\clips\a.insv", r"E:\media\trip\projects").as_deref(), Some("../../clips/a.insv"));
        // One drive, or one share, and nothing else shared: still a name.
        assert_eq!(rel(r"E:\clips\a.insv", r"E:\trip").as_deref(), Some("../clips/a.insv"));
        assert_eq!(rel(r"\\nas\media\clips\a.insv", r"\\nas\media\trip").as_deref(), Some("../clips/a.insv"));
        // Two drives, a drive and a share, two shares: no relative name.
        assert_eq!(rel(r"C:\Users\x\a.insv", r"E:\trip"), None);
        assert_eq!(rel(r"\\nas\media\a.insv", r"E:\trip"), None);
        assert_eq!(rel(r"\\nas\media\a.insv", r"\\nas\other\trip"), None);
        assert_eq!(rel("a.insv", r"E:\trip"), None);
        // And back, on another letter.
        assert_eq!(join_relative(Path::new(r"F:\trip"), "../clips/a.insv"), PathBuf::from(r"F:\clips\a.insv"));
        assert_eq!(join_relative(Path::new(r"F:\trip"), "day 1/a.insv"), PathBuf::from(r"F:\trip\day 1\a.insv"));
    }

    #[test]
    fn a_moved_project_finds_its_media() {
        let root = scratch("moved");
        let (old, new) = (root.join("old"), root.join("new"));
        std::fs::create_dir_all(old.join("clips")).unwrap();
        std::fs::write(old.join("clips/a.insv"), b"x").unwrap();
        let named = FileRef::new(&old.join("clips/a.insv"), Some(&old));
        assert_eq!(named.relative.as_deref(), Some("clips/a.insv"));
        // Where it was.
        assert_eq!(named.resolve(Some(&old)), Some(old.join("clips/a.insv")));
        // The whole folder moved: found from the project's new folder.
        std::fs::rename(&old, &new).unwrap();
        assert_eq!(named.resolve(Some(&new)), Some(new.join("clips/a.insv")));
        // Not there at all, or no folder to look from (the session).
        assert_eq!(named.resolve(Some(&root)), None);
        assert_eq!(named.resolve(None), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_project_comes_back_as_it_was_written() {
        let dir = scratch("roundtrip");
        let mut settings = Settings::default();
        settings.trim_in_s = Some(1.25);
        settings.trim_out_s = Some(7.5);
        settings.stabilize = !settings.stabilize;
        let project = Project {
            written_by: "test".into(),
            clips: vec![
                ProjectClip {
                    file: FileRef::new(&dir.join("a.insv"), Some(&dir)),
                    segments: Some(vec![FileRef::new(&dir.join("a.insv"), Some(&dir)), FileRef::new(&dir.join("b.insv"), Some(&dir))]),
                    selected: true,
                    settings,
                },
                ProjectClip { file: FileRef::new(&dir.join("c.osv"), None), segments: None, selected: false, settings: Settings::default() },
            ],
            active: Some(1),
            join_clips: true,
            output_folder: Some(dir.join("out")),
            export: Options { size: 12, fast: false },
            ..Default::default()
        };
        let file = dir.join(format!("trip.{EXTENSION}"));
        save(&project, &file).unwrap();
        assert_eq!(load::<Options>(&file).unwrap(), project);
        // Written whole: nothing of the writing is left beside it.
        let names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec![format!("trip.{EXTENSION}")]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_that_would_not_read_back_is_not_written() {
        let dir = scratch("readback");
        let file = dir.join(format!("trip.{EXTENSION}"));
        let good = Project::<Options> { written_by: "first".into(), ..Default::default() };
        save(&good, &file).unwrap();
        // A number JSON has no form for: it would be written as `null`.
        let mut settings = Settings::default();
        settings.smooth_ms = f32::NAN;
        let bad = Project::<Options> {
            written_by: "second".into(),
            clips: vec![ProjectClip { file: FileRef::new(&dir.join("a.insv"), None), segments: None, selected: false, settings }],
            ..Default::default()
        };
        let why = save(&bad, &file).unwrap_err();
        assert!(why.contains("would not read back"), "{why}");
        // The file that was there is still the one that reads.
        assert_eq!(load::<Options>(&file).unwrap(), good);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_launch_does_not_walk_into_what_ended_the_last_one() {
        let dir = scratch("guard");
        let session = Project::<Options> {
            clips: vec![ProjectClip { file: FileRef::new(&dir.join("a.insv"), None), segments: None, selected: false, settings: Settings::default() }],
            ..Default::default()
        };
        let written = || save(&session, &dir.join(SESSION_FILE)).unwrap();
        // No mark: the list, with its clip on screen.
        written();
        assert_eq!(take_session::<Options>(&dir), (Some(session.clone()), false));
        assert!(dir.join(SESSION_FILE).is_file());
        // Died loading a clip: the list, without a clip on screen. Once.
        GuardStage::Loading.leave_in(&dir);
        assert_eq!(take_session::<Options>(&dir), (Some(session.clone()), true));
        assert!(!dir.join(GUARD_FILE).exists());
        assert_eq!(take_session::<Options>(&dir), (Some(session.clone()), false));
        // A mark nobody can read is taken as that milder one.
        std::fs::write(dir.join(GUARD_FILE), b"").unwrap();
        assert_eq!(take_session::<Options>(&dir), (Some(session.clone()), true));
        // Died putting the list back: an empty list, the session kept aside.
        GuardStage::Restoring.leave_in(&dir);
        assert_eq!(take_session::<Options>(&dir), (None, false));
        assert!(!dir.join(SESSION_FILE).exists() && !dir.join(GUARD_FILE).exists());
        assert_eq!(load::<Options>(&dir.join(SESSION_ASIDE)).unwrap(), session);
        // And the launch after that starts as a first run does.
        assert_eq!(take_session::<Options>(&dir), (None, false));
        // A session that does not read is put aside too, not written over.
        std::fs::write(dir.join(SESSION_FILE), b"{ \"version\": 99 }").unwrap();
        assert_eq!(take_session::<Options>(&dir), (None, false));
        assert_eq!(std::fs::read(dir.join(SESSION_ASIDE)).unwrap(), b"{ \"version\": 99 }");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn older_and_later_files() {
        // A file with less in it than this build writes: the rest is the default.
        let old: Project<Options> = parse(r#"{ "version": 1, "clips": [ { "file": { "path": "/x/a.insv" } } ] }"#).unwrap();
        assert_eq!(old.clips.len(), 1);
        assert_eq!(old.clips[0].settings, Settings::default());
        assert_eq!((old.join_clips, old.active, old.export), (false, None, Options::default()));
        // Options this build does not know are passed over; a known one is kept.
        let extra: Project<Options> = parse(r#"{ "version": 1, "export": { "size": 3, "later_thing": 4 }, "later": true }"#).unwrap();
        assert_eq!(extra.export, Options { size: 3, fast: true });
        // A later format is refused, in words.
        let later = parse::<Options>(r#"{ "version": 2, "clips": "something else" }"#).unwrap_err();
        assert!(later.contains("later version"), "{later}");
        assert!(parse::<Options>("{}").unwrap_err().contains("not a project file"));
        assert!(parse::<Options>("not json").unwrap_err().contains("not a project file"));
    }
}
