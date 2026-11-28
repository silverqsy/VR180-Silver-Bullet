use super::{join_mismatch_text, joined_output_for, tr_zh, unused_path, App, ExportCodec, ExportOptions, ExportResolution,
    ProResProfile, JOIN_HOVER, PROJECT_HOVER};
use crate::decoder::{FisheyeOutputMode, Settings};
use std::path::{Path, PathBuf};
use vr180_pipeline::join::{mismatch, planned_part, planned_part_where, Mismatch, PartSource, PlannedPart, Unjoinable};
use vr180_pipeline::SourceKind;

/// What a plan knows of a clip of `kind` whose eye is `eye` pixels a
/// side, read from the capture.
fn source_of(kind: SourceKind, eye: u32, one_eye: bool) -> PartSource {
    PartSource { one_eye, eye: super::native_eye_dims(kind, eye, eye, eye, eye), enhanced: false }
}

/// The part a clip would be in a joined run: `kind`, `eye` pixels a
/// side, at `fps`.
fn part_of(kind: SourceKind, eye: u32, fps: f32, settings: &Settings, opts: &ExportOptions, one_eye: bool) -> Result<PlannedPart, Unjoinable> {
    let cfg = App::build_export_cfg(
        Path::new("/nonexistent/in.insv"), Path::new(""), fps, kind,
        eye, eye, eye, eye, None, settings, opts, Vec::new(),
    );
    planned_part(&cfg, source_of(kind, eye, one_eye))
}

#[test]
fn clips_of_one_shape_join_and_others_say_why() {
    let (s, opts) = (Settings::default(), ExportOptions::default());
    let osv = part_of(SourceKind::DjiOsv, 3840, 29.97, &s, &opts, false).unwrap();
    // Two cameras at the 8K size, and a frame rate stated a hair apart.
    let x6 = part_of(SourceKind::Insta360Insv, 3712, 29.97, &s, &opts, false).unwrap();
    assert_eq!(osv.frame, (8192, 4096));
    assert_eq!(mismatch(&osv, &x6), None);
    let raw = part_of(SourceKind::X6Raw, 3840, 29.969, &s, &opts, false).unwrap();
    assert_eq!(mismatch(&osv, &raw), None);
    // Each at its own size: two sizes.
    let native = ExportOptions { resolution: ExportResolution::Native, ..opts };
    let (a, b) = (part_of(SourceKind::DjiOsv, 3840, 29.97, &s, &native, false).unwrap(),
                  part_of(SourceKind::Insta360Insv, 3712, 29.97, &s, &native, false).unwrap());
    assert_eq!(mismatch(&a, &b), Some(Mismatch::Frame((7680, 3840), (7424, 3712))));
    // Another frame rate.
    let fast = part_of(SourceKind::DjiOsv, 3840, 59.94, &s, &opts, false).unwrap();
    assert!(matches!(mismatch(&osv, &fast), Some(Mismatch::FrameRate(..))));
    // 8-bit H.265 beside a raw clip, which is always written at 10 bits.
    let eight = ExportOptions { h265_bit_depth: 8, ..opts };
    let (a, b) = (part_of(SourceKind::DjiOsv, 3840, 29.97, &s, &eight, false).unwrap(),
                  part_of(SourceKind::X6Raw, 3840, 29.97, &s, &eight, false).unwrap());
    assert_eq!(mismatch(&a, &b), Some(Mismatch::Encoding));
    // One eye beside two: another frame.
    let mono = part_of(SourceKind::Insta360Insv, 3840, 29.97, &s.for_single_eye(), &opts, true).unwrap();
    assert_eq!(mono.frame, (4096, 4096));
    assert!(matches!(mismatch(&osv, &mono), Some(Mismatch::Frame(..))));
    // ProRes: the profile is the encoding.
    let prores = ExportOptions { codec: ExportCodec::ProRes, ..opts };
    let hq = ExportOptions { prores_profile: ProResProfile::Hq, ..prores };
    let (a, b) = (part_of(SourceKind::DjiOsv, 3840, 29.97, &s, &prores, false).unwrap(),
                  part_of(SourceKind::DjiOsv, 3840, 29.97, &s, &hq, false).unwrap());
    assert_eq!(mismatch(&a, &a.clone()), None);
    assert_eq!(mismatch(&a, &b), Some(Mismatch::Encoding));
}

#[test]
fn views_that_do_not_share_a_file() {
    let opts = ExportOptions::default();
    let vr180 = Settings::default();
    let disc = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, lens_space: crate::decoder::LensSpaceLens::Equidistant, ..Default::default() };
    let a = part_of(SourceKind::DjiOsv, 3840, 29.97, &vr180, &opts, false).unwrap();
    let b = part_of(SourceKind::DjiOsv, 3840, 29.97, &disc, &opts, false).unwrap();
    assert_eq!(a.frame, b.frame);
    assert_eq!(mismatch(&a, &b), Some(Mismatch::View));
    // A disc of a fisheye camera and one of a `.360` state different
    // lenses (97.5° and 92.5° at the edge).
    let eac = part_of(SourceKind::GoProEac, 0, 29.97, &disc, &opts, false);
    if let Ok(eac) = eac {
        if eac.frame == b.frame {
            assert_eq!(mismatch(&b, &eac), Some(Mismatch::View));
        }
    }
    // Through the camera's own lens the file states that clip's lens:
    // such clips share an Apple Immersive Video file where one can be
    // written, and no file elsewhere.
    let lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, lens_space: crate::decoder::LensSpaceLens::Camera, ..Default::default() };
    let framed_cfg = |kind: SourceKind, s: &Settings| App::build_export_cfg(
        Path::new("/nonexistent/in"), Path::new(""), 29.97, kind, 3840, 3840, 3840, 3840, None, s, &opts, Vec::new());
    let osv = framed_cfg(SourceKind::DjiOsv, &lens);
    // Where such a file can be written: a part of its own kind.
    let framed = planned_part_where(&osv, source_of(SourceKind::DjiOsv, 3840, false), true).unwrap();
    assert!(framed.immersive() && !a.immersive() && !b.immersive());
    assert_eq!(mismatch(&framed, &framed.clone()), None);
    // Beside a disc or a half-equirect it is another view (or another frame).
    for other in [&a, &b] {
        assert!(mismatch(&framed, other).is_some());
    }
    // A clip of another camera in the same view: one file, whatever its lens.
    let x6 = planned_part_where(&framed_cfg(SourceKind::Insta360Insv, &lens), source_of(SourceKind::Insta360Insv, 3840, false), true).unwrap();
    if x6.frame == framed.frame {
        assert_eq!(mismatch(&framed, &x6), None);
    }
    // Nowhere else, and never with one eye.
    assert_eq!(planned_part_where(&osv, source_of(SourceKind::DjiOsv, 3840, false), false).unwrap_err(), Unjoinable::LensFrame);
    assert_eq!(planned_part_where(&framed_cfg(SourceKind::Insta360Insv, &lens.for_single_eye()), source_of(SourceKind::Insta360Insv, 3840, true), true).unwrap_err(), Unjoinable::LensFrame);
    // And this system says the one or the other.
    assert_eq!(part_of(SourceKind::DjiOsv, 3840, 29.97, &lens, &opts, false).is_ok(), vr180_pipeline::join::immersive_available());
}

/// A clip of the list, as a joined run's plan reads it.
fn clip(path: &Path, kind: SourceKind, eye: u32, fps: f32, settings: &Settings) -> super::BatchItem {
    super::BatchItem {
        path: path.into(), fps, duration_sec: 10.0, source_kind: kind,
        fisheye_eye_w: eye, fisheye_eye_h: eye, width: eye, height: eye,
        settings: settings.clone(), status: super::BatchStatus::Idle, selected: false, x6: None, raw_one_lens: false, raw_frame: None,
        four_channel_sound: false, take: None, upside_down: None, offline: None, before_take: None, copy: None, take_failed: None, stand_in: None,
    }
}

/// The clips' exports as a joined run plans them (`App::export_cfg_for`).
fn exports_of(batch: &[super::BatchItem], opts: &ExportOptions) -> Vec<vr180_pipeline::fisheye_export::FisheyeExportConfig> {
    batch.iter().map(|item| App::build_export_cfg(
        &item.path, Path::new(""), item.fps, item.source_kind,
        item.fisheye_eye_w, item.fisheye_eye_h, item.width, item.height, item.sharp_eye(),
        &item.export_settings(), opts, Vec::new())).collect()
}

/// A part is planned at the frame its export writes: a lens frame with
/// the stabilizer's margin, a raw take read from enhanced frames at the
/// size picked for those. Two clips that only look alike before that
/// are told apart at the start of the run.
#[test]
fn a_part_is_planned_at_the_frame_it_is_written_at() {
    use ExportResolution::{Native, R13k, R16k};
    let native = ExportOptions { resolution: Native, ..Default::default() };
    let lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, lens_space: crate::decoder::LensSpaceLens::Camera, ..Default::default() };
    let still = Settings { stabilize: false, ..lens.clone() };
    assert!(lens.stabilize);
    let framed = |kind: SourceKind, eye: u32, s: &Settings, opts: &ExportOptions| planned_part_where(
        &App::build_export_cfg(Path::new("/nonexistent/in"), Path::new(""), 29.97, kind, eye, eye, eye, eye, None, s, opts, Vec::new()),
        source_of(kind, eye, false), true).unwrap();
    // A stabilized lens frame carries its margin, one that is not does
    // not: the two do not share a file, and the plan says so.
    let (moved, fixed) = (framed(SourceKind::DjiOsv, 3840, &lens, &native), framed(SourceKind::DjiOsv, 3840, &still, &native));
    assert_eq!((moved.frame, fixed.frame), ((8448, 4224), (7680, 3840)));
    assert_eq!(mismatch(&moved, &fixed), Some(Mismatch::Frame((8448, 4224), (7680, 3840))));
    assert_eq!(mismatch(&moved, &framed(SourceKind::Insta360Insv, 3840, &lens, &native)), None);
    // A reduced copy keeps its size either way.
    let small = ExportOptions::default();
    assert_eq!(framed(SourceKind::Insta360Insv, 4864, &lens, &small).frame, framed(SourceKind::Insta360Insv, 4864, &still, &small).frame);
    assert_eq!(framed(SourceKind::Insta360Insv, 4864, &lens, &small).frame, (8192, 4096));
    // A `.360` clip's lens frame is its cross, with the margin too.
    let gopro = |s: &Settings| planned_part_where(
        &App::build_export_cfg(Path::new("/nonexistent/in.360"), Path::new(""), 29.97, SourceKind::GoProEac, 0, 0, 5952, 1920, None, s, &native, Vec::new()),
        PartSource { one_eye: false, eye: super::native_eye_dims(SourceKind::GoProEac, 0, 0, 5952, 1920), enhanced: false }, true).unwrap();
    let cross = vr180_pipeline::lens_frame::stabilized_dims(3936, 3936);
    assert_eq!((gopro(&lens).frame, gopro(&still).frame), ((cross.0 * 2, cross.1), (7872, 3936)));

    // A raw take with its folder of camera-original DNGs beside it: the
    // size picked for enhanced frames where the part reads them, the
    // size picked without where it does not.
    let root = std::env::temp_dir().join(format!("vr180-join-enhanced-size-{}", std::process::id()));
    let take = root.join("TAKE");
    std::fs::create_dir_all(root.join("TAKE_OriginalDNG/CAM0")).unwrap();
    let raw = |s: &Settings, opts: &ExportOptions, enhanced: bool| planned_part_where(
        &App::build_export_cfg(&take, Path::new(""), 59.94, SourceKind::X6Raw, 3840, 3840, 3840, 3840, None, s, opts, Vec::new()),
        PartSource { enhanced, ..source_of(SourceKind::X6Raw, 3840, false) }, true).unwrap();
    let vr = Settings::default();
    let big = ExportOptions { enhanced_resolution: R16k, ..Default::default() };
    assert_eq!((raw(&vr, &big, false).frame, raw(&vr, &big, true).frame), ((8192, 4096), (16384, 8192)));
    assert_eq!(raw(&vr, &ExportOptions { enhanced_resolution: R13k, ..big }, true).frame, (13120, 6560));
    // A lens frame: the size picked for enhanced frames is the frame's,
    // stabilized or not; from the capture it is the take's own frame.
    let big_native = ExportOptions { resolution: Native, ..big };
    assert_eq!((raw(&lens, &big_native, true).frame, raw(&still, &big_native, true).frame), ((16384, 8192), (16384, 8192)));
    assert_eq!((raw(&lens, &big_native, false).frame, raw(&still, &big_native, false).frame), ((8448, 4224), (7680, 3840)));
    // A clip that is no raw take has no such frames to read.
    let osv = App::build_export_cfg(Path::new("/nonexistent/in.osv"), Path::new(""), 29.97, SourceKind::DjiOsv, 3840, 3840, 3840, 3840, None, &vr, &big, Vec::new());
    assert_eq!(planned_part_where(&osv, PartSource { enhanced: true, ..source_of(SourceKind::DjiOsv, 3840, false) }, true).unwrap().frame, (8192, 4096));
    // And a take without the folder, or with the option off, neither.
    let alone = App::build_export_cfg(&root.join("OTHER"), Path::new(""), 59.94, SourceKind::X6Raw, 3840, 3840, 3840, 3840, None, &vr, &big, Vec::new());
    assert_eq!(planned_part_where(&alone, PartSource { enhanced: true, ..source_of(SourceKind::X6Raw, 3840, false) }, true).unwrap().frame, (8192, 4096));
    assert_eq!(raw(&vr, &ExportOptions { raw_use_enhanced: false, ..big }, true).frame, (8192, 4096));
    let _ = std::fs::remove_dir_all(&root);
}

/// The parts of a joined file read their takes' enhanced frames all or
/// none: one clip without them leaves them out of every part, and the
/// run says so when some clips have them.
#[test]
fn a_joined_file_reads_enhanced_frames_in_every_part_or_in_none() {
    use super::{join_check_outcome, join_enhanced_note};
    use ExportResolution::{Native, R16k};
    // Every clip has some: they are read. One has none: the capture,
    // and that clip is named. No clip has any: nothing to say.
    assert_eq!(join_check_outcome(&[Some(true), Some(true), Some(true)], 3), (true, None));
    assert_eq!(join_check_outcome(&[Some(true), Some(false), Some(true)], 3), (false, Some(1)));
    assert_eq!(join_check_outcome(&[Some(false), Some(true)], 2), (false, Some(0)));
    assert_eq!(join_check_outcome(&[Some(false), Some(false)], 2), (false, None));
    // A clip that could read none (no raw take, no folder) counts as one without.
    assert_eq!(join_check_outcome(&[Some(true), None], 2), (false, Some(1)));
    assert_eq!(join_check_outcome(&[None, Some(false)], 2), (false, None));
    // A look that was stopped has no answer for the run: the capture.
    assert_eq!(join_check_outcome(&[], 2), (false, None));
    assert_eq!(join_check_outcome(&[Some(true)], 2), (false, None));
    assert!(vr180_pipeline::join::parts_read_enhanced(&[true, true]) && !vr180_pipeline::join::parts_read_enhanced(&[true, false]));
    assert!(!vr180_pipeline::join::parts_read_enhanced(&[]));
    let (en, zh) = (join_enhanced_note("B.insv", false), join_enhanced_note("B.insv", true));
    assert!(en.contains("B.insv") && zh.contains("B.insv") && en != zh, "{en} / {zh}");

    // What a run settles before it starts.
    let root = std::env::temp_dir().join(format!("vr180-join-enhanced-plan-{}", std::process::id()));
    for take in ["A", "B", "C"] {
        std::fs::create_dir_all(root.join(format!("{take}_OriginalDNG/CAM0"))).unwrap();
    }
    let vr = Settings::default();
    let raw = |name: &str, eye: u32, fps: f32| clip(&root.join(name), SourceKind::X6Raw, eye, fps, &vr);
    let plan = |batch: &[super::BatchItem], opts: &ExportOptions| {
        let run: Vec<usize> = (0..batch.len()).collect();
        App::join_enhanced_plan(batch, &run, exports_of(batch, opts))
    };
    let big = ExportOptions { enhanced_resolution: R16k, ..Default::default() };
    // Two takes that could both read enhanced frames: their folders are
    // looked through, and the parts agree either way.
    let two = [raw("A", 3840, 59.94), raw("B", 3840, 59.94)];
    let (immersive, enhanced, check) = plan(&two, &big).unwrap();
    let check = check.expect("a look");
    assert_eq!((immersive, enhanced), (false, None));
    assert!(check.exports.iter().all(|e| e.is_some()) && check.exports.len() == 2);
    assert_eq!((check.agree_plain.clone(), check.agree_enhanced.clone()), (Ok(()), Ok(())));
    // A clip of another camera beside them: read from the capture
    // whatever the folders hold, which are looked through for the note.
    let mixed = [raw("A", 3840, 29.97), clip(Path::new("/nonexistent/in.osv"), SourceKind::DjiOsv, 3840, 29.97, &vr)];
    let (_, enhanced, check) = plan(&mixed, &big).unwrap();
    let check = check.expect("a look");
    assert_eq!(enhanced, None);
    assert_eq!(check.exports.iter().map(|e| e.is_some()).collect::<Vec<_>>(), vec![true, false]);
    assert_eq!((check.agree_plain.clone(), check.agree_enhanced.clone()), (Ok(()), Ok(())));
    // No take with a folder: nothing to look for.
    let plain = [raw("X", 3840, 59.94), raw("Y", 3840, 59.94)];
    let (_, enhanced, check) = plan(&plain, &big).unwrap();
    assert!(enhanced == Some(false) && check.is_none());
    // The option off: the same.
    let (_, enhanced, check) = plan(&two, &ExportOptions { raw_use_enhanced: false, ..big }).unwrap();
    assert!(enhanced == Some(false) && check.is_none());

    // Takes of two frame sizes, each at its own size without enhanced
    // frames and at 16384 × 8192 with: one file only from enhanced
    // frames, which the look decides.
    let sizes = [raw("A", 3840, 59.94), raw("C", 5760, 59.94)];
    let own = ExportOptions { resolution: Native, ..big };
    let (_, enhanced, check) = plan(&sizes, &own).unwrap();
    let check = check.expect("a look");
    assert_eq!(enhanced, None);
    assert_eq!(check.agree_enhanced, Ok(()));
    let why = check.agree_plain.clone().expect_err("two sizes from the capture");
    assert!(why.contains("7680 × 3840") && why.contains("11520 × 5760") && why.contains('A') && why.contains('C'), "{why}");
    // And the other way round: one size without, each its own with.
    let (_, _, check) = plan(&sizes, &ExportOptions { enhanced_resolution: Native, ..Default::default() }).unwrap();
    let check = check.expect("a look");
    assert_eq!(check.agree_plain, Ok(()));
    assert!(check.agree_enhanced.clone().expect_err("two sizes from enhanced frames").contains("11520 × 5760"));
    // Clips that cannot be one file either way are refused at once.
    let rates = [raw("A", 3840, 59.94), raw("B", 3840, 29.97)];
    let why = plan(&rates, &big).err().expect("two frame rates");
    assert!(why.contains("59.94") && why.contains("29.97"), "{why}");
    // So is a mixed run whose clips do not agree from the capture.
    let odd = [raw("A", 3840, 29.97), clip(Path::new("/nonexistent/in.insv"), SourceKind::Insta360Insv, 3712, 29.97, &vr)];
    assert!(plan(&odd, &own).is_err());
    // Two Lens Space clips, one stabilized and one not, where such
    // clips can be joined at all: refused at once, with what mends it.
    if vr180_pipeline::join::immersive_available() {
        let lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, lens_space: crate::decoder::LensSpaceLens::Camera, ..Default::default() };
        let fixed = Settings { stabilize: false, ..lens.clone() };
        let osv = |name: &str, s: &Settings| clip(Path::new(name), SourceKind::DjiOsv, 3840, 29.97, s);
        let why = plan(&[osv("/nonexistent/a.osv", &lens), osv("/nonexistent/b.osv", &fixed)], &own).err().expect("two frames");
        assert!(why.contains("8448 × 4224") && why.contains("7680 × 3840") && why.contains(super::JOIN_LENS_MARGIN_HINT), "{why}");
        assert!(plan(&[osv("/nonexistent/a.osv", &lens), osv("/nonexistent/b.osv", &lens)], &own).is_ok());
        // Clips of two sizes are another matter: no word of stabilization.
        let why = plan(&[osv("/nonexistent/a.osv", &lens), clip(Path::new("/nonexistent/c.insv"), SourceKind::Insta360Insv, 3712, 29.97, &lens)], &own).err().expect("two frames");
        assert!(!why.contains(super::JOIN_LENS_MARGIN_HINT), "{why}");
    }
    // The look: a part that could read none is not asked, and a look
    // that is stopped has no answer.
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (_, _, check) = plan(&mixed, &big).unwrap();
    // (The folders here hold no frame: no enhanced one either.)
    assert_eq!(super::join_look(&check.unwrap().exports, &stop), Some(vec![Some(false), None]));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let (_, _, check) = plan(&two, &big).unwrap();
    assert_eq!(super::join_look(&check.unwrap().exports, &stop), None);
    let _ = std::fs::remove_dir_all(&root);

    // On a real take (named by `VR180_ENHANCED_TAKE`, with
    // `VR180_ENHANCED_AT=<a>,<b>`: seconds at which a stretch of a fifth
    // of a second holds an enhanced frame, and holds none; skipped
    // without them): a clip whose stretch holds one and a clip whose
    // stretch does not are read from the capture, both, and the second
    // is named; two clips that both hold one are read from enhanced
    // frames.
    let Some(sample) = std::env::var_os("VR180_ENHANCED_TAKE").map(PathBuf::from) else { return };
    let at: Vec<f64> = std::env::var("VR180_ENHANCED_AT").unwrap_or_default().split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let (Some(&with), Some(&without)) = (at.first(), at.get(1)) else { return };
    let stretch = |t_in: f64, t_out: f64| clip(&sample, SourceKind::X6Raw, 3840, 29.97, &Settings { trim_in_s: Some(t_in), trim_out_s: Some(t_out), ..Default::default() });
    let stop = std::sync::atomic::AtomicBool::new(false);
    let outcome = |batch: &[super::BatchItem]| {
        let (_, enhanced, check) = plan(batch, &big).unwrap();
        assert_eq!(enhanced, None);
        let check = check.expect("a look");
        let found = super::join_look(&check.exports, &stop).expect("an answer");
        (found.clone(), join_check_outcome(&found, batch.len()), check.agree_plain, check.agree_enhanced)
    };
    let (found, decided, agree_plain, agree_enhanced) = outcome(&[stretch(with, with + 0.2), stretch(without, without + 0.2)]);
    assert_eq!((found, decided), (vec![Some(true), Some(false)], (false, Some(1))));
    assert_eq!((agree_plain, agree_enhanced), (Ok(()), Ok(())));
    let (found, decided, _, _) = outcome(&[stretch(with, with + 0.2), stretch(with + 0.05, with + 0.15)]);
    assert_eq!((found, decided), (vec![Some(true), Some(true)], (true, None)));
    // A part's export does as the run decided.
    let exports = exports_of(&[stretch(with, with + 0.2)], &big);
    assert!(exports[0].reads_enhanced_frames());
    let mut kept = exports[0].clone();
    assert!(kept.settle_raw_enhanced(kept.trim_in_s, kept.trim_out_s));
    assert_eq!((kept.eye_w, kept.eye_h), (8192, 8192));
    let mut left_out = exports[0].clone().without_enhanced_frames();
    assert!(!left_out.reads_enhanced_frames() && !left_out.settle_raw_enhanced(left_out.trim_in_s, left_out.trim_out_s));
    assert_eq!((left_out.eye_w, left_out.eye_h, left_out.raw_enhanced_dir.clone()), (4096, 4096, None));
}

#[test]
fn every_reason_is_said_in_both_languages() {
    for why in [Mismatch::Frame((8192, 4096), (7680, 3840)), Mismatch::FrameRate(29.97, 59.94), Mismatch::Encoding, Mismatch::View] {
        let en = join_mismatch_text("a.osv", "b.insv", why, false);
        let zh = join_mismatch_text("a.osv", "b.insv", why, true);
        assert!(en.contains("a.osv") && en.contains("b.insv") && zh.contains("a.osv") && zh.contains("b.insv"));
        assert_ne!(en, zh, "{why:?}");
    }
    for text in [
        JOIN_HOVER, PROJECT_HOVER, "Join into one file", "Join all", "Join selected", "Export the ticked clips as one file",
        "joining the clips…", "enhanced frames left out", "Looking for enhanced frames…", super::JOIN_LENS_MARGIN_HINT, "Not joined", "the clips exported so far are in", "was not exported", "no clip was exported",
        "stopped", "Drag to change the order", "is a still photo: a joined file is made of video clips.",
        "A DNG sequence is a folder of frames for each clip: it cannot be joined. Pick a video format in Format.",
        "A spatial video cannot be joined: export the clips one by one, or pick H.265 or ProRes in Format.",
        "is in the Lens Space view through the camera's lens. Clips in that view are joined as an Apple Immersive Video file, which needs macOS 26 or later and two eyes in every clip: here they cannot be joined. Pick the Equidistant lens, or another view.",
        "has no four-channel sound: a joined file with Ambisonic or APAC sound needs it in every clip. Set the first clip's Audio to Stereo, or leave this clip out.",
        "Project", "New project", "Open project…", "Save project", "Save project as…", "Save project…", "Don't save",
        "Replace the clip list?", "The clip list has changes that are not saved in a project file.", "looking…", "missing",
        "The file was not found where the project says it is. Click to look again.",
    ] {
        assert_ne!(tr_zh(text), text, "no Chinese text for: {text}");
    }
}

#[test]
fn the_joined_sound_is_the_first_clips_choice_where_every_clip_can_give_it() {
    use super::joined_sound;
    use crate::decoder::AudioFormat as AF;
    use vr180_pipeline::join::JoinSound;
    // Every clip has four-channel sound: what the first clip asks for.
    assert_eq!(joined_sound(AF::Stereo, &[true, true]), Ok(JoinSound::Stereo));
    assert_eq!(joined_sound(AF::Ambisonic, &[true, true, true]), Ok(JoinSound::Ambisonic));
    assert_eq!(joined_sound(AF::Apac, &[true, true]), Ok(JoinSound::Apac));
    // One clip has not: named by its place, before anything is exported.
    assert_eq!(joined_sound(AF::Apac, &[true, false, true]), Err(1));
    assert_eq!(joined_sound(AF::Ambisonic, &[true, true, false]), Err(2));
    // Stereo takes any clips; and a first clip without such sound has
    // no say, whatever its settings were left at.
    assert_eq!(joined_sound(AF::Stereo, &[false, true]), Ok(JoinSound::Stereo));
    assert_eq!(joined_sound(AF::Apac, &[false, true]), Ok(JoinSound::Stereo));
    assert_eq!(joined_sound(AF::Apac, &[]), Ok(JoinSound::Stereo));
}

#[test]
fn the_joined_file_is_named_after_the_first_clip() {
    let src = Path::new("/clips/VID_0042.insv");
    assert_eq!(joined_output_for(src, ExportCodec::H265, false, false), PathBuf::from("/clips/VID_0042_Joined_SBS.mp4"));
    assert_eq!(joined_output_for(src, ExportCodec::ProRes, false, false), PathBuf::from("/clips/VID_0042_Joined_SBS.mov"));
    assert_eq!(joined_output_for(src, ExportCodec::H265, true, false), PathBuf::from("/clips/VID_0042_Joined_Mono.mp4"));
    // A raw take is a folder: the file goes beside it.
    assert_eq!(joined_output_for(Path::new("/takes/RAWVIDEO-1"), ExportCodec::H265, false, false), PathBuf::from("/takes/RAWVIDEO-1_Joined_SBS.mp4"));
    // Lens frames: an Apple Immersive Video file, whatever the parts' codec.
    assert_eq!(joined_output_for(src, ExportCodec::H265, false, true), PathBuf::from("/clips/VID_0042_Joined_Lens.aivu"));
    assert_eq!(joined_output_for(src, ExportCodec::ProRes, false, true), PathBuf::from("/clips/VID_0042_Joined_Lens.aivu"));
}

#[test]
fn a_name_in_use_is_never_written_over() {
    let dir = std::env::temp_dir().join(format!("vr180_unused_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("a_SBS.mp4");
    assert_eq!(unused_path(out.clone()), out);
    std::fs::write(&out, b"x").unwrap();
    assert_eq!(unused_path(out.clone()), dir.join("a_SBS (2).mp4"));
    std::fs::write(dir.join("a_SBS (2).mp4"), b"x").unwrap();
    assert_eq!(unused_path(out.clone()), dir.join("a_SBS (3).mp4"));
    // A folder of parts keeps its extension too.
    let parts = dir.join("a_Joined_SBS.parts");
    std::fs::create_dir_all(&parts).unwrap();
    assert_eq!(unused_path(parts), dir.join("a_Joined_SBS (2).parts"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_export_choices_are_kept_with_a_project() {
    let opts = ExportOptions {
        codec: ExportCodec::ProRes, prores_profile: ProResProfile::Hq, resolution: ExportResolution::Native,
        h265_bitrate_mbps: 320, inject_apmp: false, inject_youtube: true, beyondvr_scale_pct: 72.5, ..Default::default()
    };
    let json = serde_json::to_string(&opts).unwrap();
    assert_eq!(serde_json::from_str::<ExportOptions>(&json).unwrap(), opts);
    // A file from before a choice existed: that choice is the default's.
    let old: ExportOptions = serde_json::from_str(r#"{ "codec": "ProRes", "h265_bitrate_mbps": 80 }"#).unwrap();
    assert_eq!(old, ExportOptions { codec: ExportCodec::ProRes, h265_bitrate_mbps: 80, ..Default::default() });
}
