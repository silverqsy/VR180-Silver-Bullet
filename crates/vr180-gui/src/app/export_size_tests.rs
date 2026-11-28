use super::{export_runs_noise_filter, App, EnhancedFrames, ExportOptions, ExportResolution, SizeList, X6ClipInfo};
use super::{oversize_export_hint, tr_zh, ExportCodec};

/// The note for an oversize video names the encoder that runs on this
/// platform: VideoToolbox takes the frame in hardware on a Mac, NVENC
/// refuses anything above 8192 a side, and Windows' GPU ProRes encoder
/// has no such limit.
#[test]
fn oversize_hint_follows_the_platform_encoder() {
    for codec in [ExportCodec::H265, ExportCodec::ProRes] {
        assert_eq!(oversize_export_hint(codec, 8192, 4096), None);
        assert_eq!(oversize_export_hint(codec, 7680, 3840), None);
    }
    // A Crop clip's native-sharpness canvas and a 12K BRAW at its own size.
    for (w, h) in [(13120, 6560), (12288, 6144)] {
        let h265 = oversize_export_hint(ExportCodec::H265, w, h).expect("a note above 8192 wide");
        let prores = oversize_export_hint(ExportCodec::ProRes, w, h);
        if cfg!(target_os = "macos") {
            assert!(h265.contains("encodes in hardware"), "{h265}");
            assert!(prores.is_some_and(|s| s.contains("software encoder")));
        } else {
            assert!(h265.contains("NVENC") && h265.contains("libx265"), "{h265}");
            assert_eq!(prores, None);
        }
        for note in [Some(h265), prores].into_iter().flatten() {
            assert_ne!(tr_zh(note), note, "no Chinese text for: {note}");
        }
    }
    // Taller than Level 6.2 allows but within every hardware encoder's frame.
    let tall = oversize_export_hint(ExportCodec::H265, 8192, 4320).expect("a note above 4096 lines");
    assert!(tall.contains("Level 6.2") && tall.contains("encodes in hardware"), "{tall}");
}
use vr180_fisheye::{InsvWindowCrop, X6ViewChoice};

/// Per-eye size of an export under `opts`: a clip of `eye` pixels a
/// side whose native-sharpness side is `sharp_eye` (a Crop clip's).
fn eye_dims_with(opts: &ExportOptions, eye: u32, sharp_eye: Option<u32>) -> (u32, u32) {
    let cfg = App::build_export_cfg(
        std::path::Path::new("/nonexistent/in.insv"), std::path::Path::new("/nonexistent/out.mp4"),
        29.97, vr180_pipeline::SourceKind::Insta360Insv,
        eye, eye, eye, eye, sharp_eye,
        &crate::decoder::Settings::default(), opts, Vec::new(),
    );
    (cfg.eye_w, cfg.eye_h)
}

/// The same with one choice made on both sorts of clip.
fn eye_dims(resolution: ExportResolution, eye: u32, sharp_eye: Option<u32>) -> (u32, u32) {
    let opts = ExportOptions { resolution, crop_resolution: resolution, ..Default::default() };
    eye_dims_with(&opts, eye, sharp_eye)
}

#[test]
fn native_sharpness_size_is_for_crop_clips() {
    // A 10K Crop clip: its own size, 8K, or the canvas at its centre density.
    assert_eq!(eye_dims(ExportResolution::Native, 4912, Some(6560)), (4912, 4912));
    assert_eq!(eye_dims(ExportResolution::R8k, 4912, Some(6560)), (4096, 4096));
    assert_eq!(eye_dims(ExportResolution::Sharp, 4912, Some(6560)), (6560, 6560));
    // Any other clip has no such size: the choice means 8K there.
    assert_eq!(eye_dims(ExportResolution::Sharp, 3840, None), (4096, 4096));
    assert_eq!(eye_dims(ExportResolution::Native, 3840, None), (3840, 3840));
}

/// One list, both sorts of clip: what is picked on a full-view clip
/// leaves a Crop clip's size alone, and the other way round.
#[test]
fn a_crop_clip_has_its_own_resolution_choice() {
    use ExportResolution::{Native, R8k, Sharp};
    let full = (7680, None);
    let crop = (4912, Some(6560));
    for (general, crop_choice, full_side, crop_side) in [
        (Native, Sharp, 7680, 6560),
        (Native, R8k, 7680, 4096),
        (R8k, Sharp, 4096, 6560),
        (R8k, Native, 4096, 4912),
        (Native, Native, 7680, 4912),
    ] {
        let opts = ExportOptions { resolution: general, crop_resolution: crop_choice, ..Default::default() };
        assert_eq!(opts.resolution_for(SizeList::Plain), general);
        assert_eq!(opts.resolution_for(SizeList::Crop), crop_choice);
        assert_eq!(eye_dims_with(&opts, full.0, full.1), (full_side, full_side), "{general:?} / {crop_choice:?}");
        assert_eq!(eye_dims_with(&opts, crop.0, crop.1), (crop_side, crop_side), "{general:?} / {crop_choice:?}");
    }
    // Both start at 8K.
    let d = ExportOptions::default();
    assert_eq!((d.resolution, d.crop_resolution), (R8k, R8k));
}

/// Every piece of text the Format window paints for a VR180 clip.
fn panel_text(opts: &mut ExportOptions, native: (u32, u32), sharp: Option<(u32, u32)>, single_eye: bool) -> String {
    panel_text_with(opts, native, sharp, None, single_eye)
}

/// The same, for a clip in the Lens Space view (`lens_space`: the lens
/// that view writes for it).
fn panel_text_with(opts: &mut ExportOptions, native: (u32, u32), sharp: Option<(u32, u32)>, lens_space: Option<crate::decoder::LensSpaceLens>, single_eye: bool) -> String {
    panel_text_lens(opts, native, sharp, lens_space, false, single_eye)
}

/// The same, saying whether that clip is stabilized.
fn panel_text_lens(opts: &mut ExportOptions, native: (u32, u32), sharp: Option<(u32, u32)>, lens_space: Option<crate::decoder::LensSpaceLens>, stabilized: bool, single_eye: bool) -> String {
    panel_text_full(opts, native, sharp, lens_space, stabilized, single_eye, None)
}

/// The same, for a raw video clip whose enhanced frames are `raw`
/// (`None`: no raw video).
fn panel_text_full(opts: &mut ExportOptions, native: (u32, u32), sharp: Option<(u32, u32)>, lens_space: Option<crate::decoder::LensSpaceLens>, stabilized: bool, single_eye: bool, raw: Option<EnhancedFrames>) -> String {
    fn collect(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(text) => { out.push_str(text.galley.text()); out.push('\n'); }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let mut text = String::new();
    // Two frames: a new context lays its first one out without sizes to go by.
    for _ in 0..2 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 3000.0))),
            ..Default::default()
        };
        let out = ctx.run_ui(input, |ui| {
            App::export_options_ui(ui, opts, "test", None, (90.0, 0.0), 29.97, false, Some(native), sharp, lens_space, stabilized, single_eye, raw);
        });
        text.clear();
        out.shapes.iter().for_each(|clipped| collect(&clipped.shape, &mut text));
    }
    text
}

/// A raw take read from enhanced frames has two larger canvases and a
/// choice of its own; without enhanced frames, and on every other clip,
/// those sizes do not exist.
#[test]
fn enhanced_frames_have_their_own_sizes() {
    use ExportResolution::{Native, R13k, R16k, R8k, Sharp};
    let opts = ExportOptions { resolution: Native, crop_resolution: Sharp, enhanced_resolution: R16k, ..Default::default() };
    assert_eq!((opts.resolution_for(SizeList::Plain), opts.resolution_for(SizeList::Crop), opts.resolution_for(SizeList::Enhanced)), (Native, Sharp, R16k));
    // A size from another list means 8192 × 4096.
    let odd = ExportOptions { resolution: R16k, crop_resolution: R13k, enhanced_resolution: Sharp, ..Default::default() };
    assert_eq!((odd.resolution_for(SizeList::Plain), odd.resolution_for(SizeList::Crop), odd.resolution_for(SizeList::Enhanced)), (R8k, R8k, R8k));
    assert_eq!(ExportOptions::default().enhanced_resolution, R8k);
    assert_eq!((SizeList::of(false, false), SizeList::of(true, false), SizeList::of(false, true)), (SizeList::Plain, SizeList::Crop, SizeList::Enhanced));
    // The canvases, per eye.
    let dims = |r| ExportOptions { enhanced_resolution: r, ..Default::default() }.enhanced_eye_dims((3840, 3840));
    assert_eq!((dims(Native), dims(R8k), dims(R13k), dims(R16k)), ((3840, 3840), (4096, 4096), (6560, 6560), (8192, 8192)));
    assert_eq!((R13k.frame_label(false), R16k.frame_label(false), R16k.frame_label(true)), ("13120 × 6560", "16384 × 8192", "8192 × 8192"));

    // The Format window of a raw take: with enhanced frames found it shows
    // their choice and says so; without, the usual one.
    let mut opts = ExportOptions { resolution: Native, enhanced_resolution: R16k, ..Default::default() };
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("16384 × 8192") && !text.contains("7680 × 3840 (native)"), "{text}");
    assert!(text.contains("Enhanced frames found: 7680 × 7680 per eye"), "{text}");
    assert!(text.contains("the two larger sizes keep it"), "{text}");
    // The note for a frame this size is the platform encoder's own.
    let note = oversize_export_hint(opts.codec, 16384, 8192).expect("a note at this size");
    assert!(text.contains(note), "{text}");
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Absent));
    assert!(text.contains("7680 × 3840 (native)") && !text.contains("16384"), "{text}");
    assert!(text.contains("No enhanced frame found for this clip"), "{text}");
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Looking));
    assert!(text.contains("Looking for enhanced frames") && !text.contains("16384"), "{text}");
    // A build that cannot read enhanced frames says so, and offers neither
    // the option nor the sizes.
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Unreadable));
    assert!(text.contains(super::ENHANCED_UNREADABLE_NOTE) && !text.contains("Use enhanced frames") && !text.contains("16384"), "{text}");
    assert!(text.contains("7680 × 3840 (native)"), "{text}");
    assert_ne!(tr_zh(super::ENHANCED_UNREADABLE_NOTE), super::ENHANCED_UNREADABLE_NOTE);
    // The option off: the usual list, and no word of what the folder holds.
    opts.raw_use_enhanced = false;
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("7680 × 3840 (native)") && !text.contains("16384") && !text.contains("Enhanced frames found"), "{text}");
    // Showing either list changed neither choice.
    assert_eq!((opts.resolution, opts.enhanced_resolution), (Native, R16k));
    // Another clip never sees those sizes.
    let text = panel_text(&mut opts, (7680, 3840), None, false);
    assert!(!text.contains("16384") && !text.contains("enhanced"), "{text}");
}

/// The Format window of a raw take offers its view: a video, or the
/// developed view as a folder of DNG frames. The take itself (an archive
/// copy, the camera's own frames) is no export, and is not there.
#[test]
fn the_format_window_of_a_raw_take_offers_its_view_only() {
    use ExportResolution::Native;
    let mut opts = ExportOptions { resolution: Native, ..Default::default() };
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Absent));
    assert!(text.contains("DNG sequence") && text.contains("Every frame as a linear DNG in a folder") && text.contains("Codec"), "{text}");
    // (The folder of the camera's own frames is still named where enhanced frames are explained.)
    let offers = |text: &str, choice: &str| text.lines().any(|line| line == choice);
    for gone in ["Archive copy", "Compressed archive", "Camera original", "Camera original DNGs", "Developed view"] {
        assert!(!offers(&text, gone), "{gone}: {text}");
    }
    assert!(!text.contains("bit a pixel") && !text.contains("sensor's own"), "{text}");
    // The folder of frames is a picture of a size: its list is there, the codec is not.
    opts.raw_dng_sequence = true;
    let text = panel_text_full(&mut opts, (7680, 3840), None, None, false, false, Some(EnhancedFrames::Absent));
    assert!(text.contains("7680 × 3840 (native)") && text.contains("Use enhanced frames") && !text.contains("Codec"), "{text}");
    assert!(!offers(&text, "Archive copy") && !offers(&text, "Camera original") && !offers(&text, "Developed view"), "{text}");
    assert_eq!(opts.resolution, Native);

    // The export bar's "NR" is for exports that do run the noise filter.
    let raw = vr180_pipeline::SourceKind::X6Raw;
    let video = ExportOptions::default();
    assert!(export_runs_noise_filter(raw, 0.71, &video));
    assert!(!export_runs_noise_filter(raw, 0.0, &video));
    assert!(!export_runs_noise_filter(raw, 0.71, &opts));
    // Another clip in the same list is a video whatever the raw option says.
    assert!(export_runs_noise_filter(vr180_pipeline::SourceKind::Insta360Insv, 0.5, &opts));
}

/// Export choices kept by a version in which the take itself was one of
/// them: the camera's own frames, picked there, do not come back as an
/// export of the developed view, and nothing of them is written again.
#[test]
fn choices_kept_from_before_the_take_left_the_format_window() {
    let picked: ExportOptions = serde_json::from_str(r#"{"raw_dng_sequence":true,"raw_dng_original":true}"#).unwrap();
    assert_eq!(picked.settled(), ExportOptions::default());
    let developed: ExportOptions = serde_json::from_str(r#"{"raw_dng_sequence":true,"raw_dng_original":false}"#).unwrap();
    assert_eq!(developed.settled(), ExportOptions { raw_dng_sequence: true, ..Default::default() });
    let plain: ExportOptions = serde_json::from_str(r#"{"raw_dng_sequence":true}"#).unwrap();
    assert!(plain.settled().raw_dng_sequence);
    // An archive copy was a choice there for a while: a file that holds it exports the view.
    let copy: ExportOptions = serde_json::from_str(r#"{"raw_archive":true,"raw_archive_size":"Lossless","raw_archive_bits":0.6}"#).unwrap();
    assert_eq!(copy.settled(), ExportOptions::default());
    let written = serde_json::to_string(&ExportOptions { raw_dng_sequence: true, ..Default::default() }).unwrap();
    assert!(written.contains("raw_dng_sequence") && !written.contains("raw_dng_original") && !written.contains("archive"), "{written}");
    assert_eq!(App::default_archive_dir_for(std::path::Path::new("/takes/TAKE")), std::path::PathBuf::from("/takes/TAKE_Archive"));
}

/// The export of a raw take carries both sizes: the one picked without
/// enhanced frames, and the one picked for them, which it takes where
/// its stretch has some.
#[test]
fn a_raw_export_carries_the_enhanced_size() {
    use crate::decoder::{FisheyeOutputMode, Settings};
    use ExportResolution::{Native, R13k, R16k};
    let root = std::env::temp_dir().join(format!("vr180-enhanced-size-{}", std::process::id()));
    let take = root.join("TAKE");
    std::fs::create_dir_all(root.join("TAKE_OriginalDNG/CAM0")).unwrap();
    let cfg = |opts: &ExportOptions, settings: &Settings, source: &std::path::Path| App::build_export_cfg(
        source, std::path::Path::new("/nonexistent/out.mov"), 50.0, vr180_pipeline::SourceKind::X6Raw,
        3840, 3840, 3840, 3840, None, settings, opts, Vec::new());
    let vr = Settings::default();
    let big = ExportOptions { enhanced_resolution: R16k, ..Default::default() };
    let c = cfg(&big, &vr, &take);
    assert_eq!(((c.eye_w, c.eye_h), c.raw_enhanced_eye), ((4096, 4096), Some((8192, 8192))));
    assert_eq!(c.raw_enhanced_dir.as_deref(), Some(root.join("TAKE_OriginalDNG").as_path()));
    let c = cfg(&ExportOptions { enhanced_resolution: R13k, resolution: Native, ..big }, &vr, &take);
    assert_eq!(((c.eye_w, c.eye_h), c.raw_enhanced_eye), ((3840, 3840), Some((6560, 6560))));
    let c = cfg(&ExportOptions { enhanced_resolution: Native, ..big }, &vr, &take);
    assert_eq!(c.raw_enhanced_eye, Some((3840, 3840)));
    // The option off, or no folder beside the take: the plain export.
    let c = cfg(&ExportOptions { raw_use_enhanced: false, ..big }, &vr, &take);
    assert_eq!((c.raw_enhanced_dir.clone(), c.raw_enhanced_eye), (None, None));
    let c = cfg(&big, &vr, &root.join("OTHER"));
    assert_eq!((c.raw_enhanced_dir.clone(), c.raw_enhanced_eye), (None, None));
    // A reframed view keeps its own size: enhanced frames, no other canvas.
    let reframed = Settings { fisheye_output_mode: FisheyeOutputMode::Reframe, ..Settings::default() };
    let c = cfg(&big, &reframed, &take);
    assert!(c.raw_enhanced_dir.is_some() && c.raw_enhanced_eye.is_none());
    let _ = std::fs::remove_dir_all(&root);

    // The sample take with one enhanced frame (skipped when it is not
    // there): a stretch that holds it is written large, another is not.
    let sample = std::path::Path::new("/Volumes/Database/x6/1115 raw video/RAWVIDEO-20261116-011723");
    let enhanced = sample.with_file_name("RAWVIDEO-20261116-011723_OriginalDNG").join("CAM0/000000364.dng");
    if !cfg!(target_os = "macos") || !sample.is_dir() || std::fs::metadata(&enhanced).map(|m| m.len() < 30_000_000).unwrap_or(true) { return; }
    let mut c = cfg(&big, &vr, sample);
    assert!(c.settle_raw_enhanced(Some(4.9), Some(5.1)));
    assert_eq!(((c.eye_w, c.eye_h), c.raw_enhanced_eye, c.raw_enhanced_dir.is_some()), ((8192, 8192), None, true));
    let mut c = cfg(&big, &vr, sample);
    assert!(!c.settle_raw_enhanced(Some(8.0), Some(9.0)));
    assert_eq!(((c.eye_w, c.eye_h), c.raw_enhanced_eye, c.raw_enhanced_dir.is_some()), ((4096, 4096), None, false));
}

/// The Lens Space view of a raw take has the two sizes of its enhanced
/// frames as the VR180 view has them: the same list, the same choice,
/// the frame written at exactly the size picked.
#[test]
fn a_lens_frame_has_the_sizes_of_the_enhanced_frames() {
    use crate::decoder::{FisheyeOutputMode, LensSpaceLens::Camera, Settings};
    use vr180_pipeline::fisheye_export::LensFrameOut;
    use ExportResolution::{Native, R13k, R16k, R8k};
    // The frame at a size: the take's own shape, its longer side on it.
    assert_eq!((super::lens_frame_sized_dims(3840, 3840, 6560), super::lens_frame_sized_dims(3840, 3840, 8192)), ((6560, 6560), (8192, 8192)));
    assert_eq!((super::lens_frame_sized_dims(3840, 2880, 8192), super::lens_frame_sized_dims(2880, 3840, 6560)), ((8192, 6144), (4920, 6560)));
    // Written at exactly that, stabilized or not: the VR180 view's two
    // sizes. The take's own size carries the stabilizer's margin as
    // always, and the 8K choice is what it is without enhanced frames.
    let written = |r, stabilized| ExportOptions { enhanced_resolution: r, ..Default::default() }
        .lens_frame_out_dims(3840, 3840, SizeList::Enhanced, stabilized, false);
    for stabilized in [false, true] {
        assert_eq!((written(R13k, stabilized), written(R16k, stabilized)), ((6560, 6560), (8192, 8192)));
        assert_eq!(written(R8k, stabilized), ExportOptions::default().lens_frame_out_dims(3840, 3840, SizeList::Plain, stabilized, false));
    }
    assert_eq!((written(Native, false), written(Native, true)), ((3840, 3840), (4224, 4224)));
    assert_eq!((R13k.frame_label(false), R16k.frame_label(false)), ("13120 × 6560", "16384 × 8192"));
    // A clip of another list never gets those sizes.
    let odd = ExportOptions { resolution: R16k, crop_resolution: R13k, ..Default::default() };
    assert_eq!(odd.lens_frame_eye_dims(3840, 3840, SizeList::Plain, false, false), (3840, 3840));
    assert_eq!(odd.lens_frame_eye_dims(4912, 4912, SizeList::Crop, false, false), (4096, 4096));

    // The export: the frame as recorded where the stretch has no enhanced
    // frame, and the size picked for them where it has.
    let root = std::env::temp_dir().join(format!("vr180-lens-enhanced-{}", std::process::id()));
    let take = root.join("TAKE");
    std::fs::create_dir_all(root.join("TAKE_OriginalDNG/CAM0")).unwrap();
    let lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, ..Settings::default() };
    assert!(lens.lens_frame_view(vr180_pipeline::SourceKind::X6Raw));
    let cfg = |opts: &ExportOptions, source: &std::path::Path| App::build_export_cfg(
        source, std::path::Path::new("/nonexistent/out.mov"), 59.94, vr180_pipeline::SourceKind::X6Raw,
        3840, 3840, 3840, 3840, None, &lens, opts, Vec::new());
    let big = ExportOptions { resolution: Native, enhanced_resolution: R16k, ..Default::default() };
    let c = cfg(&big, &take);
    assert_eq!((c.lens_frame, (c.eye_w, c.eye_h), c.raw_enhanced_eye), (LensFrameOut::FromSource, (3840, 3840), Some((8192, 8192))));
    assert_eq!(c.raw_enhanced_dir.as_deref(), Some(root.join("TAKE_OriginalDNG").as_path()));
    assert_eq!(cfg(&ExportOptions { enhanced_resolution: R13k, ..big }, &take).raw_enhanced_eye, Some((6560, 6560)));
    assert_eq!(cfg(&ExportOptions { enhanced_resolution: Native, ..big }, &take).raw_enhanced_eye, Some((3840, 3840)));
    // The option off, or no folder beside the take: the plain lens frame.
    let c = cfg(&ExportOptions { raw_use_enhanced: false, ..big }, &take);
    assert_eq!((c.raw_enhanced_dir.clone(), c.raw_enhanced_eye, (c.eye_w, c.eye_h)), (None, None, (3840, 3840)));
    let c = cfg(&big, &root.join("OTHER"));
    assert_eq!((c.raw_enhanced_dir.clone(), c.raw_enhanced_eye), (None, None));
    let _ = std::fs::remove_dir_all(&root);

    // The Format window in that view: the choice of the enhanced frames,
    // their two sizes, and what a frame of such a size is.
    let mut opts = big;
    for stabilized in [false, true] {
        let text = panel_text_full(&mut opts, (7680, 3840), None, Some(Camera), stabilized, false, Some(EnhancedFrames::Found(7680)));
        assert!(text.contains("\n16384 × 8192\n") && !text.contains("(native"), "{text}");
        assert!(text.contains(super::LENS_FRAME_NOTE) && text.contains(super::LENS_FRAME_ENHANCED_NOTE), "{text}");
        assert!(text.contains("Enhanced frames found: 7680 × 7680 per eye"), "{text}");
        let note = oversize_export_hint(opts.codec, 16384, 8192).expect("a note at this size");
        assert!(text.contains(note), "{text}");
    }
    opts.enhanced_resolution = R13k;
    let text = panel_text_full(&mut opts, (7680, 3840), None, Some(Camera), true, false, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("\n13120 × 6560\n"), "{text}");
    // One eye in the frame.
    let text = panel_text_full(&mut opts, (3840, 3840), None, Some(Camera), true, true, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("\n6560 × 6560\n"), "{text}");
    // The take's own size from that list: the frame with its margin.
    opts.enhanced_resolution = Native;
    let text = panel_text_full(&mut opts, (7680, 3840), None, Some(Camera), true, false, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("8448 × 4224 (native + margin)") && text.contains(super::LENS_FRAME_ENHANCED_NOTE), "{text}");
    // No enhanced frame for the stretch, or the option off: the frame as
    // recorded, by the choice every other clip has, and no word of them.
    opts.enhanced_resolution = R16k;
    let text = panel_text_full(&mut opts, (7680, 3840), None, Some(Camera), false, false, Some(EnhancedFrames::Absent));
    assert!(text.contains("7680 × 3840 (native)") && !text.contains("16384") && !text.contains(super::LENS_FRAME_ENHANCED_NOTE), "{text}");
    opts.raw_use_enhanced = false;
    let text = panel_text_full(&mut opts, (7680, 3840), None, Some(Camera), false, false, Some(EnhancedFrames::Found(7680)));
    assert!(text.contains("7680 × 3840 (native)") && !text.contains("16384") && !text.contains(super::LENS_FRAME_ENHANCED_NOTE), "{text}");
    // Showing either list changed neither choice.
    assert_eq!((opts.resolution, opts.enhanced_resolution), (Native, R16k));
    assert_ne!(tr_zh(super::LENS_FRAME_ENHANCED_NOTE), super::LENS_FRAME_ENHANCED_NOTE);
}

/// The Format window shows each sort of clip its own choice, and
/// showing one never changes the other's.
#[test]
fn the_format_window_keeps_both_choices() {
    use ExportResolution::{Native, Sharp};
    let mut opts = ExportOptions { resolution: Native, crop_resolution: Sharp, ..Default::default() };
    // A clip that holds one whole eye, at its own size.
    let text = panel_text(&mut opts, (7680, 7680), None, true);
    assert!(text.contains("7680 × 7680 (native)"), "{text}");
    assert!(!text.contains("native sharpness"), "{text}");
    assert_eq!((opts.resolution, opts.crop_resolution), (Native, Sharp));
    // A 10K Crop clip in the same list.
    let text = panel_text(&mut opts, (9824, 4912), Some((13120, 6560)), false);
    assert!(text.contains("13120 × 6560 (native sharpness)"), "{text}");
    assert!(!text.contains("9824 × 4912 (native)"), "{text}");
    assert_eq!((opts.resolution, opts.crop_resolution), (Native, Sharp));
}

/// The export poses its frames as the preview does: the stabilization's
/// parameters, the view adjustment and the eye swap reach the
/// configuration whole, under every way the settings say them.
#[test]
fn the_export_poses_its_frames_as_the_preview_does() {
    use crate::decoder::{FisheyeOutputMode, Settings};
    use vr180_pipeline::SourceKind;
    let cfg = |settings: &Settings| App::build_export_cfg(
        std::path::Path::new("/nonexistent/in.insv"), std::path::Path::new("/nonexistent/out.mp4"),
        29.97, SourceKind::Insta360Insv, 3840, 3840, 3840, 3840, None, settings, &ExportOptions::default(), Vec::new(),
    );
    let cases = [
        Settings::default(),
        Settings { stabilize: true, dji_camera_lock: true, ..Default::default() },
        Settings { stabilize: true, dji_camera_lock: false, dji_smooth_ms: 0.0, dji_max_corr_deg: 0.0, ..Default::default() },
        Settings { stabilize: true, dji_camera_lock: false, dji_smooth_ms: 800.0, dji_max_corr_deg: 12.0, dji_responsiveness: 1.7, ..Default::default() },
        Settings { fisheye_swap_eyes: true, ..Default::default() },
        Settings { camera_upside_down: true, ..Default::default() },
        Settings { fisheye_swap_eyes: true, camera_upside_down: true, pano_yaw_deg: 3.0, stereo_yaw_deg: -0.4, ..Default::default() },
        Settings {
            fisheye_output_mode: FisheyeOutputMode::Reframe, reframe_yaw_deg: 20.0, reframe_pitch_deg: -8.0, reframe_roll_deg: 2.0,
            stabilize: true, ..Default::default()
        },
    ];
    for settings in &cases {
        assert_eq!(cfg(settings).motion(), settings.motion(), "{settings:?}");
    }
    // The cases do differ: a lock has no window and no cap, a soft
    // stabilization keeps its window off the lock's value.
    assert_eq!((cases[1].motion().max_corr_deg, cases[1].motion().smooth_ms), (f32::INFINITY, 0.0));
    assert_eq!((cases[2].motion().max_corr_deg, cases[2].motion().smooth_ms), (f32::INFINITY, 1.0));
    assert_eq!((cases[3].motion().max_corr_deg, cases[3].motion().smooth_ms, cases[3].motion().responsiveness), (12.0, 800.0, 1.7));
    assert!(cases[4].motion().swap_eyes && cases[5].motion().swap_eyes && !cases[6].motion().swap_eyes);
    assert_ne!(cases[7].motion().view_adjust, cases[0].motion().view_adjust);
}

/// The Lens Space view through the camera's lens exports a lens frame,
/// at the size and shape the eye was recorded; its equidistant lens and
/// every other view export as they always did.
#[test]
fn a_lens_frame_is_the_recorded_frame() {
    use crate::decoder::{FisheyeOutputMode, LensSpaceLens, Settings};
    use vr180_pipeline::fisheye_export::{FisheyeExportProjection, LensFrameOut};
    use vr180_pipeline::SourceKind;
    let cfg = |opts: &ExportOptions, settings: &Settings, kind, eye: (u32, u32), sharp_eye| App::build_export_cfg(
        std::path::Path::new("/nonexistent/in.insv"), std::path::Path::new("/nonexistent/out.mp4"),
        29.97, kind, eye.0, eye.1, eye.0, eye.1, sharp_eye, settings, opts, Vec::new(),
    );
    let vr = Settings::default();
    // The Lens Space view; the camera's lens is its default.
    let lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, ..Default::default() };
    assert_eq!(lens.lens_space, LensSpaceLens::Camera);
    let insv = SourceKind::Insta360Insv;
    let on = ExportOptions { crop_resolution: ExportResolution::Sharp, ..Default::default() };

    // A square Crop frame: its own size, whatever the VR180 canvas would be.
    let c = cfg(&on, &lens, insv, (4912, 4912), Some(6560));
    assert_eq!((c.projection, c.lens_frame), (FisheyeExportProjection::Fisheye, LensFrameOut::FromSource));
    assert_eq!((c.eye_w, c.eye_h), (4912, 4912));
    // A frame that is not square keeps its shape.
    for eye in [(6000, 3968), (3840, 3008)] {
        let c = cfg(&on, &lens, insv, eye, Some(6560));
        assert_eq!((c.eye_w, c.eye_h, c.lens_frame), (eye.0, eye.1, LensFrameOut::FromSource), "{eye:?}");
    }
    // The clip's own size and its native sharpness are the same frame here.
    let native = ExportOptions { crop_resolution: ExportResolution::Native, ..on };
    assert_eq!((cfg(&native, &lens, insv, (4912, 4912), Some(6560)).eye_w), 4912);
    // The 8K choice is a reduced copy of the same shape, never an enlargement.
    let small = ExportOptions { crop_resolution: ExportResolution::R8k, ..on };
    let c = cfg(&small, &lens, insv, (4912, 4912), Some(6560));
    assert_eq!((c.eye_w, c.eye_h), (4096, 4096));
    let c = cfg(&small, &lens, insv, (6000, 3968), Some(6560));
    assert_eq!((c.eye_w, c.eye_h), (4096, 2708));
    let c = cfg(&small, &lens, insv, (3712, 3712), Some(6560));
    assert_eq!((c.eye_w, c.eye_h), (3712, 3712));
    assert_eq!(super::lens_frame_reduced_dims(3840, 3008), None);
    // What is written: the stabilizer's margin on a frame at its own
    // size, none on a reduced copy.
    assert_eq!(native.lens_frame_out_dims(4912, 4912, SizeList::Crop, true, false), (5400, 5400));
    assert_eq!(native.lens_frame_out_dims(4912, 4912, SizeList::Crop, false, false), (4912, 4912));
    assert_eq!(small.lens_frame_out_dims(4912, 4912, SizeList::Crop, true, false), (4096, 4096));
    assert_eq!(small.lens_frame_out_dims(3712, 3712, SizeList::Crop, true, false), (4080, 4080));

    // A clip that holds the lens's whole view: its own frame too, by
    // its own resolution choice (not the Crop clips'). At 8K — the
    // default — a frame no larger than that is not enlarged: what is
    // asked for is the frame itself, or, where the hardware encoder has
    // a frame limit and the stabilized frame is past it, the size that
    // frame is held to.
    assert!(lens.stabilize);
    let held = super::lens_frame_held_dims(3840, 3840, false);
    assert_eq!(held, super::hw_h265_side_limit().map(|_| (4096, 4096)));
    let asked = held.unwrap_or((3840, 3840));
    let c = cfg(&on, &lens, insv, (3840, 3840), None);
    assert_eq!((c.projection, c.lens_frame, c.eye_w, c.eye_h), (FisheyeExportProjection::Fisheye, LensFrameOut::FromSource, asked.0, asked.1));
    // Never without stabilization, and never for the clip's own size.
    let still = Settings { stabilize: false, ..lens.clone() };
    let c = cfg(&on, &still, insv, (3840, 3840), None);
    assert_eq!((c.eye_w, c.eye_h), (3840, 3840));
    let own = ExportOptions { resolution: ExportResolution::Native, ..on };
    assert_eq!((cfg(&own, &lens, insv, (3840, 3840), None).eye_w), 3840);
    // What is written at 8K: the held size, or the frame with its margin.
    assert_eq!(on.lens_frame_out_dims(3840, 3840, SizeList::Plain, true, false), held.unwrap_or((4224, 4224)));
    assert_eq!(on.lens_frame_out_dims(3840, 3840, SizeList::Plain, false, false), (3840, 3840));
    assert_eq!(own.lens_frame_out_dims(3840, 3840, SizeList::Plain, true, false), (4224, 4224));
    assert_eq!(super::lens_frame_held_dims(3840, 3840, true), None);
    assert_eq!(on.lens_frame_out_dims(3840, 3840, SizeList::Plain, true, true), (4224, 4224));
    assert_eq!(on.lens_frame_out_dims(4864, 4864, SizeList::Plain, true, true), (4096, 4096));
    // A frame whose margin fits needs no holding, a GoPro cross does, and
    // a frame that is not square keeps its shape.
    assert_eq!(super::lens_frame_held_dims(3712, 3712, false), None);
    assert_eq!(super::lens_frame_held_dims(3936, 3936, false), super::hw_h265_side_limit().map(|_| (4096, 4096)));
    assert_eq!(super::lens_frame_held_dims(3840, 3008, false), super::hw_h265_side_limit().map(|_| (4096, 3272)));
    let c = cfg(&on, &lens, insv, (4864, 4864), None);
    assert_eq!((c.lens_frame, c.eye_w, c.eye_h), (LensFrameOut::FromSource, 4096, 4096));
    let full = ExportOptions { resolution: ExportResolution::Native, crop_resolution: ExportResolution::R8k, ..on };
    let c = cfg(&full, &lens, insv, (4864, 4864), None);
    assert_eq!((c.lens_frame, c.eye_w, c.eye_h), (LensFrameOut::FromSource, 4864, 4864));
    // Every camera's video: a DJI, a QooCam, a Blackmagic frame that is
    // not square, a generic fisheye file.
    for (kind, eye) in [(SourceKind::DjiOsv, (3840, 3840)), (SourceKind::KandaoQoocam, (3840, 3840)), (SourceKind::BlackmagicRaw, (6144, 8040))] {
        let c = cfg(&full, &lens, kind, eye, None);
        assert_eq!((c.projection, c.lens_frame, c.eye_w, c.eye_h), (FisheyeExportProjection::Fisheye, LensFrameOut::FromSource, eye.0, eye.1), "{kind:?}");
    }
    assert_eq!({ let c = cfg(&on, &lens, SourceKind::BlackmagicRaw, (6144, 8040), None); (c.eye_w, c.eye_h) }, (3132, 4096));
    let fisheye_file = Settings { sbs_dewarp_fisheye: true, ..lens.clone() };
    assert_eq!(cfg(&full, &fisheye_file, SourceKind::SbsFisheye, (4096, 4096), None).lens_frame, LensFrameOut::FromSource);
    // A GoPro `.360` has no eye frame of its own: its lens's cross
    // (5952-wide streams: 3936) is the frame.
    let gopro = App::build_export_cfg(
        std::path::Path::new("/nonexistent/in.360"), std::path::Path::new("/nonexistent/out.mp4"),
        29.97, SourceKind::GoProEac, 0, 0, 5952, 1920, None, &lens, &on, Vec::new(),
    );
    let asked = super::lens_frame_held_dims(3936, 3936, false).unwrap_or((3936, 3936));
    assert_eq!((gopro.projection, gopro.lens_frame, gopro.eye_w, gopro.eye_h), (FisheyeExportProjection::Fisheye, LensFrameOut::FromSource, asked.0, asked.1));

    // The VR180 view is VR180, whatever the lens choice says.
    let c = cfg(&on, &vr, insv, (4912, 4912), Some(6560));
    assert_eq!((c.projection, c.lens_frame, c.eye_w), (FisheyeExportProjection::HalfEquirect, LensFrameOut::Off, 6560));
    let c = cfg(&on, &vr, insv, (3840, 3840), None);
    assert_eq!((c.projection, c.lens_frame, c.eye_w), (FisheyeExportProjection::HalfEquirect, LensFrameOut::Off, 4096));
    // The equidistant lens: the disc, in a square frame. So is a clip
    // with no lens record (a file that is already VR180, a still).
    let disc = Settings { lens_space: LensSpaceLens::Equidistant, ..lens.clone() };
    let c = cfg(&native, &disc, insv, (6000, 3968), Some(6560));
    assert_eq!((c.projection, c.lens_frame, c.eye_w, c.eye_h), (FisheyeExportProjection::Fisheye, LensFrameOut::Off, 3968, 3968));
    let c = cfg(&full, &lens, SourceKind::SbsFisheye, (4096, 4096), None);
    assert_eq!((c.projection, c.lens_frame), (FisheyeExportProjection::Fisheye, LensFrameOut::Off));
    for still in [SourceKind::Insta360Dng, SourceKind::KandaoDng] {
        assert_eq!(cfg(&on, &lens, still, (7760, 7760), None).lens_frame, LensFrameOut::Off, "{still:?}");
    }
    let flat = Settings { fisheye_output_mode: FisheyeOutputMode::Reframe, ..Default::default() };
    assert_eq!(cfg(&on, &flat, insv, (4912, 4912), Some(6560)).lens_frame, LensFrameOut::Off);
    assert!(lens.lens_frame_view(insv) && lens.lens_frame_view(SourceKind::GoProEac));
    assert!(!lens.lens_frame_view(SourceKind::Insta360Dng) && !lens.lens_frame_view(SourceKind::Unknown));
    assert!(!flat.lens_frame_view(insv) && !vr.lens_frame_view(insv) && !disc.lens_frame_view(insv));
    assert!(crate::decoder::lens_frame_possible(SourceKind::SbsFisheye, &fisheye_file)
        && !crate::decoder::lens_frame_possible(SourceKind::SbsFisheye, &lens));
    // Settings saved before the lens choice existed read as the camera's.
    let old: Settings = serde_json::from_str(r#"{"fisheye_output_mode":"Fisheye"}"#).unwrap();
    assert!(old.lens_frame_view(insv));
}

/// The Format window follows the Lens Space view: through the camera's
/// lens its sizes are the frame's own and the notes say where the file
/// plays; the disc and every other view keep the usual sizes.
#[test]
fn the_format_window_follows_the_lens_space_view() {
    use super::{tr_zh, LENS_FRAME_METADATA_NOTE, LENS_FRAME_NOTE, LENS_FRAME_STAB_NOTE, LENS_SPACE_CAMERA_HOVER,
        LENS_SPACE_CAMERA_NOTE, LENS_SPACE_DISC_METADATA_NOTE, LENS_SPACE_DISC_NOTE, LENS_SPACE_EQUIDISTANT_HOVER};
    use crate::decoder::LensSpaceLens::{Camera, Equidistant};
    let mut opts = ExportOptions { crop_resolution: ExportResolution::Sharp, ..Default::default() };
    // Another view: the usual sizes, and no word of a lens.
    let text = panel_text_with(&mut opts, (9824, 4912), Some((13120, 6560)), None, false);
    assert!(text.contains("13120 × 6560 (native sharpness)"), "{text}");
    assert!(!text.contains("Lens Space") && !text.contains("equidistant disc"), "{text}");
    // Through the camera's lens: the clip's own size stands for its
    // native sharpness, and the notes say where the file plays.
    let text = panel_text_with(&mut opts, (9824, 4912), Some((13120, 6560)), Some(Camera), false);
    assert!(text.contains("9824 × 4912 (native)") && !text.contains("13120"), "{text}");
    assert!(text.contains(LENS_FRAME_NOTE) && text.contains(LENS_FRAME_METADATA_NOTE), "{text}");
    assert!(text.contains(LENS_FRAME_STAB_NOTE), "{text}");
    assert!(!text.contains("YouTube (Spherical V2"), "{text}");
    // The stored choices are untouched by the look.
    assert_eq!(opts.crop_resolution, ExportResolution::Sharp);
    // A frame that is not square, one eye.
    let text = panel_text_with(&mut opts, (6000, 3968), Some((6560, 6560)), Some(Camera), true);
    assert!(text.contains("6000 × 3968 (native)"), "{text}");
    // The equidistant disc: the usual sizes and the metadata choice,
    // with what that choice means for a disc.
    let text = panel_text_with(&mut opts, (7680, 3840), None, Some(Equidistant), false);
    assert!(text.contains(LENS_SPACE_DISC_METADATA_NOTE) && text.contains("YouTube (Spherical V2"), "{text}");
    assert!(!text.contains(LENS_FRAME_NOTE) && text.contains("\n8192 × 4096\n"), "{text}");
    for s in [LENS_SPACE_CAMERA_HOVER, LENS_SPACE_EQUIDISTANT_HOVER, LENS_SPACE_CAMERA_NOTE, LENS_SPACE_DISC_NOTE,
              LENS_FRAME_NOTE, LENS_FRAME_STAB_NOTE, LENS_FRAME_METADATA_NOTE, LENS_SPACE_DISC_METADATA_NOTE,
              "Lens Space", "Lens", "Camera lens", "Equidistant", "native + margin"] {
        assert_ne!(tr_zh(s), s, "no Chinese text for: {s}");
    }
    // A clip that holds the lens's whole view (an Osmo 360, an X6 at 8K):
    // its own size by its own choice. Its 8K choice has no smaller copy
    // to offer, and shows the frame as it is.
    let mut whole = ExportOptions { resolution: ExportResolution::Native, ..Default::default() };
    let text = panel_text_with(&mut whole, (7680, 3840), None, Some(Camera), false);
    assert!(text.contains("7680 × 3840 (native)") && text.contains(LENS_FRAME_NOTE), "{text}");
    let text = panel_text_lens(&mut whole, (7680, 3840), None, Some(Camera), true, false);
    assert!(text.contains("8448 × 4224 (native + margin)"), "{text}");
    let mut at_8k = ExportOptions::default();
    let text = panel_text_lens(&mut at_8k, (7680, 3840), None, Some(Camera), true, false);
    if super::hw_h265_side_limit().is_some() {
        // Where the hardware encoder stops at 8192 a side, the 8K choice
        // of a stabilized frame is that frame held to it, and says so.
        assert!(text.contains("\n8192 × 4096\n") && !text.contains("8448 × 4224"), "{text}");
        assert!(text.contains(super::LENS_FRAME_HELD_NOTE), "{text}");
        assert_ne!(tr_zh(super::LENS_FRAME_HELD_NOTE), super::LENS_FRAME_HELD_NOTE);
        // Not without stabilization: the frame as it is, and no such note.
        let text = panel_text_lens(&mut at_8k, (7680, 3840), None, Some(Camera), false, false);
        assert!(text.contains("7680 × 3840 (native)") && !text.contains(super::LENS_FRAME_HELD_NOTE), "{text}");
        // The clip's own size stays the frame with its margin, with the
        // encoder's note.
        let text = panel_text_lens(&mut whole, (7680, 3840), None, Some(Camera), true, false);
        assert!(!text.contains(super::LENS_FRAME_HELD_NOTE), "{text}");
        assert!(text.contains(oversize_export_hint(whole.codec, 8448, 4224).expect("a note past the encoder's frame")), "{text}");
    } else {
        assert!(text.contains("8448 × 4224 (native + margin)") && !text.contains("\n8192 × 4096\n"), "{text}");
    }
    let text = panel_text_lens(&mut at_8k, (3840, 3840), None, Some(Camera), true, true);
    assert!(text.contains("4224 × 4224 (native + margin)") && !text.contains(super::LENS_FRAME_HELD_NOTE), "{text}");
    assert_eq!(at_8k.resolution, ExportResolution::R8k);
    // A larger one (an X6 at 10K) does have its 8K copy.
    let text = panel_text_with(&mut at_8k, (9728, 4864), None, Some(Camera), false);
    assert!(text.contains("\n8192 × 4096\n"), "{text}");
    // A stabilized clip: the size shown is the one written, the recorded
    // frame plus the stabilizer's margin; the 8K copy keeps its size.
    let text = panel_text_lens(&mut opts, (9824, 4912), Some((13120, 6560)), Some(Camera), true, false);
    assert!(text.contains("10800 × 5400 (native + margin)") && !text.contains("9824 × 4912"), "{text}");
    let text = panel_text_lens(&mut opts, (6000, 3968), Some((6560, 6560)), Some(Camera), true, true);
    assert!(text.contains("6396 × 4364 (native + margin)"), "{text}");
    let mut copy = ExportOptions { crop_resolution: ExportResolution::R8k, ..Default::default() };
    let text = panel_text_lens(&mut copy, (9824, 4912), Some((13120, 6560)), Some(Camera), true, false);
    assert!(text.contains("8192 × 4096"), "{text}");
    // The margin the note names is the pipeline's.
    let margin = (vr180_pipeline::lens_frame::STABILIZER_MARGIN * 100.0).round() as u32;
    assert!(LENS_FRAME_STAB_NOTE.contains(&format!("{margin} % larger on each side")), "{LENS_FRAME_STAB_NOTE}");
    assert!(tr_zh(LENS_FRAME_STAB_NOTE).contains(&format!("每边比拍摄画幅大 {margin}%")));
}

/// The Output panel offers the lens choice on a clip that has a lens
/// record, in the Lens Space view, and says what each lens writes.
#[test]
fn the_output_panel_offers_the_lens_of_the_lens_space_view() {
    use super::{LENS_SPACE_CAMERA_NOTE, LENS_SPACE_DISC_NOTE};
    use crate::decoder::{FisheyeOutputMode, LensSpaceLens, Settings};
    use vr180_pipeline::SourceKind;
    fn collect(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(text) => { out.push_str(text.galley.text()); out.push('\n'); }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
            _ => {}
        }
    }
    let panel = |s: &mut Settings, kind: Option<SourceKind>| -> String {
        let ctx = egui::Context::default();
        let mut text = String::new();
        for _ in 0..2 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 600.0))),
                ..Default::default()
            };
            let out = ctx.run_ui(input, |ui| App::lens_space_ui(ui, s, kind));
            text.clear();
            out.shapes.iter().for_each(|clipped| collect(&clipped.shape, &mut text));
        }
        text
    };
    let mut lens = Settings { fisheye_output_mode: FisheyeOutputMode::Fisheye, ..Default::default() };
    // A clip with a lens: the choice, and the camera's lens to start with.
    let text = panel(&mut lens, Some(SourceKind::Insta360Insv));
    assert!(text.contains("Lens\n") && text.contains("Camera lens") && text.contains("Equidistant"), "{text}");
    assert!(text.contains(LENS_SPACE_CAMERA_NOTE) && !text.contains(LENS_SPACE_DISC_NOTE), "{text}");
    assert_eq!(lens.lens_space, LensSpaceLens::Camera);
    // The equidistant lens: the disc of the source's field.
    let mut disc = Settings { lens_space: LensSpaceLens::Equidistant, ..lens.clone() };
    let text = panel(&mut disc, Some(SourceKind::DjiOsv));
    assert!(text.contains("Camera lens") && text.contains(&format!("195° {LENS_SPACE_DISC_NOTE}")), "{text}");
    let text = panel(&mut disc, Some(SourceKind::GoProEac));
    assert!(text.contains(&format!("185° {LENS_SPACE_DISC_NOTE}")), "{text}");
    // No lens record (a still, a file that is already VR180): no choice,
    // the disc, and the stored choice left alone.
    for kind in [SourceKind::Insta360Dng, SourceKind::SbsFisheye] {
        let text = panel(&mut lens, Some(kind));
        assert!(!text.contains("Camera lens") && text.contains(&format!("195° {LENS_SPACE_DISC_NOTE}")), "{kind:?}: {text}");
    }
    assert_eq!(lens.lens_space, LensSpaceLens::Camera);
    // A generic file whose halves are fisheye images has a lens.
    let mut file = Settings { sbs_dewarp_fisheye: true, ..lens.clone() };
    assert!(panel(&mut file, Some(SourceKind::SbsFisheye)).contains("Camera lens"));
    // The other views say nothing of it.
    for mode in [FisheyeOutputMode::HalfEquirect, FisheyeOutputMode::Reframe] {
        let mut other = Settings { fisheye_output_mode: mode, ..Default::default() };
        assert_eq!(panel(&mut other, Some(SourceKind::Insta360Insv)), "", "{mode:?}");
    }
}

/// "Apply settings to all" stays inside one settings class: a clip that
/// keeps a setup of its own within a camera type is not overwritten.
#[test]
fn apply_to_all_stays_within_a_settings_class() {
    use super::{BatchItem, BatchStatus};
    use vr180_pipeline::SourceKind;
    let item = |source_kind, x6| BatchItem {
        path: "/nonexistent/in.insv".into(),
        fps: 29.97, duration_sec: 10.0, source_kind,
        fisheye_eye_w: 3840, fisheye_eye_h: 3840, width: 3840, height: 3840,
        settings: Default::default(),
        status: BatchStatus::Idle, selected: false, x6, raw_one_lens: false, raw_frame: None, four_channel_sound: false, take: None, upside_down: None, offline: None, before_take: None, copy: None, take_failed: None, stand_in: None,
    };
    let x6 = |single_eye| Some(X6ClipInfo { window: None, ilog: Some(true), paraxial_native: 2086.97, single_lens: single_eye, single_eye });
    let two = item(SourceKind::Insta360Insv, x6(false));
    let one = item(SourceKind::Insta360Insv, x6(true));
    assert_eq!(two.settings_class(), item(SourceKind::Insta360Insv, x6(false)).settings_class());
    assert_eq!(one.settings_class(), item(SourceKind::Insta360Insv, x6(true)).settings_class());
    assert_ne!(one.settings_class(), two.settings_class());
    assert_ne!(two.settings_class(), item(SourceKind::DjiOsv, None).settings_class());
    assert_eq!(item(SourceKind::DjiOsv, None).settings_class(), item(SourceKind::DjiOsv, None).settings_class());
}

#[test]
fn sharp_eye_follows_the_recording_mode() {
    let win = |dw, dh| InsvWindowCrop { src_w: 7744, src_h: 7744, dst_w: dw, dst_h: dh, offset_x: 0, offset_y: 0 };
    let info = |window| X6ClipInfo { window, ilog: Some(true), paraxial_native: 2086.97, single_lens: false, single_eye: false };
    // Both Crop modes are native pixels: 36.4 px/° → 6560 px per 180°.
    assert_eq!(info(Some(win(4912, 4912))).sharp_eye(4912, 4912, X6ViewChoice::Auto), Some(6560));
    assert_eq!(info(Some(win(4912, 3968))).sharp_eye(3712, 3712, X6ViewChoice::Auto), Some(6560));
    // Full-view modes cover more than 180° at their own size.
    assert_eq!(info(Some(win(7680, 7680))).sharp_eye(3712, 3712, X6ViewChoice::Auto), None);
    assert_eq!(info(Some(win(7912, 7912))).sharp_eye(4864, 4864, X6ViewChoice::Auto), None);
    assert_eq!(info(Some(win(7680, 7680))).sharp_eye(3840, 3840, X6ViewChoice::Auto), None);
    // An unconfirmed 3712² clip: only once the user says it is the Crop.
    assert_eq!(info(None).sharp_eye(3712, 3712, X6ViewChoice::Auto), None);
    assert_eq!(info(None).sharp_eye(3712, 3712, X6ViewChoice::Crop), Some(6560));
}
/// A raw take's frame is placed by its own record: one that holds part
/// of the lens's view has a native-sharpness size (the one a Crop video
/// clip of the same pixels has), one that holds the whole view has
/// none, and a reduced copy of a frame is the same frame.
#[test]
fn a_raw_takes_crop_features_follow_its_frame() {
    use vr180_fisheye::insta360::StatedFrame;
    let frame = |side: u32, origin: f64, scale: f64| super::RawFrame {
        stated: StatedFrame { width: side, height: side, origin: (origin, origin), scale: (scale, scale) },
        paraxial_native: 2086.97, one_lens: false,
    };
    assert_eq!(frame(4912, 1416.0, 1.0).sharp_eye(4912, 4912), Some(6560));
    assert_eq!(frame(3968, 1888.0, 1.0).sharp_eye(3968, 3968), Some(6560));
    assert_eq!(frame(3840, 32.0, 0.5).sharp_eye(3840, 3840), None);
    assert_eq!(frame(5504, -84.0, 16.0 / 23.0).sharp_eye(5504, 5504), None);
    let half = frame(4912, 1416.0, 1.0).geometry(2456, 2456);
    assert_eq!((half.scale_x, half.origin_x, half.cropped), (0.5, 1416.0, true));
}
