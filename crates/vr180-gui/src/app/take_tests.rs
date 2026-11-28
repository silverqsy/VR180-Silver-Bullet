use super::*;
use crate::decoder::Settings;
use vr180_pipeline::SourceKind;

/// A take of 5760² twice over at 30 frames a second and 0.8 GB a second.
fn take(name: &str, duration_s: f64, trim: (Option<f64>, Option<f64>)) -> TakeView {
    TakeView {
        path: PathBuf::from(format!("/takes/{name}")), fps: 30.0, duration_s, trim,
        frame_samples: 2 * 5760 * 5760, capture_bytes: (duration_s * 0.8e9) as u64, outside: 0.163, copy: false, lossy_copy: false,
    }
}

fn entry(path: &str, kind: SourceKind, selected: bool, archive: Option<bool>, status: BatchStatus) -> BatchItem {
    let note = |exact: bool| vr180_fisheye::x6b::ArchiveNote {
        exact, target_bits_per_pixel: (!exact).then_some(1.0), bits_per_pixel: Some(1.0), source_bits_per_pixel: Some(3.2), source: None,
        pairs: None, pairs_of_source: None, first_pair: None, first_pair_s: None, outside_black: None,
    };
    BatchItem {
        path: path.into(), fps: 30.0, duration_sec: 120.0, source_kind: kind,
        fisheye_eye_w: 5760, fisheye_eye_h: 5760, width: 5760, height: 5760,
        settings: Settings::default(), status, selected, x6: None, raw_one_lens: false, raw_frame: None, four_channel_sound: false,
        take: kind.is_raw_video().then(|| TakeFacts { capture_bytes: 96_000_000_000, archive: archive.map(note), outside: 0.163 }),
        upside_down: None, offline: None, before_take: None, copy: None, take_failed: None, stand_in: None,
    }
}

/// The take tools with neither button open: their heading and the two.
const CLOSED: &str = "RAW Footage Archiving\nCompressed archive\nCamera original DNGs";

/// Everything the take tools paint, and what they ask for. `click`: a
/// text on them to click first; what is painted and asked for is then
/// what follows the click.
fn shown(tools: &mut TakeTools, view: &TakeToolsView, click: Option<&str>) -> (String, Option<TakeAsk>) {
    fn collect(shape: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
        match shape {
            egui::Shape::Text(text) => out.push((text.galley.text().to_string(), text.pos)),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let mut texts: Vec<(String, egui::Pos2)> = Vec::new();
    let mut ask = None;
    let mut frame = |events: Vec<egui::Event>, tools: &mut TakeTools, texts: &mut Vec<(String, egui::Pos2)>, ask: &mut Option<TakeAsk>| {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(340.0, 3000.0))),
            events,
            ..Default::default()
        };
        let out = ctx.run_ui(input, |ui| {
            if let Some(asked) = App::take_tools_ui(ui, tools, view) {
                *ask = Some(asked);
            }
        });
        texts.clear();
        out.shapes.iter().for_each(|clipped| collect(&clipped.shape, texts));
    };
    // Two frames: a new context lays its first one out without sizes to go by.
    frame(Vec::new(), tools, &mut texts, &mut ask);
    frame(Vec::new(), tools, &mut texts, &mut ask);
    if let Some(label) = click {
        let at = texts.iter().find(|(text, _)| text == label).unwrap_or_else(|| panic!("no {label:?} among {texts:?}")).1 + egui::vec2(3.0, 3.0);
        let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        ask = None;
        frame(vec![egui::Event::PointerMoved(at)], tools, &mut texts, &mut ask);
        frame(vec![button(true)], tools, &mut texts, &mut ask);
        frame(vec![button(false)], tools, &mut texts, &mut ask);
        frame(Vec::new(), tools, &mut texts, &mut ask);
    }
    (texts.iter().map(|(text, _)| text.as_str()).collect::<Vec<_>>().join("\n"), ask)
}

#[test]
fn what_a_run_writes_of_a_take() {
    assert_eq!(take("A", 120.0, (None, None)).written(false), (0.0, 120.0, false));
    assert_eq!(take("A", 120.0, (Some(12.3), Some(45.6))).written(false), (12.3, 45.6, true));
    assert_eq!(take("A", 120.0, (Some(12.3), None)).written(false), (12.3, 120.0, true));
    assert_eq!(take("A", 120.0, (None, Some(45.6))).written(false), (0.0, 45.6, true));
    // The whole take, whatever its trim.
    assert_eq!(take("A", 120.0, (Some(12.3), Some(45.6))).written(true), (0.0, 120.0, false));
    // A trim that leaves out less than half a frame at either end leaves nothing out.
    assert_eq!(take("A", 120.0, (Some(0.01), Some(119.99))).written(false), (0.0, 120.0, false));
    // Ends past the take, or in the wrong order, stay inside it.
    assert_eq!(take("A", 120.0, (Some(-3.0), Some(500.0))).written(false), (0.0, 120.0, false));
    assert_eq!(take("A", 120.0, (Some(50.0), Some(20.0))).written(false), (50.0, 50.0, true));
    // A third of the take is a third of its capture.
    assert_eq!(take("A", 120.0, (None, None)).bytes_of(40.0), 32_000_000_000);
    assert_eq!(take("A", 0.0, (None, None)).bytes_of(40.0), 0);
}

/// Before a copy of a stretch is made the user is told what it leaves
/// out, in both languages, with the lengths.
#[test]
fn the_warning_says_what_a_copy_leaves_out() {
    let both = [take("A", 120.0, (Some(12.3), Some(45.6)))];
    let en = take_cut_warning(&both, false).unwrap();
    assert!(en.contains("Only the 33.3 s between In and Out") && en.contains("the 12.3 s before In and the 1 min 14 s after Out are left out") && en.contains("Keep the take"), "{en}");
    let zh = take_cut_warning(&both, true).unwrap();
    assert!(zh.contains("入点到出点之间的 33.3 秒") && zh.contains("入点之前的 12.3 秒和出点之后的 1 分 14 秒不会进副本") && zh.contains("再删除原素材"), "{zh}");
    let head = take_cut_warning(&[take("A", 120.0, (Some(12.3), None))], false).unwrap();
    assert!(head.contains("the 12.3 s before In are left out") && !head.contains("after Out"), "{head}");
    let tail = take_cut_warning(&[take("A", 120.0, (None, Some(45.6)))], true).unwrap();
    assert!(tail.contains("出点之后的 1 分 14 秒不会进副本") && !tail.contains("入点之前"), "{tail}");
    // Nothing to say of a take without a trim.
    assert_eq!(take_cut_warning(&[take("A", 120.0, (None, None))], false), None);
    // Several takes: how many of them are trimmed.
    let several = [take("A", 120.0, (Some(12.3), Some(45.6))), take("B", 60.0, (None, None)), take("C", 60.0, (None, Some(30.0)))];
    let en = take_cut_warning(&several, false).unwrap();
    assert!(en.starts_with("2 of these takes are trimmed") && en.contains("Keep the takes"), "{en}");
    assert!(take_cut_warning(&several, true).unwrap().starts_with("2 条素材设了入出点"));
    assert_eq!(take_cut_warning(&several[1..2], false), None);
}

#[test]
fn the_sizes_a_run_comes_to() {
    let cut = [take("A", 120.0, (Some(12.3), Some(45.6)))];
    // 33.3 s of 120 s at 0.8 GB a second; one bit of 2 × 5760² samples, 999 times.
    assert_eq!(take_sizes_text(&cut, false, Some(1.0), false), "Take 96.0 GB · the part copied 26.6 GB · the copy about 8.3 GB");
    assert_eq!(take_sizes_text(&cut, false, Some(1.0), true), "原素材 96.0 GB · 要复制的部分 26.6 GB · 副本约 8.3 GB");
    // A copy that keeps every sample is as large as its frames make it: no size is promised.
    assert_eq!(take_sizes_text(&cut, false, None, false), "Take 96.0 GB · the part copied 26.6 GB");
    // The whole take.
    assert_eq!(take_sizes_text(&cut, true, Some(0.5), false), "Take 96.0 GB · the copy about 14.9 GB");
    let several = [take("A", 120.0, (Some(12.3), Some(45.6))), take("B", 60.0, (None, None))];
    assert_eq!(take_sizes_text(&several, false, Some(1.0), false), "2 takes 144 GB · the part copied 74.6 GB · the copies about 23.2 GB");
    assert_eq!(take_sizes_text(&several, true, None, true), "2 条素材共 144 GB");
    assert_eq!((size_text(950_000_000), size_text(5_400_000_000), size_text(144_000_000_000)), ("950 MB".into(), "5.4 GB".into(), "144 GB".into()));
    assert_eq!((span_text(33.34, false), span_text(59.97, false), span_text(74.4, true), span_text(-1.0, false)), ("33.3 s".into(), "1 min 0 s".into(), "1 分 14 秒".into(), "0.0 s".into()));
}

#[test]
fn a_take_is_listed_with_what_is_written_of_it() {
    let cut = take("TAKE", 120.0, (Some(12.3), Some(45.6)));
    assert_eq!(take_line(&cut, false, false), "TAKE · 00:12.30 to 00:45.60 · 33.3 s of 2 min 0 s");
    assert_eq!(take_line(&cut, false, true), "TAKE · 00:12.30 到 00:45.60 · 2 分 0 秒中的 33.3 秒");
    assert_eq!(take_line(&cut, true, false), "TAKE · whole take · 2 min 0 s");
    assert_eq!(take_line(&take("TAKE", 9.5, (None, None)), false, true), "TAKE · 整条 · 9.5 秒");
    // Where the folders go, and what the button says.
    let archive = TakeJob::Archive(None);
    assert_eq!(take_place_text(std::slice::from_ref(&cut), archive, None, false), "Written beside the take: TAKE_Archive");
    assert_eq!(take_place_text(std::slice::from_ref(&cut), TakeJob::Originals, None, true), "写在素材旁边：TAKE_OriginalDNG");
    assert_eq!(take_place_text(std::slice::from_ref(&cut), archive, Some("Disk"), false), "Written into the output folder Disk: TAKE_Archive");
    let two = [cut.clone(), take("OTHER", 60.0, (None, None))];
    assert_eq!(take_place_text(&two, archive, None, false), "Written beside the take: <take>_Archive");
    assert_eq!(take_place_text(&two, TakeJob::Originals, Some("Disk"), true), "写到输出文件夹 Disk：<素材名>_OriginalDNG");
    // What a start button says: what its run writes, and whose takes it is of.
    let (this, ticked) = (TakeSet::OnScreen, TakeSet::Ticked);
    assert_eq!((take_start_label(archive, this, 1, false), take_start_label(archive, this, 1, true)), ("Make the archive copy: this take".into(), "制作存档副本：当前素材".into()));
    assert_eq!((take_start_label(archive, ticked, 3, false), take_start_label(archive, ticked, 3, true)), ("Make the archive copies: 3 ticked takes".into(), "制作存档副本：勾选的 3 条".into()));
    assert_eq!((take_start_label(archive, ticked, 1, false), take_start_label(archive, ticked, 1, true)), ("Make the archive copy: 1 ticked take".into(), "制作存档副本：勾选的 1 条".into()));
    assert_eq!((take_start_label(TakeJob::Originals, this, 1, true), take_start_label(TakeJob::Originals, this, 1, false)), ("导出相机原片 DNG：当前素材".into(), "Write the camera original DNGs: this take".into()));
    assert_eq!((take_start_label(TakeJob::Originals, ticked, 2, true), take_start_label(TakeJob::Originals, ticked, 2, false)), ("导出相机原片 DNG：勾选的 2 条".into(), "Write the camera original DNGs: 2 ticked takes".into()));
    // A copy that was made, beside the same frames of its take.
    let copy = TakeCopy { dir: "/takes/TAKE_Archive".into(), bytes: 5_400_000_000, source_bytes: 8_000_000_000 };
    assert_eq!(take_copy_line(&copy, false), "✔ TAKE_Archive · 5.4 GB, 68% of the same frames in the take");
    assert_eq!(take_copy_line(&copy, true), "✔ TAKE_Archive · 5.4 GB，是原素材同样这些帧的 68%");
    assert_eq!(take_copy_line(&TakeCopy { source_bytes: 0, ..copy }, false), "✔ TAKE_Archive");
    // The mark is drawn by the fonts the app always has.
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let font = egui::FontId::default();
    assert!(DONE_MARK.chars().all(|c| ctx.fonts_mut(|f| f.has_glyph(&font, c))), "{DONE_MARK}");
}

/// The Source panel: two buttons under the clip's facts; the one that is
/// open shows its choices, what the run would write and leave out, and
/// the button that starts that run.
#[test]
fn the_take_tools_show_what_a_run_would_write_and_start_it() {
    let takes = [take("TAKE", 120.0, (Some(12.3), Some(45.6)))];
    let view = TakeToolsView { archive: &takes, originals: &takes, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() };
    let mut tools = TakeTools::default();
    // Closed: the heading and the two buttons, nothing else.
    let (text, ask) = shown(&mut tools, &view, None);
    assert_eq!((text.as_str(), ask), (CLOSED, None));

    // The archive copy: its sizes, the stretch from In to Out by default, and what that leaves out.
    let (text, ask) = shown(&mut tools, &view, Some("Compressed archive"));
    assert_eq!((tools.open, ask), (Some(TakeTool::Archive), None));
    for said in [
        TAKE_EXACT_LABEL, "High quality: 1 bit a pixel", "Custom", TAKE_ARCHIVE_NOTE, TAKE_LOSSY_NOTE,
        "The corners of the frame outside the lens's image circle, 16% of its samples, hold no picture, only stray light and noise: they are stored as black.",
        "Range", "In to Out", "Whole take", "TAKE · 00:12.30 to 00:45.60 · 33.3 s of 2 min 0 s",
        "Only the 33.3 s between In and Out go into the copy", TAKE_STRETCH_NOTE,
        "Take 96.0 GB · the part copied 26.6 GB · the copy about 8.3 GB", "Written beside the take: TAKE_Archive", "Make the archive copy: this take",
    ] {
        assert!(text.contains(said), "{said}: {text}");
    }
    assert!(!text.contains(TAKE_EXACT_NOTE) && !text.contains("Use the copy"), "{text}");
    // One start button, for the take on screen: nothing is ticked besides it, and nothing is headed.
    assert!(!text.contains("ticked") && !text.contains(TAKE_ON_SCREEN_TITLE) && !text.contains(TAKE_TICKED_TITLE), "{text}");
    // Its button starts that run, and nothing else.
    let (_, ask) = shown(&mut tools, &view, Some("Make the archive copy: this take"));
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: false }, TakeSet::OnScreen)));

    // The whole take: nothing is left out, and nothing is warned of.
    let (text, ask) = shown(&mut tools, &view, Some("Whole take"));
    assert!(tools.archive_whole && ask.is_none());
    assert!(text.contains("TAKE · whole take · 2 min 0 s") && text.contains("Take 96.0 GB · the copy about 29.9 GB"), "{text}");
    assert!(!text.contains("go into the copy") && !text.contains(TAKE_STRETCH_NOTE) && !text.contains("the part copied"), "{text}");
    // Every sample kept: the size picked is to be kept, and no size is promised.
    let (text, ask) = shown(&mut tools, &view, Some(TAKE_EXACT_LABEL));
    assert_eq!((tools.size, ask), (ArchiveSize::Lossless, Some(TakeAsk::Keep)));
    assert!(text.contains(TAKE_EXACT_NOTE) && !text.contains(TAKE_LOSSY_NOTE) && !text.contains("the copy about"), "{text}");
    assert!(text.contains("stored as black"), "{text}");
    let (_, ask) = shown(&mut tools, &view, Some("Make the archive copy: this take"));
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(None), whole: true }, TakeSet::OnScreen)));
    // A size picked by hand.
    let (text, _) = shown(&mut tools, &view, Some("Custom"));
    assert_eq!(tools.size, ArchiveSize::Custom);
    assert!(text.contains("\n0.75\n") && text.contains("bit a pixel") && text.contains("the copy about 22.4 GB"), "{text}");
    assert_eq!(tools.archive_run(&takes), TakeRun { job: TakeJob::Archive(Some(0.75)), whole: true });

    // While a run is in flight nothing starts.
    let busy = TakeToolsView { idle: false, ..TakeToolsView { archive: &takes, originals: &takes, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() } };
    let (text, ask) = shown(&mut tools, &busy, Some("Make the archive copy: this take"));
    assert!(text.contains("Make the archive copy: this take") && ask.is_none(), "{ask:?}");

    // A take without a trim, whose frame is the middle of the sensor: no range to pick, nothing left out, no
    // corner outside the image circle to say anything of.
    let plain = [TakeView { outside: 0.0, ..take("PLAIN", 60.0, (None, None)) }];
    let mut tools = TakeTools { open: Some(TakeTool::Archive), ..Default::default() };
    let (text, _) = shown(&mut tools, &TakeToolsView { archive: &plain, originals: &plain, copies: &[], failed: &[], out_dir: Some("Disk"), idle: true, ..Default::default() }, None);
    assert!(text.contains("PLAIN · whole take · 1 min 0 s") && text.contains("Written into the output folder Disk: PLAIN_Archive"), "{text}");
    assert!(!text.contains("Range") && !text.contains("In to Out") && !text.contains("left out") && !text.contains("image circle"), "{text}");
    assert_eq!(tools.archive_run(&plain), TakeRun { job: TakeJob::Archive(Some(1.0)), whole: true });
    // What is said of the corners, for one take and for several, in both languages.
    assert_eq!(take_corners_note(&plain, false), None);
    assert_eq!(take_corners_note(&takes, true).unwrap(), "画幅四角在镜头成像圈之外，占样本的 16%，没有画面，只有杂散光和噪声：存成黑。");
    let mixed = [plain[0].clone(), takes[0].clone()];
    assert_eq!(take_corners_note(&mixed, false).unwrap(), "The corners of a frame outside the lens's image circle hold no picture, only stray light and noise: they are stored as black.");
    assert!(take_corners_note(&mixed, true).unwrap().contains("存成黑"));

    // Takes ticked in the list besides the one on screen: a second start button, for them. Each of the two
    // stands under its heading and its takes, each take with what is written of it, the rest of a long run
    // by their number.
    let many: Vec<TakeView> = (0..8).map(|k| take(&format!("T{k}"), 60.0, if k == 0 { (Some(10.0), None) } else { (None, None) })).collect();
    let both = TakeToolsView { archive: &plain, originals: &plain, archive_ticked: &many, originals_ticked: &many, idle: true, ..Default::default() };
    let (text, ask) = shown(&mut tools, &both, Some("Make the archive copies: 8 ticked takes"));
    assert!(text.contains("T0 · 00:10.00 to 01:00.00 · 50.0 s of 1 min 0 s") && text.contains("T5 · whole take") && !text.contains("T6 ·") && text.contains("and 2 more"), "{text}");
    assert!(text.contains("1 of these takes are trimmed") && text.contains("<take>_Archive"), "{text}");
    assert_eq!(text.matches(TAKE_STRETCH_NOTE).count(), 1, "{text}");
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: false }, TakeSet::Ticked)));
    // The take on screen comes first, and its button starts a run of it alone: whole here, as it has no trim.
    let lines: Vec<&str> = text.lines().collect();
    let at = |said: &str| lines.iter().position(|line| *line == said).unwrap_or_else(|| panic!("{said}: {text}"));
    let order = [TAKE_ON_SCREEN_TITLE, "PLAIN · whole take · 1 min 0 s", "Written beside the take: PLAIN_Archive", "Make the archive copy: this take",
        TAKE_TICKED_TITLE, "T0 · 00:10.00 to 01:00.00 · 50.0 s of 1 min 0 s", "Written beside the take: <take>_Archive", "Make the archive copies: 8 ticked takes"];
    assert!(order.windows(2).all(|pair| at(pair[0]) < at(pair[1])), "{text}");
    let (_, ask) = shown(&mut tools, &both, Some("Make the archive copy: this take"));
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: true }, TakeSet::OnScreen)));
    // One take ticked, and another on screen: the first button is still the take on screen's, and the ticked
    // take's own says that it is the ticked one.
    let other = [take("OTHER", 600.0, (Some(39.3), Some(230.2)))];
    let two = TakeToolsView { archive: &plain, originals: &plain, archive_ticked: &other, originals_ticked: &other, idle: true, ..Default::default() };
    let (text, ask) = shown(&mut tools, &two, Some("Make the archive copy: this take"));
    assert!(text.contains("Make the archive copy: 1 ticked take") && text.contains("Written beside the take: OTHER_Archive"), "{text}");
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: true }, TakeSet::OnScreen)));
    let (_, ask) = shown(&mut tools, &two, Some("Make the archive copy: 1 ticked take"));
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: false }, TakeSet::Ticked)));
}

/// A clip that is an archive copy: no copy is made of it. The copies a
/// run made are listed with what they came to, and the user is told
/// that the takes may now be moved or deleted: by them, the app offers
/// no control that does it.
#[test]
fn copies_are_not_copied_and_the_takes_are_the_users_to_remove() {
    let on_screen = [TakeView { copy: true, ..take("TAKE_Archive", 33.3, (None, None)) }];
    let mut tools = TakeTools { open: Some(TakeTool::Archive), ..Default::default() };
    let (text, ask) = shown(&mut tools, &TakeToolsView { archive: &[], originals: &on_screen, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() }, None);
    assert!(text.contains(TAKE_COPY_SOURCE_NOTE) && !text.contains("Make the archive copy") && !text.contains("Lossless"), "{text}");
    assert_eq!(ask, None);
    // A take ticked in the list while a copy is on screen: the copy is still not copied, and the ticked take
    // has its button.
    let ticked = [take("OTHER", 60.0, (None, None))];
    let view = TakeToolsView { originals: &on_screen, archive_ticked: &ticked, originals_ticked: &ticked, idle: true, ..Default::default() };
    let (text, ask) = shown(&mut tools, &view, Some("Make the archive copy: 1 ticked take"));
    assert!(text.contains(TAKE_COPY_SOURCE_NOTE) && text.contains("OTHER · whole take · 1 min 0 s") && !text.contains("this take") && !text.contains(TAKE_TICKED_TITLE), "{text}");
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Archive(Some(1.0)), whole: true }, TakeSet::Ticked)));
    // A take that is no copy and cannot be copied (its description did not read): nothing is claimed of it.
    let unread = [take("TAKE", 33.3, (None, None))];
    let (text, _) = shown(&mut tools, &TakeToolsView { archive: &[], originals: &unread, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() }, None);
    assert_eq!(text, CLOSED);

    let takes = [take("TAKE", 120.0, (None, None))];
    let copy = |name: &str| TakeCopy { dir: format!("/takes/{name}_Archive").into(), bytes: 5_400_000_000, source_bytes: 8_000_000_000 };
    // Before a copy is made nothing is said of removing the take.
    let (text, _) = shown(&mut tools, &TakeToolsView { archive: &takes, originals: &takes, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() }, None);
    assert!(!text.contains(TAKE_COPIED_NOTE), "{text}");
    let two = [copy("TAKE"), copy("OTHER")];
    let (text, ask) = shown(&mut tools, &TakeToolsView { archive: &takes, originals: &takes, copies: &two, failed: &[], out_dir: None, idle: true, ..Default::default() }, None);
    assert!(text.contains("✔ TAKE_Archive · 5.4 GB, 68% of the same frames in the take") && text.contains("✔ OTHER_Archive") && text.contains(TAKE_COPIED_NOTE), "{text}");
    assert_eq!(ask, None);
    // The only thing to click there is what starts a run.
    for gone in ["Use the copy", "Use all the copies", "Delete", "Move"] {
        assert!(!text.lines().any(|line| line.starts_with(gone)), "{gone}: {text}");
    }
    assert!(TAKE_COPIED_NOTE.contains("yours to do") && TAKE_COPIED_NOTE.contains("moves and deletes nothing"));
}

/// The camera's own frames: from In to Out by default, as before, and
/// from a lossy copy with a word on what they then are.
#[test]
fn the_cameras_own_frames_are_written_from_the_source_panel() {
    let takes = [take("TAKE", 120.0, (Some(12.3), Some(45.6)))];
    let mut tools = TakeTools::default();
    let view = TakeToolsView { archive: &takes, originals: &takes, copies: &[], failed: &[], out_dir: None, idle: true, ..Default::default() };
    let (text, _) = shown(&mut tools, &view, Some("Camera original DNGs"));
    assert_eq!(tools.open, Some(TakeTool::Originals));
    for said in [TAKE_ORIGINALS_NOTE, "Range", "TAKE · 00:12.30 to 00:45.60 · 33.3 s of 2 min 0 s", "999 frames", "Written beside the take: TAKE_OriginalDNG", TAKE_ORIGINALS_PLACE_NOTE, "Write the camera original DNGs: this take"] {
        assert!(text.contains(said), "{said}: {text}");
    }
    assert!(!text.contains("Lossless") && !text.contains(TAKE_LOSSY_SOURCE_NOTE) && !text.contains("go into the copy") && !text.contains("ticked"), "{text}");
    let (_, ask) = shown(&mut tools, &view, Some("Write the camera original DNGs: this take"));
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Originals, whole: false }, TakeSet::OnScreen)));
    // The ticked takes have a start button of their own; a lossy copy among them is said to be one there.
    let ticked = [take("OTHER", 60.0, (None, None)), TakeView { copy: true, lossy_copy: true, ..take("OLD_Archive", 10.0, (None, None)) }];
    let both = TakeToolsView { archive: &takes, originals: &takes, originals_ticked: &ticked, idle: true, ..Default::default() };
    let (text, ask) = shown(&mut tools, &both, Some("Write the camera original DNGs: 2 ticked takes"));
    assert!(text.contains(TAKE_LOSSY_TICKED_NOTE) && !text.contains(TAKE_LOSSY_SOURCE_NOTE), "{text}");
    assert!(text.contains(TAKE_ON_SCREEN_TITLE) && text.contains(TAKE_TICKED_TITLE) && text.contains("999 frames") && text.contains("2100 frames"), "{text}");
    assert!(text.contains("Write the camera original DNGs: this take") && text.contains("Written beside the take: <take>_OriginalDNG"), "{text}");
    assert_eq!(text.matches(TAKE_ORIGINALS_PLACE_NOTE).count(), 1, "{text}");
    assert_eq!(ask, Some(TakeAsk::Start(TakeRun { job: TakeJob::Originals, whole: true }, TakeSet::Ticked)));
    let (text, _) = shown(&mut tools, &view, Some("Whole take"));
    assert!(tools.originals_whole && !tools.archive_whole && text.contains("3600 frames"), "{text}");
    // The other button closes this one's choices and opens its own.
    let (text, _) = shown(&mut tools, &view, Some("Compressed archive"));
    assert!(tools.open == Some(TakeTool::Archive) && !text.contains(TAKE_ORIGINALS_NOTE), "{text}");
    // Clicked again, it closes.
    let (text, _) = shown(&mut tools, &view, Some("Compressed archive"));
    assert_eq!((tools.open, text.as_str()), (None, CLOSED));

    let lossy = [TakeView { copy: true, lossy_copy: true, ..take("TAKE_Archive", 33.3, (None, None)) }];
    let mut tools = TakeTools { open: Some(TakeTool::Originals), ..Default::default() };
    let (text, _) = shown(&mut tools, &TakeToolsView { archive: &[], originals: &lossy, copies: &[], failed: &[], out_dir: Some("Disk"), idle: true, ..Default::default() }, None);
    assert!(text.contains(TAKE_LOSSY_SOURCE_NOTE) && !text.contains(TAKE_ORIGINALS_PLACE_NOTE) && !text.contains("Range"), "{text}");
}

/// Every word of the take tools is there in Chinese.
#[test]
fn the_take_tools_read_in_both_languages() {
    for text in [
        TAKE_ARCHIVE_HOVER, TAKE_ARCHIVE_NOTE, TAKE_EXACT_NOTE, TAKE_LOSSY_NOTE, TAKE_STRETCH_NOTE, TAKE_COPY_SOURCE_NOTE, TAKE_ORIGINALS_NOTE,
        TAKE_ORIGINALS_PLACE_NOTE, TAKE_LOSSY_SOURCE_NOTE, TAKE_LOSSY_TICKED_NOTE, TAKE_COPIED_NOTE, STAND_IN_HOVER, TAKE_BUSY_HOVER,
        TAKE_ON_SCREEN_TITLE, TAKE_TICKED_TITLE,
        TAKE_TOOLS_TITLE, TAKE_ARCHIVE_LABEL, TAKE_ORIGINALS_LABEL, "Archive copy", TAKE_EXACT_LABEL, "High quality: 1 bit a pixel", "Custom", "bit a pixel",
        "Range", "In to Out", "Whole take", "In place of",
    ] {
        assert_ne!(tr_zh(text), text, "{text}");
    }
}

/// Which takes a run writes: the clip on screen, or the ticked ones,
/// as the button clicked says. A tick in the list sends a run of the
/// clip on screen nowhere else. A copy is made of a camera's take only.
#[test]
fn which_takes_a_run_writes() {
    let raw = SourceKind::X6Raw;
    let idle = || BatchStatus::Idle;
    let (this, ticked) = (TakeSet::OnScreen, TakeSet::Ticked);
    let mut batch = vec![
        entry("/takes/A", raw, true, None, idle()),
        entry("/takes/A_Archive", raw, true, Some(true), idle()),
        entry("/clips/b.insv", SourceKind::Insta360Insv, true, None, idle()),
        entry("/takes/C", raw, false, None, idle()),
    ];
    let (archive, originals) = (TakeJob::Archive(None), TakeJob::Originals);
    // The clip on screen, whatever is ticked elsewhere in the list.
    assert_eq!((take_clips(&batch, Some(3), archive, this), take_clips(&batch, Some(3), originals, this)), (vec![3], vec![3]));
    // The ticked ones the run is for, behind the other button.
    assert_eq!((take_clips(&batch, Some(3), archive, ticked), take_clips(&batch, Some(3), originals, ticked)), (vec![0], vec![0, 1]));
    assert_eq!((take_buttons(&batch, Some(3), archive), take_buttons(&batch, Some(3), originals)), ((vec![3], vec![0]), (vec![3], vec![0, 1])));
    // The one on screen is ticked with others: it is in both runs.
    assert_eq!(take_buttons(&batch, Some(0), originals), (vec![0], vec![0, 1]));
    // It is the only one ticked that the run is for: one button, the first.
    assert_eq!(take_buttons(&batch, Some(0), archive), (vec![0], vec![]));
    // A copy on screen: none is made of it, and the ticked take keeps its button.
    assert_eq!((take_buttons(&batch, Some(1), archive), take_buttons(&batch, Some(1), originals)), ((vec![], vec![0]), (vec![1], vec![0, 1])));
    // Nothing ticked that the run is for: the clip on screen alone, where it is for that.
    batch.iter_mut().for_each(|b| b.selected = false);
    batch[2].selected = true;
    assert_eq!((take_buttons(&batch, Some(3), archive), take_buttons(&batch, Some(3), originals)), ((vec![3], vec![]), (vec![3], vec![])));
    assert_eq!((take_buttons(&batch, Some(1), archive), take_buttons(&batch, Some(1), originals)), ((vec![], vec![]), (vec![1], vec![])));
    assert_eq!((take_buttons(&batch, Some(2), archive), take_buttons(&batch, None, originals)), ((vec![], vec![]), (vec![], vec![])));
    // A take whose files are not at hand, or whose description did not read, is left out.
    batch[3].take = None;
    assert_eq!(take_clips(&batch, Some(3), archive, this), Vec::<usize>::new());
    assert!(entry("/takes/A", raw, false, None, idle()).camera_take() && !entry("/takes/A", raw, false, Some(false), idle()).camera_take());
    assert_eq!((entry("/takes/A", raw, false, None, idle()).frame_samples(), BatchItem { raw_one_lens: true, ..entry("/takes/A", raw, false, None, idle()) }.frame_samples()), (2 * 5760 * 5760, 5760 * 5760));
}

/// A run that writes the takes themselves leaves the export queue as it
/// found it: what was exported stays exported, what waits still waits,
/// and the run's own outcomes are not taken for exports.
#[test]
fn a_take_run_leaves_the_export_queue_as_it_found_it() {
    let raw = SourceKind::X6Raw;
    let before = [BatchStatus::Done, BatchStatus::Idle, BatchStatus::Failed("x".into()), BatchStatus::Queued, BatchStatus::Skipped];
    let mut batch: Vec<BatchItem> = before.iter().enumerate().map(|(i, s)| entry(&format!("/takes/T{i}"), raw, false, None, s.clone())).collect();
    queue_put_aside(&mut batch, &[1, 4]);
    // The run's takes alone are armed: an export that was waiting does not run as part of it.
    assert_eq!(batch.iter().map(|b| b.status.clone()).collect::<Vec<_>>(),
        vec![BatchStatus::Idle, BatchStatus::Queued, BatchStatus::Idle, BatchStatus::Idle, BatchStatus::Queued]);
    batch[1].status = BatchStatus::Done;
    batch[4].status = BatchStatus::Failed("ffmpeg: archive copy: the disk is full".into());
    let job = TakeJob::Archive(None);
    queue_brought_back(&mut batch, Some(job));
    assert_eq!(batch.iter().map(|b| b.status.clone()).collect::<Vec<_>>(), before.to_vec());
    assert!(batch.iter().all(|b| b.before_take.is_none()));
    // The take the run could not write keeps the reason; an export that had failed before is no such take.
    assert_eq!(batch.iter().map(|b| b.take_failed.clone()).collect::<Vec<_>>(),
        vec![None, None, None, None, Some((job, "archive copy: the disk is full".to_string()))]);
    // Nothing put aside: nothing to bring back.
    batch[0].status = BatchStatus::Running;
    queue_brought_back(&mut batch, Some(job));
    assert_eq!((batch[0].status.clone(), batch[2].take_failed.clone()), (BatchStatus::Running, None));
    // The next run of that take starts without the old reason.
    queue_put_aside(&mut batch, &[4]);
    assert_eq!(batch[4].take_failed, None);
    queue_brought_back(&mut batch, None);

    // What could not be written is said under the button it belongs to.
    let takes = [take("TAKE", 60.0, (None, None))];
    let failed = [(job, "T4".to_string(), "archive copy: the disk is full".to_string()), (TakeJob::Originals, "T2".to_string(), "no room".to_string())];
    let view = TakeToolsView { archive: &takes, originals: &takes, copies: &[], failed: &failed, out_dir: None, idle: true, ..Default::default() };
    let (text, _) = shown(&mut TakeTools { open: Some(TakeTool::Archive), ..Default::default() }, &view, None);
    assert!(text.contains("✗ T4: archive copy: the disk is full") && !text.contains("T2"), "{text}");
    let (text, _) = shown(&mut TakeTools { open: Some(TakeTool::Originals), ..Default::default() }, &view, None);
    assert!(text.contains("✗ T2: no room") && !text.contains("T4"), "{text}");
    // A take of noise, of which no copy that keeps every sample would be smaller: the reason, in Chinese too.
    let why = format!("archive copy: {} (8.14 against 7.65 bits a pixel on 3 of its frames): its frames are noise for the most part", vr180_pipeline::raw_archive::NOT_SMALLER);
    assert_eq!(take_failed_text(&why, false), why);
    assert_eq!(take_failed_text(&why, true), "无损副本不会比这条素材更小：每像素 8.14 bit，原素材 7.65 bit。它的帧以噪声为主，相机已经压得差不多了。可改用有损档。");
    assert_eq!(take_failed_text("archive copy: the disk is full", true), "archive copy: the disk is full");
}

/// An archive copy reads the folder of camera-original DNGs of the take
/// it was made from when it has none of its own: what was enhanced for
/// the take goes on being used once the copy has taken its place.
#[test]
fn a_copy_reads_its_takes_camera_originals() {
    let root = std::env::temp_dir().join(format!("vr180-copy-originals-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let clip = |archive: &str| format!(r#"{{"layout":"paired-bayer-v1","fps_setting":60,"size":[3840,3840]{archive}}}"#);
    for (dir, archive) in [("TAKE", String::new()), ("TAKE_Archive", r#","archive":{"version":1,"kind":"exact","source":"TAKE"}"#.to_string()),
        ("LONE_Archive", r#","archive":{"version":1,"kind":"lossy","source":"LONE"}"#.to_string())]
    {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        std::fs::write(root.join(dir).join("clip.json"), clip(&archive)).unwrap();
    }
    std::fs::create_dir_all(root.join("TAKE_OriginalDNG/CAM0")).unwrap();
    assert_eq!(App::originals_dir_for(&root.join("TAKE")), root.join("TAKE_OriginalDNG"));
    assert_eq!(App::originals_dir_for(&root.join("TAKE_Archive")), root.join("TAKE_OriginalDNG"));
    // Its take's folder is not beside it: its own name, as for any take.
    assert_eq!(App::originals_dir_for(&root.join("LONE_Archive")), root.join("LONE_Archive_OriginalDNG"));
    // Frames written from the copy itself come first.
    std::fs::create_dir_all(root.join("TAKE_Archive_OriginalDNG")).unwrap();
    assert_eq!(App::originals_dir_for(&root.join("TAKE_Archive")), root.join("TAKE_Archive_OriginalDNG"));
    let _ = std::fs::remove_dir_all(&root);
}

/// What a run writes of a clip: the take itself only in a run started
/// for that, and only of a raw take.
#[test]
fn a_run_writes_the_take_or_its_view() {
    let (raw, insv, still) = (SourceKind::X6Raw, SourceKind::Insta360Insv, SourceKind::Insta360Dng);
    let video = ExportOptions::default();
    let frames = ExportOptions { raw_dng_sequence: true, ..Default::default() };
    let copy = TakeRun { job: TakeJob::Archive(None), whole: false };
    let originals = TakeRun { job: TakeJob::Originals, whole: true };
    assert_eq!(run_output(None, raw, &video), RunOutput::Video);
    assert_eq!(run_output(None, raw, &frames), RunOutput::DngFrames);
    assert_eq!(run_output(None, insv, &frames), RunOutput::Video);
    assert_eq!(run_output(None, still, &frames), RunOutput::Still);
    // Whatever the export choices are.
    for opts in [&video, &frames] {
        assert_eq!(run_output(Some(copy), raw, opts), RunOutput::Take(TakeJob::Archive(None)));
        assert_eq!(run_output(Some(originals), raw, opts), RunOutput::Take(TakeJob::Originals));
        assert_eq!(run_output(Some(copy), insv, opts), RunOutput::Video);
    }
    assert_eq!(App::take_job_dir_for(std::path::Path::new("/takes/TAKE"), TakeJob::Archive(Some(1.0))), PathBuf::from("/takes/TAKE_Archive"));
    assert_eq!(App::take_job_dir_for(std::path::Path::new("/takes/TAKE"), TakeJob::Originals), PathBuf::from("/takes/TAKE_OriginalDNG"));
    // What a finished run says it did.
    assert_eq!((run_done_text(3, None, true), run_done_text(1, Some(copy.job), false), run_done_text(2, Some(copy.job), false)),
        ("3 exported".into(), "1 archive copy made".into(), "2 archive copies made".into()));
    assert_eq!((run_done_text(2, Some(copy.job), true), run_done_text(1, Some(originals.job), false), run_done_text(2, Some(originals.job), true)),
        ("已制作 2 个存档副本".into(), "camera original DNGs of 1 take written".into(), "已导出 2 条素材的相机原片 DNG".into()));
}

fn note(exact: bool, pairs: Option<(u64, u64, u64)>, source: &str) -> vr180_fisheye::x6b::ArchiveNote {
    vr180_fisheye::x6b::ArchiveNote {
        exact, target_bits_per_pixel: (!exact).then_some(1.0), bits_per_pixel: Some(if exact { 2.1 } else { 0.95 }), source_bits_per_pixel: Some(3.2),
        source: Some(source.into()), pairs: pairs.map(|p| p.1), pairs_of_source: pairs.map(|p| p.2), first_pair: pairs.map(|p| p.0),
        first_pair_s: pairs.map(|p| p.0 as f64 / 30.0), outside_black: None,
    }
}

/// A clip is its take, whichever file it is read from: a trim set on
/// the take lies in a copy where the copy lies in the take, and comes
/// back to the take as it was.
#[test]
fn a_trim_follows_the_clip_between_take_and_copy() {
    // A copy of the stretch from 12.3 s on, 33.3 s long, of a take of 120 s at 30 a second.
    let stretch = CopyExtent::of(&note(true, Some((369, 999, 3600)), "TAKE"), 33.3);
    assert_eq!(stretch, CopyExtent { begins_s: 12.3, duration_s: 33.3, from_start: false, to_end: false });
    // The trim the copy was made by is the whole copy: no trim there. (Its first frame is the one nearest In.)
    assert_eq!(stretch.trim_in_copy((Some(12.3), Some(45.6)), 30.0), (None, None));
    assert_eq!(stretch.trim_in_copy((Some(12.31), Some(45.59)), 30.0), (None, None));
    // Back on the take the clip is that stretch again: from the copy's first frame to half a frame before the
    // end of its last (the take's next frame, a little early as frames can be, stays out).
    let (t_in, t_out) = stretch.trim_in_take((None, None), 30.0);
    assert!((t_in.unwrap() - 12.3).abs() < 1e-9 && (t_out.unwrap() - (45.6 - 1.0 / 60.0)).abs() < 1e-9, "{t_in:?} {t_out:?}");
    // A trim moved after the copy was made lies in the copy where it lay in the take, and comes back as it was.
    let (t_in, t_out) = stretch.trim_in_copy((Some(15.0), Some(40.0)), 30.0);
    assert!((t_in.unwrap() - 2.7).abs() < 1e-9 && (t_out.unwrap() - 27.7).abs() < 1e-9, "{t_in:?} {t_out:?}");
    let (t_in, t_out) = stretch.trim_in_take((t_in, t_out), 30.0);
    assert!((t_in.unwrap() - 15.0).abs() < 1e-9 && (t_out.unwrap() - 40.0).abs() < 1e-9, "{t_in:?} {t_out:?}");
    // Past the copy's ends: no trim there.
    assert_eq!(stretch.trim_in_copy((Some(5.0), Some(100.0)), 30.0), (None, None));
    assert_eq!(stretch.trim_in_copy((None, None), 30.0), (None, None));

    // A copy of the whole take: the trim is the take's, both ways.
    let whole = CopyExtent::of(&note(true, Some((0, 3600, 3600)), "TAKE"), 120.0);
    assert_eq!(whole, CopyExtent { begins_s: 0.0, duration_s: 120.0, from_start: true, to_end: true });
    assert_eq!(whole.trim_in_copy((Some(12.3), Some(45.6)), 30.0), (Some(12.3), Some(45.6)));
    assert_eq!(whole.trim_in_take((Some(12.3), Some(45.6)), 30.0), (Some(12.3), Some(45.6)));
    assert_eq!(whole.trim_in_take((None, None), 30.0), (None, None));
    // A copy that begins with the take and ends before it, and one that does not say what it holds.
    let head = CopyExtent::of(&note(false, Some((0, 900, 3600)), "TAKE"), 30.0);
    assert_eq!(head.trim_in_take((None, None), 30.0), (None, Some(30.0 - 1.0 / 60.0)));
    assert_eq!(CopyExtent::of(&note(false, None, "TAKE"), 30.0), CopyExtent { begins_s: 0.0, duration_s: 30.0, from_start: true, to_end: true });

    // An entry that reads a copy in its take's place is saved as the take, with its trim there: the trim it
    // had on the take, to the digit, while it has not been touched on the copy.
    let mut entry = entry("/takes/TAKE_Archive", SourceKind::X6Raw, false, Some(true), BatchStatus::Idle);
    let on_copy = Settings { ettr_stops: 0.5, ..Default::default() };
    assert_eq!(entry.as_saved(&on_copy), (PathBuf::from("/takes/TAKE_Archive"), on_copy.clone()));
    let stand_in = StandIn::of("/takes/TAKE".into(), stretch, (Some(12.3), Some(45.6)), 30.0);
    assert_eq!(stand_in.trim_on_copy, (None, None));
    entry.stand_in = Some(stand_in);
    let (file, saved) = entry.as_saved(&on_copy);
    assert_eq!((file, saved.trim_in_s, saved.trim_out_s, saved.ettr_stops), (PathBuf::from("/takes/TAKE"), Some(12.3), Some(45.6), 0.5));
    // Moved on the copy: where that lies in the take.
    let moved = Settings { trim_in_s: Some(2.7), ..on_copy.clone() };
    let (_, saved) = entry.as_saved(&moved);
    assert!((saved.trim_in_s.unwrap() - 15.0).abs() < 1e-9 && (saved.trim_out_s.unwrap() - (45.6 - 1.0 / 60.0)).abs() < 1e-9, "{:?} {:?}", saved.trim_in_s, saved.trim_out_s);
    assert_eq!((stand_ins_notice(3, false), stand_ins_notice(3, true)),
        ("3 takes are not at hand: their archive copies are read in their place.".to_string(), "3 条原素材不在：改读它们的存档副本。".to_string()));
}

/// The take first, a copy of it only while the take is not at hand:
/// found beside where the take was, or in the output folder, by its
/// name and by what it says it was made of.
#[test]
fn a_missing_take_is_read_from_its_copy_and_a_take_that_is_back_from_itself() {
    let root = std::env::temp_dir().join(format!("vr180-stand-in-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let folder = |dir: &std::path::Path, archive: Option<&str>| {
        std::fs::create_dir_all(dir).unwrap();
        let archive = archive.map(|a| format!(r#","archive":{a}"#)).unwrap_or_default();
        std::fs::write(dir.join("clip.json"), format!(r#"{{"layout":"paired-bayer-v1","fps_setting":60,"size":[3840,3840]{archive}}}"#)).unwrap();
    };
    let (takes, out) = (root.join("takes"), root.join("out"));
    let take = takes.join("TAKE");
    // Nothing there: nothing to read in its place, and the entry stays as it is.
    std::fs::create_dir_all(&takes).unwrap();
    assert_eq!(copy_standing_in(&take, None), None);
    assert_eq!(source_choice(&take, None, None), SourceChoice::Same);
    // A lossy copy beside where the take was.
    folder(&takes.join("TAKE_Archive"), Some(r#"{"version":1,"kind":"lossy","bits_per_pixel":0.95,"source":"TAKE","pairs":900,"pairs_of_source":900,"first_pair":0}"#));
    // What is not that: a copy of another take under a name that begins alike, a take that is no copy, a file.
    folder(&takes.join("TAKE_Archive_of_another"), Some(r#"{"version":1,"kind":"exact","source":"TAKE2","pairs":900,"pairs_of_source":900}"#));
    folder(&takes.join("TAKE_2"), None);
    std::fs::write(takes.join("TAKE_SBS.mp4"), b"x").unwrap();
    let found = copy_standing_in(&take, None).expect("the copy");
    assert_eq!((found.0.as_path(), found.1.exact, found.1.source.as_deref()), (takes.join("TAKE_Archive").as_path(), false, Some("TAKE")));
    assert_eq!(source_choice(&take, None, None), SourceChoice::Copy(found.0.clone(), found.1.clone()));
    // One that keeps every sample comes first, wherever it is: here in the output folder.
    folder(&out.join("TAKE_Archive (2)"), Some(r#"{"version":1,"kind":"exact","bits_per_pixel":2.1,"source":"TAKE","pairs":300,"pairs_of_source":900,"first_pair":100}"#));
    assert_eq!(copy_standing_in(&take, None).unwrap().0, takes.join("TAKE_Archive"));
    assert_eq!(copy_standing_in(&take, Some(&out)).unwrap().0, out.join("TAKE_Archive (2)"));
    // Of two of a kind, the one that holds more of the take.
    folder(&out.join("TAKE_Archive (3)"), Some(r#"{"version":1,"kind":"exact","bits_per_pixel":2.1,"source":"TAKE","pairs":900,"pairs_of_source":900,"first_pair":0}"#));
    assert_eq!(copy_standing_in(&take, Some(&out)).unwrap().0, out.join("TAKE_Archive (3)"));

    // The take is there: it is read, whatever copies lie about.
    folder(&take, None);
    assert_eq!(source_choice(&take, None, Some(&out)), SourceChoice::Same);
    // An entry that reads a copy in the take's place: back to the take once it is there, and not before.
    let stand_in = StandIn::of(take.clone(), CopyExtent { begins_s: 0.0, duration_s: 30.0, from_start: true, to_end: true }, (None, None), 30.0);
    assert_eq!(source_choice(&out.join("TAKE_Archive (3)"), Some(&stand_in), Some(&out)), SourceChoice::Take(take.clone()));
    std::fs::remove_dir_all(&take).unwrap();
    assert_eq!(source_choice(&out.join("TAKE_Archive (3)"), Some(&stand_in), Some(&out)), SourceChoice::Same);
    // A file of another kind that is gone has no copy to stand in.
    assert_eq!(source_choice(&takes.join("clip.insv"), None, Some(&out)), SourceChoice::Same);
    let _ = std::fs::remove_dir_all(&root);

    // What the list says when an entry changes its file, in both languages.
    let (take, copy) = (std::path::Path::new("/takes/TAKE"), std::path::Path::new("/takes/TAKE_Archive"));
    assert_eq!(stand_in_notice(take, copy, true, false), "TAKE is not at hand: its archive copy TAKE_Archive is read in its place.");
    assert_eq!(stand_in_notice(take, copy, true, true), "TAKE 不在：改读它的存档副本 TAKE_Archive。");
    assert_eq!(stand_in_notice(take, copy, false, false), "TAKE is back: it is read again, and its archive copy TAKE_Archive no longer.");
    assert_eq!(stand_in_notice(take, copy, false, true), "TAKE 回来了：重新读原素材，不再读副本 TAKE_Archive。");
}

/// A project names a take; the take is gone and a copy of a stretch of
/// it lies where it was (a take on disk named by `VR180_ARCHIVE_TAKE`,
/// skipped without one): the clip is found as that copy, with the place
/// of the stretch in the take, and as the take itself while that is
/// there.
#[test]
fn a_project_whose_take_is_gone_opens_its_copy() {
    let Some(take) = std::env::var_os("VR180_ARCHIVE_TAKE").map(PathBuf::from) else { return };
    let name = take.file_name().unwrap().to_owned();
    let root = std::env::temp_dir().join(format!("vr180-gone-take-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let index = vr180_pipeline::x6_raw::open(&take).unwrap();
    let first = 30.min(index.frames().saturating_sub(1));
    let frames = first..index.frames().min(first + 4);
    let copy = root.join(format!("{}_Archive", name.to_string_lossy()));
    vr180_pipeline::raw_archive::archive_take(&take, &copy, frames.clone(), &vr180_pipeline::raw_archive::ArchiveOptions::default(), |_| {}, &std::sync::atomic::AtomicBool::new(false)).unwrap();
    // The take is where the project says: it is the clip.
    let there = crate::project::FileRef::new(&take, None);
    let found = locate_clip(&there, None, None, Some(&root)).expect("the take");
    assert!(found.path == take && found.stand_in.is_none());
    // The project names a take that was in `root` and is not: its copy there is read in its place.
    let gone = crate::project::FileRef::new(&root.join(&name), None);
    let found = locate_clip(&gone, None, None, None).expect("the copy");
    let stand_in = found.stand_in.expect("in the take's place");
    assert_eq!((found.path.as_path(), stand_in.take.as_path()), (copy.as_path(), root.join(&name).as_path()));
    assert!(found.facts.source_kind.is_raw_video() && found.facts.take.as_ref().is_some_and(|t| t.archive.is_some()));
    let extent = stand_in.extent;
    assert!((extent.begins_s - index.pts_s(first)).abs() < 1e-5 && !extent.from_start, "{extent:?}");
    assert_eq!(extent.to_end, frames.end == index.frames());
    assert!((extent.duration_s - found.facts.duration_sec).abs() < 1e-9 && (extent.duration_s - frames.len() as f64 / found.facts.fps as f64).abs() < 0.5 / found.facts.fps as f64, "{extent:?}");
    // A trim set on the take that covers the stretch is the whole copy; one inside it lies where it lay.
    let dt = 1.0 / found.facts.fps as f64;
    let trim = (Some(index.pts_s(first)), Some(index.pts_s(frames.end - 1) + dt));
    assert_eq!(extent.trim_in_copy(trim, found.facts.fps), (None, None));
    let (t_in, _) = extent.trim_in_copy((Some(index.pts_s(first + 2)), None), found.facts.fps);
    assert!((t_in.unwrap() - (index.pts_s(first + 2) - index.pts_s(first))).abs() < 1e-5);
    // Nothing where the take was and no copy either: not found.
    assert!(locate_clip(&crate::project::FileRef::new(&root.join("ANOTHER"), None), None, None, None).is_none());
    // The copy elsewhere, in the output folder: found there.
    let elsewhere = crate::project::FileRef::new(&std::env::temp_dir().join("vr180-nowhere").join(&name), None);
    assert!(locate_clip(&elsewhere, None, None, None).is_none());
    assert_eq!(locate_clip(&elsewhere, None, None, Some(&root)).map(|l| l.path), Some(copy.clone()));
    let _ = std::fs::remove_dir_all(&root);
}

/// The Upside-down mount switch follows the clip's accelerometer until
/// the user sets it by hand; a clip without such a record keeps what
/// it has. Settings kept from before the switch could follow anything
/// follow it.
#[test]
fn the_upside_down_switch_follows_the_accelerometer() {
    let mut s = Settings::default();
    assert!(s.camera_upside_down_auto && !s.camera_upside_down);
    follow_the_mount(&mut s, Some(true));
    assert!(s.camera_upside_down);
    follow_the_mount(&mut s, None);
    assert!(s.camera_upside_down, "nothing to go by: as it was");
    follow_the_mount(&mut s, Some(false));
    assert!(!s.camera_upside_down);
    // Set by hand: it stays, whatever the accelerometer says.
    let mut by_hand = Settings { camera_upside_down: true, camera_upside_down_auto: false, ..Default::default() };
    follow_the_mount(&mut by_hand, Some(false));
    assert!(by_hand.camera_upside_down && !by_hand.camera_upside_down_auto);
    // A clip's settings from a project written before: the switch as it was, and free to follow.
    let old: Settings = serde_json::from_str(r#"{"camera_upside_down":false,"stereo_roll_deg":-0.23}"#).unwrap();
    assert!(old.camera_upside_down_auto && !old.camera_upside_down && old.stereo_roll_deg == -0.23);
    let mut old = old;
    follow_the_mount(&mut old, Some(true));
    assert!(old.camera_upside_down);
    // The list takes the fact in with the clip's other facts.
    let mut entry = entry("/takes/TAKE", SourceKind::X6Raw, false, None, BatchStatus::Idle);
    entry.upside_down = Some(true);
    follow_the_mount(&mut entry.settings, entry.upside_down);
    assert!(entry.settings.camera_upside_down && entry.export_settings().output_view_adjust().upside_down);
    // Other sources have no such record.
    assert_eq!(mounted_upside_down(SourceKind::Insta360Dng, std::path::Path::new("/nonexistent/a.dng")), None);
    assert_eq!(mounted_upside_down(SourceKind::X6Raw, std::path::Path::new("/nonexistent/TAKE")), None);
    for text in [UPSIDE_DOWN_HOVER, "Set from the clip's accelerometer.", "Follow the accelerometer", "The switch was set by hand. This puts it back to what the clip's accelerometer says."] {
        assert_ne!(tr_zh(text), text, "{text}");
    }
}

/// A take and its copy opened together are one recording: the take is
/// the one that is opened. A copy opened by itself is opened.
#[test]
fn a_take_opened_with_its_copy_is_opened_alone() {
    let source_of = |p: &std::path::Path| match p.file_name()?.to_str()? {
        "TAKE_Archive" | "TAKE_Archive (2)" => Some("TAKE".to_string()),
        "LONE_Archive" => Some("LONE".to_string()),
        _ => None,
    };
    let paths = |names: &[&str]| names.iter().map(|n| PathBuf::from(format!("/takes/{n}"))).collect::<Vec<_>>();
    let (opened, left) = without_copies_of_opened_takes(paths(&["OTHER", "TAKE", "TAKE_Archive", "LONE_Archive", "clip.insv", "TAKE_Archive (2)"]), source_of);
    assert_eq!(opened, paths(&["OTHER", "TAKE", "LONE_Archive", "clip.insv"]));
    assert_eq!(left, paths(&["TAKE_Archive", "TAKE_Archive (2)"]));
    // The take is elsewhere than its copy: the same recording all the same.
    let (opened, left) = without_copies_of_opened_takes(vec!["/disk/TAKE_Archive".into(), "/takes/TAKE".into()], source_of);
    assert_eq!((opened, left), (vec![PathBuf::from("/takes/TAKE")], vec![PathBuf::from("/disk/TAKE_Archive")]));
    // A copy opened by itself, or with other takes than its own.
    let (opened, left) = without_copies_of_opened_takes(paths(&["TAKE_Archive", "OTHER"]), source_of);
    assert_eq!((opened, left), (paths(&["TAKE_Archive", "OTHER"]), Vec::new()));
}

/// The size picked for an archive copy is kept across launches; what
/// else the panel holds starts over.
#[test]
fn the_size_picked_is_kept() {
    for (size, bits) in [(ArchiveSize::Lossless, 0.75f32), (ArchiveSize::HighQuality, 0.75), (ArchiveSize::Custom, 0.6)] {
        let tools = TakeTools { size, bits, open: Some(TakeTool::Archive), archive_whole: true, originals_whole: true };
        assert_eq!(TakeTools::from_kept(&tools.kept()), TakeTools { size, bits, ..Default::default() });
    }
    assert_eq!(TakeTools::from_kept(""), TakeTools::default());
    assert_eq!(TakeTools::from_kept("something else"), TakeTools::default());
    assert_eq!(TakeTools::from_kept("custom 7").bits, 1.0);
    assert_eq!(TakeTools::from_kept("custom nan").bits, 0.75);
    let d = TakeTools::default();
    assert_eq!((d.size, d.archive_bits(), d.archive_whole, d.originals_whole), (ArchiveSize::HighQuality, Some(1.0), false, false));
    assert_eq!(TakeTools { size: ArchiveSize::Lossless, ..d }.archive_bits(), None);
    for (asked, got) in [(0.75f32, 0.75f64), (0.2, 0.5), (3.0, 1.0)] {
        assert!((TakeTools { size: ArchiveSize::Custom, bits: asked, ..d }.archive_bits().unwrap() - got).abs() < 1e-6);
    }
}

/// A clip that is an archive copy says how it was made, and which
/// stretch of its take it holds when that is not all of it.
#[test]
fn a_copy_says_which_stretch_of_its_take_it_holds() {
    let whole = vr180_fisheye::x6b::ArchiveNote {
        exact: true, target_bits_per_pixel: None, bits_per_pixel: Some(2.1), source_bits_per_pixel: Some(3.2), source: Some("TAKE".into()),
        pairs: Some(1800), pairs_of_source: Some(1800), first_pair: Some(0), first_pair_s: Some(0.0), outside_black: None,
    };
    assert_eq!(archive_note_text(&whole, false), "every sample kept");
    assert_eq!(archive_note_text(&whole, true), "每个样本都保留");
    // A copy that stores the corners outside the image circle as black says so, of either kind.
    let cornered = vr180_fisheye::x6b::ArchiveNote { outside_black: Some(0.163), ..whole.clone() };
    assert_eq!(archive_note_text(&cornered, false), "every sample of the picture kept · corners black");
    assert_eq!(archive_note_text(&cornered, true), "画面内每个样本都保留 · 四角为黑");
    let cornered = vr180_fisheye::x6b::ArchiveNote { exact: false, bits_per_pixel: Some(0.9), ..cornered };
    assert_eq!(archive_note_text(&cornered, false), "0.90 bit a pixel · corners black");
    let part = vr180_fisheye::x6b::ArchiveNote { pairs: Some(300), first_pair: Some(368), ..whole.clone() };
    assert_eq!(archive_note_text(&part, false), "every sample kept · frames 369 to 668 of the take's 1800");
    assert_eq!(archive_note_text(&part, true), "每个样本都保留 · 原素材 1800 帧中的第 369 到 668 帧");
    let lossy = vr180_fisheye::x6b::ArchiveNote { exact: false, target_bits_per_pixel: Some(1.0), bits_per_pixel: Some(0.97), ..part };
    assert_eq!(archive_note_text(&lossy, false), "0.97 bit a pixel · frames 369 to 668 of the take's 1800");
    // A copy made before the note said so: no word on it.
    let old = vr180_fisheye::x6b::ArchiveNote { pairs: None, pairs_of_source: None, first_pair: None, first_pair_s: None, ..lossy };
    assert_eq!(archive_note_text(&old, false), "0.97 bit a pixel");
}
