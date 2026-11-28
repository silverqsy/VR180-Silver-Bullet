use super::{App, ExportCodec, ExportOptions, ReframeSize};
use super::{spatial_life_size_hfov_deg, spatial_view_scaled_note, spatial_view_shown_as, tr_zh, SPATIAL_VIEW_SCALED_NOTE};
use crate::decoder::{FisheyeOutputMode, ReframeAspect, Settings};
use vr180_pipeline::fisheye_export::FisheyeExportProjection;

fn reframed(aspect: ReframeAspect, hfov_deg: f32, defish: f32) -> Settings {
    Settings {
        fisheye_output_mode: FisheyeOutputMode::Reframe,
        reframe_aspect: aspect,
        reframe_hfov_deg: hfov_deg,
        reframe_defish: defish,
        ..Default::default()
    }
}

fn cfg(settings: &Settings, opts: &ExportOptions, kind: vr180_pipeline::SourceKind)
    -> vr180_pipeline::fisheye_export::FisheyeExportConfig
{
    App::build_export_cfg(
        std::path::Path::new("/nonexistent/in.insv"), std::path::Path::new("/nonexistent/out.mov"),
        59.94, kind, 3712, 3712, 3712, 3712, None, settings, opts, Vec::new(),
    )
}

#[test]
fn spatial_video_is_a_flat_rectilinear_export_on_macos() {
    let opts = ExportOptions {
        codec: ExportCodec::Spatial,
        reframe_size: ReframeSize::P2160,
        h265_bit_depth: 8,
        spatial_baseline_mm: 63.0,
        spatial_disparity_pct: 2.5,
        ..Default::default()
    };
    let insv = vr180_pipeline::SourceKind::Insta360Insv;
    let on_mac = cfg!(target_os = "macos");

    // The reframed view, rectilinear: a spatial video (where the encoder exists).
    let flat = reframed(ReframeAspect::Wide, 96.0, 0.0);
    assert_eq!(opts.codec_for(&flat, false), if on_mac { ExportCodec::Spatial } else { ExportCodec::H265 });
    let c = cfg(&flat, &opts, insv);
    assert_eq!((c.eye_w, c.eye_h), (3840, 2160));
    assert_eq!(c.projection, FisheyeExportProjection::Reframe { hfov_deg: 96.0, defish: 0.0 });
    assert_eq!(c.spatial.is_some(), on_mac);
    if let Some(sp) = c.spatial {
        assert_eq!(sp.baseline_mm, 63.0);
        assert!((sp.disparity - 0.025).abs() < 1e-6);
        // Always Main10, whatever the H.265 depth choice says.
        assert_eq!(c.bit_depth, 10);
    }

    // Anything else the choice is left on exports as H.265 side by side:
    // a defished view, a VR180 frame, a still.
    for settings in [reframed(ReframeAspect::Square, 96.0, 0.3), Settings::default()] {
        assert_eq!(opts.codec_for(&settings, false), ExportCodec::H265);
        let c = cfg(&settings, &opts, insv);
        assert!(c.spatial.is_none());
        assert_eq!(c.bit_depth, 8);
    }
    assert_eq!(opts.codec_for(&flat, true), ExportCodec::H265);

    // H.265 and ProRes never turn into a spatial video.
    for codec in [ExportCodec::H265, ExportCodec::ProRes] {
        let o = ExportOptions { codec, ..opts };
        assert_eq!(o.codec_for(&flat, false), codec);
        assert!(cfg(&flat, &o, insv).spatial.is_none());
    }
}

/// A clip that holds ONE eye: the settings it renders with carry nothing
/// that sets two eyes against each other, its frame is one eye wide, and
/// it is never a spatial video.
#[test]
fn a_single_eye_clip_exports_its_one_eye() {
    use crate::decoder::PreviewMode;
    let insv = vr180_pipeline::SourceKind::Insta360Insv;
    let mut s = reframed(ReframeAspect::Wide, 96.0, 0.0);
    s.stereo_yaw_deg = 0.4;
    s.stereo_pitch_deg = -0.2;
    s.stereo_roll_deg = 0.1;
    s.eye_match_ct = 0.1;
    s.eye_match_exposure = 0.2;
    s.auto_eye_match = true;
    s.preview_mode = PreviewMode::Anaglyph;
    s.preview_eye_right = true;
    s.reframe_yaw_deg = 12.0;
    let one = s.for_single_eye();
    assert!(one.single_eye && !s.single_eye);
    assert_eq!((one.preview_mode, one.preview_eye_right), (PreviewMode::SingleEye, false));
    // The view keeps its aim and loses the stereo offset.
    let va = one.output_view_adjust();
    assert_eq!((va.stereo_yaw_deg, va.stereo_pitch_deg, va.stereo_roll_deg), (0.0, 0.0, 0.0));
    assert_eq!(va.pano_yaw_deg, s.output_view_adjust().pano_yaw_deg);
    let plan = one.build_color_stack();
    assert!(plan.auto_eye_match.is_none() && plan.eye_match_ct == 0.0 && plan.eye_match_exposure == 0.0);
    // Everything else is the user's own setting.
    assert_eq!((one.reframe_hfov_deg, one.reframe_aspect, one.fisheye_output_mode),
               (s.reframe_hfov_deg, s.reframe_aspect, s.fisheye_output_mode));

    // Spatial video needs two eye views: the choice runs as H.265.
    let opts = ExportOptions {
        codec: ExportCodec::Spatial, reframe_size: ReframeSize::P2160, beyondvr_hack: true, ..Default::default()
    };
    assert_eq!(opts.codec_for(&one, false), ExportCodec::H265);
    let c = cfg(&one, &opts, insv);
    assert!(c.spatial.is_none());
    assert_eq!((c.eye_w, c.eye_h), (3840, 2160));
    assert_eq!(c.view_adjust.stereo_yaw_deg, 0.0);
    // The rate follows the pixels of the frame written: one eye, not two.
    let two = cfg(&s, &ExportOptions { codec: ExportCodec::H265, ..opts }, insv);
    assert_eq!(two.bitrate_kbps, 210_000);
    assert_eq!(c.bitrate_kbps, 105_000);
    assert_eq!(ReframeSize::P2160.frame_label(ReframeAspect::Wide, true), "3840 × 2160");
    assert_eq!(ReframeSize::P2160.frame_label(ReframeAspect::Wide, false), ReframeSize::P2160.label(ReframeAspect::Wide));

    // VR180: the same per-eye sizes, and no per-eye scale on one eye.
    let vr = Settings::default().for_single_eye();
    let c = cfg(&vr, &ExportOptions { beyondvr_hack: true, ..Default::default() }, insv);
    assert_eq!((c.eye_w, c.eye_h, c.color_stack.eye_scale), (4096, 4096, 1.0));
    assert!(cfg(&Settings::default(), &ExportOptions { beyondvr_hack: true, ..Default::default() }, insv).color_stack.eye_scale < 1.0);
    assert_eq!(super::ExportResolution::R8k.frame_label(true), "4096 × 4096");
    assert_eq!(super::ExportResolution::R8k.frame_label(false), "8192 × 4096");

    // It keeps a setup of its own, apart from the camera's two-eye one.
    assert_ne!(App::settings_class_key(insv, true), App::settings_class_key(insv, false));
    assert_eq!(App::settings_class_key(vr180_pipeline::SourceKind::DjiOsv, true), "osv");
    assert_eq!(FisheyeOutputMode::Reframe.label(false), FisheyeOutputMode::Reframe.as_str());
    assert_ne!(FisheyeOutputMode::Fisheye.label(true), FisheyeOutputMode::Fisheye.as_str());
}

#[test]
fn spatial_video_gets_its_own_file_name() {
    let src = std::path::Path::new("/clips/VID_0005.insv");
    assert_eq!(App::default_output_for(src, ExportCodec::H265, false, false), std::path::Path::new("/clips/VID_0005_SBS.mp4"));
    assert_eq!(App::default_output_for(src, ExportCodec::ProRes, false, false), std::path::Path::new("/clips/VID_0005_SBS.mov"));
    assert_eq!(App::default_output_for(src, ExportCodec::Spatial, false, false), std::path::Path::new("/clips/VID_0005_Spatial.mov"));
    assert_eq!(App::default_output_for(src, ExportCodec::H265, true, false), std::path::Path::new("/clips/VID_0005_Mono.mp4"));
    assert_eq!(App::default_output_for(src, ExportCodec::ProRes, true, false), std::path::Path::new("/clips/VID_0005_Mono.mov"));
    // A lens frame has its own name, next to the VR180 export of the same clip.
    assert_eq!(App::default_output_for(src, ExportCodec::H265, false, true), std::path::Path::new("/clips/VID_0005_Lens.mp4"));
    assert_eq!(App::default_output_for(src, ExportCodec::ProRes, true, true), std::path::Path::new("/clips/VID_0005_Lens.mov"));
}

#[test]
fn a_view_past_the_headset_limits_is_shown_scaled_down() {
    let near = |a: f32, b: f32| (a - b).abs() < 0.02;
    let wide = 16.0 / 9.0;
    // Inside both limits, and exactly at them: life size.
    for (hfov, aspect) in [(60.0, 1.0), (80.0, 1.0), (70.0, wide), (86.0, wide), (90.0, wide)] {
        assert_eq!(spatial_view_shown_as(hfov, aspect), None, "{hfov}° at {aspect}");
    }
    // 1:1: the height reaches 80° first, whatever the frame's own angle.
    for hfov in [80.2, 90.0, 120.0, 150.0] {
        let (h, v) = spatial_view_shown_as(hfov, 1.0).unwrap();
        assert!(near(h, 80.0) && near(v, 80.0), "{hfov}°: {h} × {v}");
    }
    // 16:9: the width reaches 90° first; the height follows the shape.
    for hfov in [90.2, 100.0, 120.0, 150.0] {
        let (h, v) = spatial_view_shown_as(hfov, wide).unwrap();
        assert!(near(h, 90.0) && near(v, 58.72), "{hfov}°: {h} × {v}");
    }
    assert!(near(spatial_life_size_hfov_deg(1.0), 80.0));
    assert!(near(spatial_life_size_hfov_deg(wide), 90.0));
    // The one shape that meets both limits at once.
    let both = 45f32.to_radians().tan() / 40f32.to_radians().tan();
    let (h, v) = spatial_view_shown_as(120.0, both).unwrap();
    assert!(near(h, 90.0) && near(v, 80.0), "{h} × {v}");
}

#[test]
fn the_display_match_switch_reads_in_both_languages() {
    use super::{MATCH_PLAYBACK_HOVER, MATCH_PLAYBACK_LABEL};
    for en in [MATCH_PLAYBACK_LABEL, MATCH_PLAYBACK_HOVER] {
        assert_ne!(tr_zh(en), en, "no translation in the table");
    }
    // Both say what it changes and what it leaves alone.
    assert!(MATCH_PLAYBACK_HOVER.contains("Only the preview changes"));
    assert!(tr_zh(MATCH_PLAYBACK_HOVER).contains("只改变预览"));
}

#[test]
fn the_scaled_view_note_is_complete_in_both_languages() {
    let zh = tr_zh(SPATIAL_VIEW_SCALED_NOTE);
    assert_ne!(zh, SPATIAL_VIEW_SCALED_NOTE, "no translation in the table");
    for template in [SPATIAL_VIEW_SCALED_NOTE, zh] {
        for key in ["{max_w}", "{max_h}", "{view}", "{shown}", "{limit}"] {
            assert!(template.contains(key), "{key} missing in: {template}");
        }
        let note = spatial_view_scaled_note(template, 120.0, 1.0).unwrap();
        assert!(!note.contains('{') && !note.contains('}'), "{note}");
        for part in ["90°", "120.0° × 120.0°", "80.0° × 80.0°"] {
            assert!(note.contains(part), "{part} missing in: {note}");
        }
        assert_eq!(spatial_view_scaled_note(template, 80.0, 1.0), None);
    }
}

/// Every piece of text the export panel paints for a reframed view.
fn panel_text(opts: &mut ExportOptions, aspect: ReframeAspect, hfov_deg: f32, defish: f32) -> String {
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
            App::export_options_ui(ui, opts, "test", Some(aspect), (hfov_deg, defish), 29.97, false, None, None, None, false, false, None);
        });
        text.clear();
        out.shapes.iter().for_each(|clipped| collect(&clipped.shape, &mut text));
    }
    text
}

#[test]
fn the_export_panel_notes_a_view_the_headset_scales_down() {
    // The spatial video choice exists on macOS only.
    if !cfg!(target_os = "macos") {
        return;
    }
    let mut opts = ExportOptions { codec: ExportCodec::Spatial, ..Default::default() };
    let scaled = "is scaled down to";

    let text = panel_text(&mut opts, ReframeAspect::Square, 120.0, 0.0);
    assert!(text.contains("Field of view written: 120.0°"), "{text}");
    assert!(text.contains("This view (120.0° × 120.0°) is scaled down to 80.0° × 80.0°"), "{text}");
    assert!(text.contains("set Zoom to 80° or less"), "{text}");
    let text = panel_text(&mut opts, ReframeAspect::Wide, 120.0, 0.0);
    assert!(text.contains("This view (120.0° × 88.5°) is scaled down to 90.0° × 58.7°"), "{text}");
    assert!(text.contains("set Zoom to 90° or less"), "{text}");

    // At the limits and inside them: nothing to note.
    for (aspect, hfov) in [(ReframeAspect::Square, 80.0), (ReframeAspect::Square, 60.0), (ReframeAspect::Wide, 90.0)] {
        let text = panel_text(&mut opts, aspect, hfov, 0.0);
        assert!(text.contains("Field of view written"), "{text}");
        assert!(!text.contains(scaled), "{text}");
    }
    // Defish above 0: the clip is not exported as spatial, and says so instead.
    let text = panel_text(&mut opts, ReframeAspect::Square, 120.0, 0.3);
    assert!(text.contains("must be rectilinear") && !text.contains(scaled), "{text}");
    // Another codec has no spatial section.
    let mut h265 = ExportOptions { codec: ExportCodec::H265, ..Default::default() };
    assert!(!panel_text(&mut h265, ReframeAspect::Square, 120.0, 0.0).contains(scaled));
}
