use super::{takes_ilog, App, BatchItem, BatchStatus, ExportOptions, X6ClipInfo};
use crate::decoder::Settings;
use vr180_core::LogCurve;
use vr180_pipeline::SourceKind;

fn x6(ilog: Option<bool>) -> X6ClipInfo {
    X6ClipInfo { window: None, ilog, paraxial_native: 2086.97, single_lens: false, single_eye: false }
}

fn item(source_kind: SourceKind, x6: Option<X6ClipInfo>, ettr_stops: f32) -> BatchItem {
    BatchItem {
        path: "/nonexistent/in.insv".into(),
        fps: 29.97, duration_sec: 10.0, source_kind,
        fisheye_eye_w: 3840, fisheye_eye_h: 3840, width: 3840, height: 3840,
        settings: Settings { ettr_stops, ..Default::default() },
        status: BatchStatus::Idle, selected: false, x6, raw_one_lens: false, raw_frame: None, four_channel_sound: false, take: None, upside_down: None, offline: None, before_take: None, copy: None, take_failed: None, stand_in: None,
    }
}

#[test]
fn which_clips_are_ilog() {
    let insv = SourceKind::Insta360Insv;
    assert!(takes_ilog(insv, Some(&x6(Some(true)))));
    // A file that does not state its profile, or has no readable
    // trailer at all, is taken as I-Log.
    assert!(takes_ilog(insv, Some(&x6(None))));
    assert!(takes_ilog(insv, None));
    assert!(!takes_ilog(insv, Some(&x6(Some(false)))));
    for other in [SourceKind::DjiOsv, SourceKind::GoProEac, SourceKind::SbsFisheye, SourceKind::Insta360Dng] {
        assert!(!takes_ilog(other, None), "{other:?}");
    }
}

/// The amount a clip was shot over comes back out as that many stops
/// down, in the I-Log curve, and only for a clip that is I-Log.
#[test]
fn the_recovery_reaches_the_export_of_an_ilog_clip_only() {
    let insv = SourceKind::Insta360Insv;
    for stops in [1.0_f32, 2.0, 3.0] {
        let plan = item(insv, Some(x6(Some(true))), stops).export_settings().build_color_stack();
        let e = plan.log_exposure.expect("the stage");
        assert_eq!((e.stops, e.curve), (-stops, LogCurve::INSTA360_ILOG));
        assert!(plan.any_active());
    }
    // Off is no stage at all.
    assert!(item(insv, Some(x6(Some(true))), 0.0).export_settings().build_color_stack().log_exposure.is_none());
    // A value that reached a clip of another profile or camera does nothing.
    for it in [item(insv, Some(x6(Some(false))), 2.0), item(SourceKind::DjiOsv, None, 2.0)] {
        assert!(it.export_settings().build_color_stack().log_exposure.is_none(), "{:?}", it.source_kind);
    }
    // A single-eye clip is graded the same way.
    let one = BatchItem { x6: Some(X6ClipInfo { single_eye: true, ..x6(Some(true)) }), ..item(insv, None, 2.0) };
    assert!(one.export_settings().single_eye);
    assert_eq!(one.export_settings().build_color_stack().log_exposure.map(|e| e.stops), Some(-2.0));

    // ...and the export itself carries it.
    let it = item(insv, Some(x6(Some(true))), 2.0);
    let cfg = App::build_export_cfg(
        &it.path, std::path::Path::new("/nonexistent/out.mov"),
        it.fps, it.source_kind, it.width, it.height, it.fisheye_eye_w, it.fisheye_eye_h, None,
        &it.export_settings(), &ExportOptions::default(), Vec::new(),
    );
    assert_eq!(cfg.color_stack.log_exposure.map(|e| e.stops), Some(-2.0));
}

/// Whether the source is I-Log is told to the copy a renderer works
/// from; the settings as kept and saved never say so.
#[test]
fn the_stored_settings_carry_the_amount_and_not_the_source() {
    let s = Settings { ettr_stops: 2.0, ..Default::default() };
    assert!(s.build_color_stack().log_exposure.is_none());
    let json = serde_json::to_string(&s).expect("json");
    assert!(json.contains("\"ettr_stops\":2.0") && !json.contains("ilog_source"), "{json}");
    let back: Settings = serde_json::from_str(&json).expect("parse");
    assert_eq!((back.ettr_stops, back.ilog_source), (2.0, false));
    // A file from before the setting existed: off.
    let old: Settings = serde_json::from_str("{}").expect("parse");
    assert_eq!(old.ettr_stops, 0.0);
    // A linear DNG has no place for it.
    let dng = Settings { ettr_stops: 2.0, ilog_source: true, still_dng_mode: true, ..Default::default() };
    assert!(dng.build_color_stack().log_exposure.is_none());
}

/// The tone controls reach a clip's colour stack as one stage in linear
/// light: the scene's where the clip is I-Log, with the exposure it was
/// shot over taken out by the same stage; the light the picture shows
/// for any other clip. Nothing moved is no stage, and the clip's stack
/// is what it was before the controls existed.
#[test]
fn the_tone_controls_reach_a_clip_as_one_stage_in_linear_light() {
    use vr180_pipeline::tone_linear::{ToneLight, ToneSettings};
    let insv = SourceKind::Insta360Insv;
    let tone = ToneSettings { highlights: -60.0, shadows: 30.0, ..Default::default() };
    let with = |mut it: BatchItem| { it.settings.tone = tone; it };

    // Nothing moved: no stage, and the recovery is the stage it was.
    let idle = item(insv, Some(x6(Some(true))), 2.0).export_settings().build_color_stack();
    assert!(idle.tone.is_none() && idle.active_tone().is_none());
    assert_eq!(idle.log_exposure.map(|e| e.stops), Some(-2.0));
    assert!(!Settings::default().build_color_stack().any_active());

    // An I-Log clip: the scene's light, the recovery in the same stage.
    let plan = with(item(insv, Some(x6(Some(true))), 2.0)).export_settings().build_color_stack();
    let stage = plan.tone.expect("the stage");
    assert_eq!((stage.settings, stage.light, stage.pre_stops), (tone, ToneLight::Scene(LogCurve::INSTA360_ILOG), -2.0));
    assert!(plan.log_exposure.is_none() && plan.any_active());
    let plan = with(item(insv, Some(x6(Some(true))), 0.0)).export_settings().build_color_stack();
    assert_eq!(plan.tone.map(|t| t.pre_stops), Some(0.0));

    // Any other clip: the light its picture shows, and no recovery.
    for it in [item(insv, Some(x6(Some(false))), 2.0), item(SourceKind::DjiOsv, None, 0.0), item(SourceKind::GoProEac, None, 0.0)] {
        let plan = with(it).export_settings().build_color_stack();
        let stage = plan.tone.expect("the stage");
        assert_eq!((stage.light, stage.pre_stops), (ToneLight::Shown, 0.0));
        assert!(plan.log_exposure.is_none());
    }

    // A raw clip has the controls in its development, a linear DNG no
    // place for them.
    let raw = Settings { tone, raw_source: true, ..Default::default() }.build_color_stack();
    assert!(raw.tone.is_none() && raw.raw_develop.is_some());
    assert!(Settings { tone, still_dng_mode: true, ..Default::default() }.build_color_stack().tone.is_none());

    // The export carries the stage, and waits for each frame's levels.
    let it = with(item(insv, Some(x6(Some(true))), 1.0));
    let cfg = App::build_export_cfg(
        &it.path, std::path::Path::new("/nonexistent/out.mov"),
        it.fps, it.source_kind, it.width, it.height, it.fisheye_eye_w, it.fisheye_eye_h, None,
        &it.export_settings(), &ExportOptions::default(), Vec::new(),
    );
    let stage = cfg.color_stack.tone.expect("the stage");
    assert_eq!((stage.pre_stops, stage.measure), (-1.0, vr180_pipeline::raw_tone::Measure::Wait));

    // Kept with the clip's settings; a file from before they existed has none.
    let json = serde_json::to_string(&it.settings).expect("json");
    let back: Settings = serde_json::from_str(&json).expect("parse");
    assert_eq!(back.tone, tone);
    let old: Settings = serde_json::from_str("{}").expect("parse");
    assert!(old.tone.is_identity());
    // A single-eye clip is toned the same way.
    let one = with(BatchItem { x6: Some(X6ClipInfo { single_eye: true, ..x6(Some(true)) }), ..item(insv, None, 0.0) });
    assert_eq!(one.export_settings().build_color_stack().tone.map(|t| t.settings), Some(tone));
}
