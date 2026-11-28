use super::{camera_raw_status, draw_raw_develop_section, tr_zh};
use std::path::PathBuf;
use vr180_pipeline::raw_export::ReadDevelop;
use vr180_pipeline::x6_raw::CameraRawSettings;

/// What a read came to is said in one line: the file, what stayed in
/// Camera Raw, and a word when the two eyes were left differently.
#[test]
fn the_outcome_of_a_read_is_said_under_its_button() {
    assert!(camera_raw_status(&None).starts_with("No Camera Raw settings found"));
    let found = |not_carried: Vec<(&'static str, String)>, eyes_differ: bool| Some(CameraRawSettings {
        file: PathBuf::from("/takes/TAKE_OriginalDNG/CAM0/000000364.dng"),
        develop: ReadDevelop { settings: Default::default(), not_carried },
        eyes_differ,
    });
    assert_eq!(camera_raw_status(&found(Vec::new(), false)), "Read from CAM0/000000364.dng");
    let text = camera_raw_status(&found(vec![("Texture", "+20".into()), ("Tone curve", String::new()), ("Profile", "Adobe Vivid".into())], true));
    assert_eq!(text, "Read from CAM0/000000364.dng. Not carried over: Texture +20, Tone curve, Profile Adobe Vivid. \
            The other eye's file has other settings: these are channel 0's.");
    // Every name a read can come back with has its Chinese.
    for name in ["Texture", "Clarity", "Dehaze", "Vibrance", "Saturation", "Luminance noise reduction", "Color noise reduction detail",
        "Color noise reduction smoothness", "Grain", "Tone curve", "Color mixer", "Color grading", "Calibration", "Vignette",
        "Lens corrections", "Geometry", "Crop", "Black & white", "Masks", "Healing", "Profile", "Read from Camera Raw", "Not carried over:"] {
        assert_ne!(tr_zh(name), name, "{name} has no translation");
    }
}

/// The raw development section has the button, shows the line under it,
/// and leaves the settings alone until something is pressed.
#[test]
fn the_raw_section_offers_the_read() {
    fn collect(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(text) => { out.push_str(text.galley.text()); out.push('\n'); }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
            _ => {}
        }
    }
    let ctx = egui::Context::default();
    let mut raw = vr180_pipeline::raw_tone::RawSettings { shadows: 30.0, ..Default::default() };
    let before = raw;
    let mut text = String::new();
    let mut pressed = false;
    for _ in 0..2 {
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(700.0, 2000.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| {
            pressed |= draw_raw_develop_section(ui, &mut raw, vr180_pipeline::raw_color::RawCamera::X6, Some("Read from CAM0/000000364.dng"), true, super::RawAutoShown::Off);
        });
        text.clear();
        out.shapes.iter().for_each(|clipped| collect(&clipped.shape, &mut text));
    }
    assert!(text.contains("Read from Camera Raw") && text.contains("Reset raw development"), "{text}");
    assert!(text.contains("Read from CAM0/000000364.dng"), "{text}");
    // Drawing the section changes nothing: a white balance left as shot
    // stays as shot (it used to become a custom one of the same kelvin).
    assert!(!pressed, "nothing was pressed");
    assert_eq!(raw, before);
    // The automatic exposure: offered, its two controls only while it
    // is on, and what it is doing said under them.
    assert!(text.contains("Auto exposure") && !text.contains("Smoothness (s)") && !text.contains("Over the clip:"), "{text}");
    let mut auto = vr180_pipeline::raw_tone::RawSettings { auto_exposure: true, ..Default::default() };
    let before = auto;
    for (shown, says) in [
        (super::RawAutoShown::Metering(None), "Metering the clip…"),
        (super::RawAutoShown::Metering(Some(0.431)), "Metering the clip… 43 %"),
        (super::RawAutoShown::Now { stops: 1.7, range: (1.51, 2.72) }, "Over the clip: +1.51 … +2.72"),
    ] {
        for _ in 0..2 {
            let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(700.0, 2000.0))), ..Default::default() };
            let out = ctx.run_ui(input, |ui| {
                draw_raw_develop_section(ui, &mut auto, vr180_pipeline::raw_color::RawCamera::X6, None, true, shown);
            });
            text.clear();
            out.shapes.iter().for_each(|clipped| collect(&clipped.shape, &mut text));
        }
        assert!(text.contains("EV") && text.contains("Smoothness (s)") && text.contains(says), "{shown:?}: {text}");
    }
    // Shown, the frame's exposure is not written into the setting.
    assert_eq!(auto, before);
    for name in ["Auto exposure", "Smoothness (s)", "Metering the clip…", "Over the clip:"] {
        assert_ne!(tr_zh(name), name, "{name} has no translation");
    }
}
