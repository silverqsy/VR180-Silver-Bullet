use super::bitrate_label;

#[test]
fn bitrate_text() {
    // Dual-stream: the total, with each eye's stream in the info grid.
    let dual = [126_637_340u64, 126_014_114];
    assert_eq!(bitrate_label(&dual, Some(257_000_000), true).as_deref(), Some("252.7 Mbps (126.6 + 126.0)"));
    assert_eq!(bitrate_label(&dual, Some(257_000_000), false).as_deref(), Some("252.7 Mbps"));
    // One stream: no brackets.
    assert_eq!(bitrate_label(&[98_400_000], None, true).as_deref(), Some("98.4 Mbps"));
    // The container states none: the file's size over its duration, marked.
    assert_eq!(bitrate_label(&[], Some(1_234_560_000), true).as_deref(), Some("≈ 1234.6 Mbps"));
    // Nothing known (a still).
    assert_eq!(bitrate_label(&[], None, true), None);
}

#[test]
fn dropped_frames_text() {
    use super::{dropped_frames_hover, dropped_frames_label, dropped_hole_text, DROPPED_LISTED};
    use vr180_pipeline::frame_timeline::Hole;
    let hole = |row: usize, at_s: f64, frames: usize| Hole { row, at_s, frames };
    // None missing.
    assert_eq!((dropped_frames_label(&[], false), dropped_frames_label(&[], true)), ("none".to_string(), "无".to_string()));
    assert!(dropped_frames_hover(&[], false).starts_with("No frame is missing"));
    // One place, several: the frames are summed.
    let one = [hole(10, 0.2, 1)];
    assert_eq!((dropped_frames_label(&one, false), dropped_frames_label(&one, true)), ("1 in 1 place".to_string(), "1 帧，1 处".to_string()));
    let three = [hole(10, 0.2, 1), hole(700, 12.5, 3), hole(4000, 75.25, 5)];
    assert_eq!((dropped_frames_label(&three, false), dropped_frames_label(&three, true)), ("9 in 3 places".to_string(), "9 帧，3 处".to_string()));
    // A place: when and how many.
    assert_eq!(dropped_hole_text(&three[0], false), "00:00.20: 1 frame dropped");
    assert_eq!(dropped_hole_text(&three[2], false), "01:15.25: 5 frames dropped");
    assert_eq!(dropped_hole_text(&three[1], true), "00:12.50：丢 3 帧");
    // The hover text lists the places, each on its own line, after what they are.
    let text = dropped_frames_hover(&three, false);
    assert_eq!(text.lines().count(), 4);
    assert!(text.lines().nth(2) == Some("00:12.50: 3 frames dropped"));
    // More places than it lists: the rest are counted.
    let many: Vec<Hole> = (0..DROPPED_LISTED + 7).map(|i| hole(i * 10, i as f64, 1)).collect();
    let text = dropped_frames_hover(&many, true);
    assert_eq!(text.lines().count(), 1 + DROPPED_LISTED + 1);
    assert_eq!(text.lines().last(), Some("……另有 7 处"));
}

#[test]
fn a_frame_as_the_sensor_recorded_it_goes_where_its_snapshot_does() {
    use super::{unused_frame_name, original_frame_saved, saved_frame_place};
    use std::path::Path;
    // Beside the clip, or in the export folder when one is set, under
    // the clip's name and the frame's number.
    let take = Path::new("/media/card/RAWVIDEO-20261118-154401");
    assert_eq!(saved_frame_place(take, None, 5181), (Path::new("/media/card").to_path_buf(), "RAWVIDEO-20261118-154401_frame5181".to_string()));
    assert_eq!(saved_frame_place(take, Some(Path::new("/out")), 0), (Path::new("/out").to_path_buf(), "RAWVIDEO-20261118-154401_frame0".to_string()));
    assert_eq!(saved_frame_place(Path::new("/clips/VID_0042.insv"), None, 7).1, "VID_0042_frame7");
    // A name that is taken gives way to the next one.
    let taken = ["A_frame5", "A_frame5-2"];
    assert_eq!(unused_frame_name("A_frame4", |n| taken.contains(&n)), "A_frame4");
    assert_eq!(unused_frame_name("A_frame5", |n| taken.contains(&n)), "A_frame5-3");
    // The files: the frame's name, then the channel.
    let pair = |files: Vec<(usize, u64)>| vr180_pipeline::sensor_dng::WrittenPair { request: 5324, exposure_ns: [1, 2], signature: String::new(), files };
    assert_eq!(
        original_frame_saved(&pair(vec![(0, 10), (1, 11)]), "RAWVIDEO-20261118-154401_frame5181"),
        "RAWVIDEO-20261118-154401_frame5181_CAM0.dng + _CAM1.dng");
    assert_eq!(original_frame_saved(&pair(vec![(1, 11)]), "TAKE_frame3-2"), "TAKE_frame3-2_CAM1.dng");
}

/// A frame the sensor recorded takes the next name while a file of
/// either channel is in the folder.
#[test]
fn a_frame_as_the_sensor_recorded_it_takes_an_unused_name() {
    use super::unused_original_name;
    let dir = std::env::temp_dir().join(format!("vr180-original-name-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    assert_eq!(unused_original_name(&dir, "T_frame9"), "T_frame9");
    std::fs::write(dir.join("T_frame9_CAM1.dng"), b"x").unwrap();
    assert_eq!(unused_original_name(&dir, "T_frame9"), "T_frame9-2");
    std::fs::write(dir.join("T_frame9-2_CAM0.dng"), b"x").unwrap();
    assert_eq!(unused_original_name(&dir, "T_frame9"), "T_frame9-3");
    // The snapshot's own files of the frame do not take the name.
    std::fs::write(dir.join("T_frame8.jpg"), b"x").unwrap();
    std::fs::write(dir.join("T_frame8.dng"), b"x").unwrap();
    assert_eq!(unused_original_name(&dir, "T_frame8"), "T_frame8");
    let _ = std::fs::remove_dir_all(&dir);
}
