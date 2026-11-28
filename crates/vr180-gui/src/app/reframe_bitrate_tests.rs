use super::ReframeSize;
use crate::decoder::ReframeAspect;

#[test]
fn size_follows_what_the_view_offers() {
    use ReframeAspect::{Square, Wide};
    use ReframeSize::*;
    // The 1:1 view's 3840² per eye, for video and stills.
    assert_eq!(P3840.eye_dims(Square), (3840, 3840));
    assert!(ReframeSize::choices(false, Square).contains(&P3840));
    assert!(!ReframeSize::choices(false, Wide).contains(&P3840));
    assert_eq!(P3840.clamped(false, Square), P3840);
    assert_eq!(P3840.clamped(true, Square), P3840);
    // A size the view does not offer drops to the largest one below it.
    assert_eq!(P3840.clamped(false, Wide), P2160);
    assert_eq!(P3840.clamped(true, Wide), P3240);
    assert_eq!(P4320.clamped(false, Square), P3840);
    assert_eq!(P4320.clamped(true, Wide), P3240);
    assert_eq!(P3240.clamped(false, Square), P2160);
    assert_eq!(P3240.clamped(false, Wide), P2160);
    // Offered sizes stay as they are.
    for &(still, aspect) in &[(false, Square), (false, Wide), (true, Square), (true, Wide)] {
        for &sz in ReframeSize::choices(still, aspect) {
            assert_eq!(sz.clamped(still, aspect), sz);
        }
    }
}

#[test]
fn recommended_rates_match_the_table() {
    let cases = [
        (ReframeSize::P1080, ReframeAspect::Square, 30.0, 20),
        (ReframeSize::P1080, ReframeAspect::Wide,   30.0, 35),
        (ReframeSize::P1440, ReframeAspect::Square, 30.0, 35),
        (ReframeSize::P1440, ReframeAspect::Wide,   30.0, 60),
        (ReframeSize::P2160, ReframeAspect::Square, 30.0, 80),
        (ReframeSize::P2160, ReframeAspect::Wide,   30.0, 140),
        (ReframeSize::P1080, ReframeAspect::Wide,   60.0, 50),
        (ReframeSize::P2160, ReframeAspect::Wide,   60.0, 210),
        (ReframeSize::P1080, ReframeAspect::Square, 24.0, 20), // flat below 30 fps
        (ReframeSize::P3840, ReframeAspect::Square, 30.0, 250),
        (ReframeSize::P3840, ReframeAspect::Square, 60.0, 300), // the slider's ceiling
    ];
    for (sz, aspect, fps, want) in cases {
        assert_eq!(sz.recommended_bitrate_mbps(aspect, fps), want, "{sz:?} {aspect:?} @ {fps}");
    }
    // Always inside the slider's range.
    for sz in [ReframeSize::P1080, ReframeSize::P1440, ReframeSize::P2160, ReframeSize::P3840] {
        for fps in [23.976_f32, 30.0, 50.0, 59.94, 120.0] {
            let r = sz.recommended_bitrate_mbps(ReframeAspect::Wide, fps);
            assert!((10..=300).contains(&r) && r % 5 == 0, "{sz:?} @ {fps}: {r}");
        }
    }
}
