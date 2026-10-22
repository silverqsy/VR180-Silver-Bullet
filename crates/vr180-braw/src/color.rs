//! The SDK's colour processing: what a clip carries (the probe's `color`
//! block — the current values with their ranges and accepted lists) and the
//! user's overrides on top of them.
//!
//! Every override is optional: `None` keeps the clip's own value. A fresh
//! helper gets the overrides as `--decode` flags ([`BrawColorParams::
//! extend_argv`]); a live zero-copy helper gets `P <flag> <value>` lines for
//! the EFFECTIVE values that changed ([`BrawColorValues::changed_lines`]),
//! so a cleared override sends the clip's own value back — both roads end at
//! the same picture.
//!
//! Values and ranges measured on a PYXIS 12K clip (SDK 6.0): colour science
//! Gen 5 of {4, 5, 6}; 17 gammas / 16 gamuts; tone curve contrast
//! 0.002–1.998, saturation 0–4, midpoint 0–1, highlights 0–1.998, shadows
//! 0.002–2, black level −1–1, white level 0–2; white balance 2000–50000 K,
//! tint −150–150 (the clip's own tint was 10, not 0), exposure ±5 stops,
//! 15 ISO steps 125–3200. The ranges did not change with the generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;

/// A value with the range the SDK accepts for it (`[value, min, max]` in
/// the probe's JSON).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(from = "[T; 3]")]
pub struct Ranged<T: Copy> {
    pub value: T,
    pub min: T,
    pub max: T,
}

impl<T: Copy> From<[T; 3]> for Ranged<T> {
    fn from(a: [T; 3]) -> Self { Self { value: a[0], min: a[1], max: a[2] } }
}

/// The tone-curve parameters a gamma is built from (the SDK's
/// `IBlackmagicRawToneCurve::GetToneCurve` for the camera and colour
/// science generation; `[contrast, saturation, midpoint, highlights,
/// shadows, black_level, white_level, video_black_level]` in the probe).
/// The SDK applies the `ToneCurve*` attributes only under the "Blackmagic
/// Design Custom" gamma, and Custom seeded with a gamma's preset reproduces
/// that gamma (Extended Video to within one code on the PYXIS), so an edit
/// starts from the current gamma's preset rather than the flat defaults.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(from = "[f32; 8]")]
pub struct TonePreset {
    pub contrast: f32,
    pub saturation: f32,
    pub midpoint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub black_level: f32,
    pub white_level: f32,
    pub video_black_level: bool,
}

impl From<[f32; 8]> for TonePreset {
    fn from(a: [f32; 8]) -> Self {
        Self { contrast: a[0], saturation: a[1], midpoint: a[2], highlights: a[3], shadows: a[4],
               black_level: a[5], white_level: a[6], video_black_level: a[7] != 0.0 }
    }
}

/// The clip's colour processing as the probe found it: the clip-level
/// attributes' current values, ranges and lists, and frame 0's white
/// balance, exposure and ISO.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BrawColorInfo {
    /// Blackmagic colour science generation (4, 5, 6 on SDK 6.0).
    pub color_science_gen: u16,
    #[serde(default)]
    pub color_science_gen_list: Vec<u16>,
    /// The gamma the clip decodes to ("Blackmagic Design Film" as shot).
    pub gamma: String,
    #[serde(default)]
    pub gamma_list: Vec<String>,
    /// The gamut ("Blackmagic Design" as shot).
    pub gamut: String,
    #[serde(default)]
    pub gamut_list: Vec<String>,
    pub contrast: Ranged<f32>,
    pub saturation: Ranged<f32>,
    pub midpoint: Ranged<f32>,
    pub highlights: Ranged<f32>,
    pub shadows: Ranged<f32>,
    pub black_level: Ranged<f32>,
    pub white_level: Ranged<f32>,
    /// 0 / 1 flags.
    pub video_black_level: u16,
    pub highlight_recovery: u16,
    pub gamut_compression: u16,
    /// Frame 0's white balance (Kelvin), tint and exposure (stops).
    pub wb_kelvin: Ranged<u32>,
    pub wb_tint: Ranged<i32>,
    pub exposure: Ranged<f32>,
    /// Frame 0's ISO and the camera's ISO steps.
    pub iso: u32,
    #[serde(default)]
    pub iso_list: Vec<u32>,
    /// Per-gamma tone-curve presets for this camera (see [`TonePreset`]);
    /// the gammas without one (Video, Rec.709, …) are absent.
    #[serde(default)]
    pub tone_presets: HashMap<String, TonePreset>,
}

/// The gamma under which the SDK applies the tone-curve attributes.
pub const CUSTOM_GAMMA: &str = "Blackmagic Design Custom";

impl BrawColorInfo {
    /// The values as shot (no override).
    pub fn as_shot(&self) -> BrawColorValues { BrawColorParams::default().resolve(self) }

    /// The tone curve a gamma is built from, when the SDK states one.
    pub fn tone_preset(&self, gamma: &str) -> Option<TonePreset> { self.tone_presets.get(gamma).copied() }

    /// The tone-curve values in effect under `gamma` before any override:
    /// the gamma's preset, else the clip's stored values (what the Custom
    /// gamma uses by default).
    pub fn tone_baseline(&self, gamma: &str) -> TonePreset {
        self.tone_preset(gamma).unwrap_or(TonePreset {
            contrast: self.contrast.value, saturation: self.saturation.value, midpoint: self.midpoint.value,
            highlights: self.highlights.value, shadows: self.shadows.value,
            black_level: self.black_level.value, white_level: self.white_level.value,
            video_black_level: self.video_black_level != 0,
        })
    }

    /// Whether `gamma` is the camera's Film log — what the clip's embedded
    /// display LUT ("Gen 5 Film to Extended Video") expects as its input.
    pub fn is_film_gamma(gamma: &str) -> bool { gamma.contains("Film") }
}

/// The user's overrides of the SDK's colour processing; `None` = as shot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrawColorParams {
    pub color_science_gen: Option<u16>,
    pub gamma: Option<String>,
    pub gamut: Option<String>,
    pub contrast: Option<f32>,
    pub saturation: Option<f32>,
    pub midpoint: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
    pub black_level: Option<f32>,
    pub white_level: Option<f32>,
    pub video_black_level: Option<bool>,
    pub highlight_recovery: Option<bool>,
    pub gamut_compression: Option<bool>,
    pub wb_kelvin: Option<u32>,
    pub wb_tint: Option<i32>,
    pub exposure: Option<f32>,
    pub iso: Option<u32>,
}

/// Effective values: the overrides laid over the clip's own.
#[derive(Debug, Clone, PartialEq)]
pub struct BrawColorValues {
    pub color_science_gen: u16,
    pub gamma: String,
    pub gamut: String,
    pub contrast: f32,
    pub saturation: f32,
    pub midpoint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub black_level: f32,
    pub white_level: f32,
    pub video_black_level: bool,
    pub highlight_recovery: bool,
    pub gamut_compression: bool,
    pub wb_kelvin: u32,
    pub wb_tint: i32,
    pub exposure: f32,
    pub iso: u32,
}

// The helper's `--decode` flags (also the keys of its zero-copy `P` command).
const F_CSGEN: &str = "--csgen";
const F_GAMMA: &str = "--gamma";
const F_GAMUT: &str = "--gamut";
const F_CONTRAST: &str = "--tc-contrast";
const F_SATURATION: &str = "--tc-saturation";
const F_MIDPOINT: &str = "--tc-midpoint";
const F_HIGHLIGHTS: &str = "--tc-highlights";
const F_SHADOWS: &str = "--tc-shadows";
const F_BLACK: &str = "--tc-black-level";
const F_WHITE: &str = "--tc-white-level";
const F_VIDEO_BLACK: &str = "--tc-video-black";
const F_HLR: &str = "--highlight-recovery";
const F_GAMUT_COMP: &str = "--gamut-compression";
const F_WB: &str = "--wb";
const F_TINT: &str = "--tint";
const F_EXPOSURE: &str = "--exposure";
const F_ISO: &str = "--iso";

fn flag_of(b: bool) -> String { if b { "1".into() } else { "0".into() } }

impl BrawColorParams {
    /// No override set.
    pub fn is_as_shot(&self) -> bool { *self == Self::default() }

    /// The gamma in effect for a clip (the override or the clip's own).
    pub fn effective_gamma<'a>(&'a self, info: &'a BrawColorInfo) -> &'a str {
        self.gamma.as_deref().unwrap_or(&info.gamma)
    }

    /// The effective values for `info`'s clip.
    pub fn resolve(&self, info: &BrawColorInfo) -> BrawColorValues {
        BrawColorValues {
            color_science_gen: self.color_science_gen.unwrap_or(info.color_science_gen),
            gamma: self.gamma.clone().unwrap_or_else(|| info.gamma.clone()),
            gamut: self.gamut.clone().unwrap_or_else(|| info.gamut.clone()),
            contrast: self.contrast.unwrap_or(info.contrast.value),
            saturation: self.saturation.unwrap_or(info.saturation.value),
            midpoint: self.midpoint.unwrap_or(info.midpoint.value),
            highlights: self.highlights.unwrap_or(info.highlights.value),
            shadows: self.shadows.unwrap_or(info.shadows.value),
            black_level: self.black_level.unwrap_or(info.black_level.value),
            white_level: self.white_level.unwrap_or(info.white_level.value),
            video_black_level: self.video_black_level.unwrap_or(info.video_black_level != 0),
            highlight_recovery: self.highlight_recovery.unwrap_or(info.highlight_recovery != 0),
            gamut_compression: self.gamut_compression.unwrap_or(info.gamut_compression != 0),
            wb_kelvin: self.wb_kelvin.unwrap_or(info.wb_kelvin.value),
            wb_tint: self.wb_tint.unwrap_or(info.wb_tint.value),
            exposure: self.exposure.unwrap_or(info.exposure.value),
            iso: self.iso.unwrap_or(info.iso),
        }
    }

    /// `(flag, value)` for each override set — the helper's `--decode` flags.
    pub fn argv_pairs(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if let Some(v) = self.color_science_gen { out.push((F_CSGEN, v.to_string())); }
        if let Some(v) = &self.gamma { out.push((F_GAMMA, v.clone())); }
        if let Some(v) = &self.gamut { out.push((F_GAMUT, v.clone())); }
        if let Some(v) = self.contrast { out.push((F_CONTRAST, format!("{v}"))); }
        if let Some(v) = self.saturation { out.push((F_SATURATION, format!("{v}"))); }
        if let Some(v) = self.midpoint { out.push((F_MIDPOINT, format!("{v}"))); }
        if let Some(v) = self.highlights { out.push((F_HIGHLIGHTS, format!("{v}"))); }
        if let Some(v) = self.shadows { out.push((F_SHADOWS, format!("{v}"))); }
        if let Some(v) = self.black_level { out.push((F_BLACK, format!("{v}"))); }
        if let Some(v) = self.white_level { out.push((F_WHITE, format!("{v}"))); }
        if let Some(v) = self.video_black_level { out.push((F_VIDEO_BLACK, flag_of(v))); }
        if let Some(v) = self.highlight_recovery { out.push((F_HLR, flag_of(v))); }
        if let Some(v) = self.gamut_compression { out.push((F_GAMUT_COMP, flag_of(v))); }
        if let Some(v) = self.wb_kelvin { out.push((F_WB, v.to_string())); }
        if let Some(v) = self.wb_tint { out.push((F_TINT, v.to_string())); }
        if let Some(v) = self.exposure { out.push((F_EXPOSURE, format!("{v}"))); }
        if let Some(v) = self.iso { out.push((F_ISO, v.to_string())); }
        out
    }

    pub(crate) fn extend_argv(&self, cmd: &mut Command) {
        for (flag, value) in self.argv_pairs() {
            cmd.arg(flag).arg(value);
        }
    }
}

impl BrawColorValues {
    /// Every value as a `(flag, value)` pair, in a fixed order.
    pub fn argv_pairs(&self) -> Vec<(&'static str, String)> {
        vec![
            (F_CSGEN, self.color_science_gen.to_string()),
            (F_GAMMA, self.gamma.clone()),
            (F_GAMUT, self.gamut.clone()),
            (F_CONTRAST, format!("{}", self.contrast)),
            (F_SATURATION, format!("{}", self.saturation)),
            (F_MIDPOINT, format!("{}", self.midpoint)),
            (F_HIGHLIGHTS, format!("{}", self.highlights)),
            (F_SHADOWS, format!("{}", self.shadows)),
            (F_BLACK, format!("{}", self.black_level)),
            (F_WHITE, format!("{}", self.white_level)),
            (F_VIDEO_BLACK, flag_of(self.video_black_level)),
            (F_HLR, flag_of(self.highlight_recovery)),
            (F_GAMUT_COMP, flag_of(self.gamut_compression)),
            (F_WB, self.wb_kelvin.to_string()),
            (F_TINT, self.wb_tint.to_string()),
            (F_EXPOSURE, format!("{}", self.exposure)),
            (F_ISO, self.iso.to_string()),
        ]
    }

    /// `P <flag> <value>` lines for the zero-copy helper's stdin: every value
    /// that differs from `prev` (what the helper currently has).
    pub fn changed_lines(&self, prev: &Self) -> Vec<String> {
        self.argv_pairs().into_iter().zip(prev.argv_pairs())
            .filter(|(now, before)| now.1 != before.1)
            .map(|((flag, value), _)| format!("P {flag} {value}"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> BrawColorInfo {
        serde_json::from_str(r#"{"color_science_gen": 5, "color_science_gen_list": [4, 5, 6],
            "gamma": "Blackmagic Design Film", "gamma_list": ["Blackmagic Design Film", "Rec.709"],
            "gamut": "Blackmagic Design", "gamut_list": ["Blackmagic Design", "Rec.709"],
            "contrast": [1, 0.002, 1.998], "saturation": [1, 0, 4], "midpoint": [0.5, 0, 1],
            "highlights": [1, 0, 1.998], "shadows": [1, 0.002, 2], "black_level": [0, -1, 1],
            "white_level": [1, 0, 2], "video_black_level": 0, "highlight_recovery": 0, "gamut_compression": 0,
            "wb_kelvin": [5600, 2000, 50000], "wb_tint": [10, -150, 150], "exposure": [0, -5, 5],
            "iso": 800, "iso_list": [125, 160, 200, 250, 320, 400, 500, 640, 800],
            "tone_presets": {"Blackmagic Design Film": [1, 1, 0.383562, 1, 1, 0, 1, 0],
                             "Blackmagic Design Extended Video": [1.58, 1.4, 0.409008, 1, 1.25, 0, 1, 0]}}"#).expect("parse")
    }

    #[test]
    fn tone_presets_and_baseline() {
        let i = info();
        let ev = i.tone_preset("Blackmagic Design Extended Video").expect("preset");
        assert_eq!((ev.contrast, ev.saturation, ev.shadows, ev.video_black_level), (1.58, 1.4, 1.25, false));
        assert!(i.tone_preset("Rec.709").is_none());
        // Without a preset the baseline is the clip's stored (flat) values.
        let b = i.tone_baseline(CUSTOM_GAMMA);
        assert_eq!((b.contrast, b.midpoint), (1.0, 0.5));
        assert_eq!(i.tone_baseline("Blackmagic Design Film").midpoint, 0.383562);
    }

    #[test]
    fn parses_the_probe_block() {
        let i = info();
        assert_eq!(i.wb_tint, Ranged { value: 10, min: -150, max: 150 });
        assert_eq!(i.contrast.max, 1.998);
        assert_eq!(i.gamma_list.len(), 2);
        assert!(BrawColorInfo::is_film_gamma(&i.gamma));
    }

    #[test]
    fn overrides_resolve_and_diff() {
        let i = info();
        let shot = i.as_shot();
        assert_eq!(shot.wb_tint, 10);
        let mut p = BrawColorParams::default();
        assert!(p.is_as_shot());
        assert!(p.argv_pairs().is_empty());
        p.exposure = Some(1.5);
        p.gamma = Some("Rec.709".into());
        assert_eq!(p.argv_pairs(), vec![("--gamma", "Rec.709".to_string()), ("--exposure", "1.5".to_string())]);
        let now = p.resolve(&i);
        let lines = now.changed_lines(&shot);
        assert_eq!(lines, vec!["P --gamma Rec.709".to_string(), "P --exposure 1.5".to_string()]);
        // Clearing an override sends the clip's own value back.
        p.exposure = None;
        let back = p.resolve(&i);
        assert_eq!(back.changed_lines(&now), vec!["P --exposure 0".to_string()]);
        assert!(back.changed_lines(&back).is_empty());
    }
}
