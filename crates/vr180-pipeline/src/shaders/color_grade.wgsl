// Color grade: temperature + tint + saturation, fused per-pixel.
//
// Port of `apply_temp_tint` (vr180_gui.py:6402-6422) and the saturation
// step from `apply_export_post` (vr180_gui.py:7776-7779).
//
// Temperature / tint operate as a multiplicative per-channel scale.
// Python uses BGR order (legacy from cv2); we use RGB. The mapping:
//
//   Python `scales[BGR]` = [1 - 0.30*temp, 1 - 0.30*tint, 1 + 0.30*temp]
//                          ----B----------- ----G--------- ----R-----------
//
// In RGB order that becomes:
//
//   scales[RGB] = [1 + 0.30*temp, 1 - 0.30*tint, 1 - 0.30*temp]
//
// Positive temperature warms (more red, less blue); positive tint
// pushes toward magenta (less green). The 0.30 strength matches the
// Python knob feel — gives the user a [-1..+1] slider with a useful
// range without going completely off-white.
//
// Saturation uses the BT.601 grayscale (cv2's `cvtColor(BGR2GRAY)`):
//
//   luma = 0.114*B + 0.587*G + 0.299*R
//   result = luma * (1 - sat) + rgb * sat
//
// sat=0 → grayscale, sat=1 → unchanged, sat>1 → boosted saturation
// (potentially clipping per-channel).
//
// Order matters: temp/tint applied BEFORE saturation, because the
// Python pipeline applies saturation as the final color op (after
// mid-detail), and temp/tint changes the channel ratios that feed
// the luma calculation.

struct ColorGradeUniforms {
    temperature: f32,
    tint:        f32,
    saturation:  f32,
    _pad: f32,
};

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: ColorGradeUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    var px = textureLoad(in_tex, vec2<i32>(gid.xy), 0);

    // 1. Temp / tint — per-channel multiplicative.
    let r_scale = 1.0 + 0.30 * u.temperature;
    let g_scale = 1.0 - 0.30 * u.tint;
    let b_scale = 1.0 - 0.30 * u.temperature;
    var rgb = vec3<f32>(px.r * r_scale, px.g * g_scale, px.b * b_scale);

    // 2. Saturation — BT.601 luma + linear blend.
    let luma = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    let gray = vec3<f32>(luma);
    rgb = mix(gray, rgb, u.saturation);

    // Clamp to 0..1 — temp/tint can push individual channels out of
    // range; we clip rather than normalize so highlights stay highlights.
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, px.a));
}
