// "BeyondVR Hack" — final output-stage per-eye scale.
//
// Scales the image about each eye's own center by `u.scale` (< 1.0
// shrinks), filling the revealed border with opaque black. `u.eye_w`
// makes the same shader serve both layouts:
//
//   per-eye texture   → eye_w = texture width  (one half, center = w/2)
//   full SBS texture  → eye_w = width / 2      (each half about its own
//                                               half-center)
//
// MUST run as the LAST stage of the color stack: the padding is written
// after grading, so a LUT / lift can never tint it away from true black.
//
// Bilinear 4-tap via textureLoad (the input is a storage-written
// intermediate with no sampler bound in the PerPixelPipeline layout).
// Taps are clamped inside the eye's own half so nothing bleeds across
// the SBS seam.

struct EyeScaleUniforms {
    scale: f32,
    eye_w: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var in_tex:  texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: EyeScaleUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let h = f32(dims.y);
    let eye_w = u.eye_w;
    // Which half is this pixel in, and its x-offset in the texture.
    let x_off = floor((f32(gid.x) + 0.5) / eye_w) * eye_w;

    // Inverse map about the half's center (pixel-center convention).
    let dst = vec2<f32>(f32(gid.x) + 0.5 - x_off, f32(gid.y) + 0.5);
    let center = vec2<f32>(0.5 * eye_w, 0.5 * h);
    let src = center + (dst - center) / max(u.scale, 1.0e-4);

    // Outside the source image → black padding.
    if src.x < 0.0 || src.y < 0.0 || src.x > eye_w || src.y > h {
        textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // 4-tap bilinear in texel space, taps clamped inside this half.
    let p = clamp(src - vec2<f32>(0.5), vec2<f32>(0.0),
                  vec2<f32>(eye_w - 1.0, h - 1.0));
    let p0 = floor(p);
    let f = p - p0;
    let x0 = i32(p0.x + x_off);
    let y0 = i32(p0.y);
    let x1 = min(x0 + 1, i32(x_off + eye_w) - 1);
    let y1 = min(y0 + 1, i32(dims.y) - 1);
    let c00 = textureLoad(in_tex, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(in_tex, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(in_tex, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(in_tex, vec2<i32>(x1, y1), 0);
    let px = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(px.rgb, 1.0));
}
