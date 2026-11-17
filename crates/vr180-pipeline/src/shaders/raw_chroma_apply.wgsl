// Colour noise reduction, last pass: every pixel of the camera-linear
// picture takes the smoothed colour (the half-size picture, read
// bilinearly) at its OWN luminance.
//
//   w  = camera · balance              L = luma · w
//   s  = the smoothed values at the pixel
//   w' = s · L / (luma · s)            camera' = w' / balance
//
// Bindings:
//   (0) src      — the camera-linear picture
//   (1) dst      — Rgba16Unorm storage
//   (2) p        — Params
//   (3) smoothed — Rgba16Float, half size

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16unorm, write>;

struct Params {
    // xyz = what a camera value is multiplied by (1 / the camera's neutral)
    balance: vec4<f32>,
    // xyz = the luminance's weights; w = the darkest luminance a colour is taken of
    luma: vec4<f32>,
    // xy = the pass's direction; z = taps to each side; w = 1: back at the pixel's own luminance
    dir: vec4<i32>,
    // x = 1 / (2 tau²)
    range: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var smoothed: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(src, at, 0);
    let w = texel.rgb * p.balance.xyz;
    let level = max(dot(p.luma.xyz, w), p.luma.w);

    let last = vec2<i32>(textureDimensions(smoothed)) - vec2<i32>(1);
    let pos = (vec2<f32>(at) - vec2<f32>(0.5)) * 0.5;
    let base = floor(pos);
    let f = pos - base;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), last);
    let i1 = clamp(vec2<i32>(base) + vec2<i32>(1), vec2<i32>(0), last);
    let top = mix(textureLoad(smoothed, vec2<i32>(i0.x, i0.y), 0).rgb, textureLoad(smoothed, vec2<i32>(i1.x, i0.y), 0).rgb, f.x);
    let bottom = mix(textureLoad(smoothed, vec2<i32>(i0.x, i1.y), 0).rgb, textureLoad(smoothed, vec2<i32>(i1.x, i1.y), 0).rgb, f.x);
    let s = mix(top, bottom, f.y);

    let out = s * (level / max(dot(p.luma.xyz, s), p.luma.w)) / p.balance.xyz;
    textureStore(dst, at, vec4<f32>(clamp(out, vec3<f32>(0.0), vec3<f32>(1.0)), texel.a));
}
