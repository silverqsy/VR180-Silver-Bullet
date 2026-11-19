// Colour noise reduction, first pass: the camera-linear picture, white
// balanced, at half size (the mean of each 2×2; the last row or column
// repeats on an odd side).
//
// Bindings:
//   (0) src — the camera-linear picture
//   (1) dst — Rgba16Float storage, half size: the balanced values
//   (2) p   — Params (as raw_chroma_apply.wgsl)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

struct Params {
    // xyz = what a camera value is multiplied by (1 / the camera's neutral)
    balance: vec4<f32>,
    // xyz = the luminance's weights; w = the darkest luminance a colour is taken of
    luma: vec4<f32>,
    // xy = the pass's direction; z = taps to each side; w = 1: back at the pixel's own luminance
    dir: vec4<i32>,
    // x = 1 / (2 tau²); y = 1 / (2 s²), s the stops of luminance a neighbour may be away
    range: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    let at = 2 * vec2<i32>(i32(gid.x), i32(gid.y));
    let sum = textureLoad(src, min(at, last), 0).rgb
            + textureLoad(src, min(at + vec2<i32>(1, 0), last), 0).rgb
            + textureLoad(src, min(at + vec2<i32>(0, 1), last), 0).rgb
            + textureLoad(src, min(at + vec2<i32>(1, 1), last), 0).rgb;
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.25 * sum * p.balance.xyz, 1.0));
}
