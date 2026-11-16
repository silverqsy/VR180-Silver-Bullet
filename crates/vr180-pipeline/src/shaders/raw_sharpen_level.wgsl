// Raw sharpening, first pass: the developed picture's luminance as the
// level the sharpening works on, phi = Y^(1/gamma), and (entry `edge`) the
// size of that level's gradient, which Masking reads.
//
// Bindings:
//   (0) src — `level`: the developed picture (sRGB-encoded Rec.709);
//             `edge`: the level plane
//   (1) dst — R32Float
//   (2) p   — Params

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // xyz = luminance weights; w = 1 / gamma
    luma: vec4<f32>,
    // x = the darkest luminance a level is taken of
    floor: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;

fn decode(e: vec3<f32>) -> vec3<f32> {
    let lo = e / 12.92;
    let hi = pow((max(e, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, e <= vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8)
fn level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let y = max(dot(p.luma.xyz, decode(textureLoad(src, at, 0).rgb)), p.floor.x);
    textureStore(dst, at, vec4<f32>(pow(y, p.luma.w), 0.0, 0.0, 1.0));
}

// Central differences; nothing on the picture's border.
@compute @workgroup_size(8, 8)
fn edge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(dst));
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    if (at.x >= size.x || at.y >= size.y) { return; }
    let l = textureLoad(src, vec2<i32>(max(at.x - 1, 0), at.y), 0).r;
    let r = textureLoad(src, vec2<i32>(min(at.x + 1, size.x - 1), at.y), 0).r;
    let u = textureLoad(src, vec2<i32>(at.x, max(at.y - 1, 0)), 0).r;
    let d = textureLoad(src, vec2<i32>(at.x, min(at.y + 1, size.y - 1)), 0).r;
    let inside = vec2<f32>(
        select(0.0, 0.5, at.x > 0 && at.x < size.x - 1),
        select(0.0, 0.5, at.y > 0 && at.y < size.y - 1));
    let g = vec2<f32>(r - l, d - u) * inside;
    textureStore(dst, at, vec4<f32>(length(g), 0.0, 0.0, 1.0));
}
