// The reduced copy the Shadows / Highlights base of a frame is built from,
// for a picture that arrives as code values (`tone_linear.rs`): every output
// texel is the mean of log2(luminance) over its n × n pixels of the eye, the
// light being the code through the `decode` table, at the frame's exposure.
// The counterpart of `raw_reduce.wgsl`; the grid is then built and smoothed
// by the same passes (`raw_grid_build.wgsl`, `raw_grid_blur.wgsl`).
//
// Bindings:
//   (0) src    — the eye, code values
//   (1) dst    — R32Float storage, ceil(size / n)
//   (2) p      — Params
//   (3) decode — light over the code

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // xyz = luminance weights; w = pixels per texel (n)
    luma: vec4<f32>,
    // x = last index of `decode`; y = exposure gain; z = luminance floor
    light: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> decode: array<f32>;

fn light1(code: f32) -> f32 {
    let y = clamp(code, 0.0, 1.0) * p.light.x;
    let i = min(floor(y), p.light.x - 1.0);
    let f = y - i;
    let k = u32(i);
    return decode[k] * (1.0 - f) + decode[k + 1u] * f;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    let n = i32(p.luma.w);
    let origin = vec2<i32>(i32(gid.x), i32(gid.y)) * n;
    var sum = 0.0;
    for (var j: i32 = 0; j < n; j = j + 1) {
        for (var i: i32 = 0; i < n; i = i + 1) {
            let at = min(origin + vec2<i32>(i, j), last);
            let code = textureLoad(src, at, 0).rgb;
            let lit = max(vec3<f32>(light1(code.r), light1(code.g), light1(code.b)) * p.light.y, vec3<f32>(0.0));
            sum = sum + log2(max(dot(p.luma.xyz, lit), p.light.z));
        }
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sum / f32(n * n), 0.0, 0.0, 1.0));
}
