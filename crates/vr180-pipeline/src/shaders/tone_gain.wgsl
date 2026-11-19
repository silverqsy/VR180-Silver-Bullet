// The Shadows / Highlights gain of every pixel of an eye as its base gives
// it, in stops, before it is smoothed, for a picture that arrives as code
// values (`tone_linear.rs`): the counterpart of `raw_gain.wgsl`.
//
//   codes → light (the `decode` table), × the exposure → its level
//   → base = the frame's bilateral grid at the pixel's place and own level
//     (where the grid holds nothing at that level, the local average)
//   → gains(base)
//
// Bindings:
//   (0) src    — the eye, code values
//   (1) dst    — R32Float storage, same size
//   (2) p      — Params
//   (3) decode — light over the code
//   (4) gains  — log2 gain over the base level
//   (5) grid   — the smoothed bilateral grid (Rgba16Float, 3D)
//   (6) samp   — linear, clamped

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // x = last index of `decode`; y = exposure gain; z = luminance floor;
    // w = weight of the local average where the grid holds nothing
    light: vec4<f32>,
    // xyz = luminance weights; w = grid cell in pixels of the eye
    luma: vec4<f32>,
    // `gains`: x = level of the first sample, y = samples per stop, z = last index
    gains: vec4<f32>,
    // x = the grid's first level, y = 1 / range
    grid: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> decode: array<f32>;
@group(0) @binding(4) var<storage, read> gains: array<f32>;
@group(0) @binding(5) var grid: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;

fn light1(code: f32) -> f32 {
    let y = clamp(code, 0.0, 1.0) * p.light.x;
    let i = min(floor(y), p.light.x - 1.0);
    let f = y - i;
    let k = u32(i);
    return decode[k] * (1.0 - f) + decode[k + 1u] * f;
}

// The gain of a base level, in stops.
fn stops_of(base: f32) -> f32 {
    let y = clamp((base - p.gains.x) * p.gains.y, 0.0, p.gains.z);
    let i = min(floor(y), p.gains.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return gains[k] * (1.0 - f) + gains[k + 1u] * f;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let code = textureLoad(src, at, 0).rgb;
    // As the grid was gathered: a channel below no light counts as none.
    let lit = max(vec3<f32>(light1(code.r), light1(code.g), light1(code.b)) * p.light.y, vec3<f32>(0.0));
    let level = log2(max(dot(p.luma.xyz, lit), p.light.z));
    let dims = vec3<f32>(textureDimensions(grid));
    let v = vec3<f32>(
        (f32(gid.x) + 0.5) / p.luma.w + 0.5,
        (f32(gid.y) + 0.5) / p.luma.w + 0.5,
        clamp((level - p.grid.x) * p.grid.y + 0.5, 0.0, dims.z - 1.0));
    let g = textureSampleLevel(grid, samp, (v + vec3<f32>(0.5)) / dims, 0.0);
    let base = (g.x + p.light.w * g.z) / (g.y + p.light.w) + p.grid.x;
    textureStore(dst, at, vec4<f32>(stops_of(base), 0.0, 0.0, 1.0));
}
