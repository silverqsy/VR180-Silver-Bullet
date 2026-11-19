// The Shadows / Highlights gain of every pixel of a raw frame's eye as its
// base gives it, in stops, before it is smoothed (`gain_cells.wgsl`, then
// the read in `raw_develop_local.wgsl`):
//
//   scene (ProPhoto, exposure applied) → its luminance's level
//   → base = the frame's bilateral grid at the pixel's place and own level
//     (where the grid holds nothing at that level, the local average)
//   → gains(base)
//
// The GPU twin of `raw_local::BaseGrid::base` and `LocalTone::gain_at`.
//
// Bindings:
//   (0) src   — the eye, camera-linear
//   (1) dst   — R32Float storage, same size
//   (2) p     — Params
//   (3) gains — log2 gain over the base level
//   (4) grid  — the smoothed bilateral grid (Rgba16Float, 3D)
//   (5) samp  — linear, clamped

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // xyz = camera values of the white (largest = 1); w = exposure gain
    neutral: vec4<f32>,
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    // x, y = a channel is trusted up to x and counts as clipped at y; z = luminance floor;
    // w = weight of the local average where the grid holds nothing
    clip: vec4<f32>,
    // xyz = luminance weights of the working space; w = grid cell in pixels of the eye
    luma: vec4<f32>,
    // x = the grid's first level, y = 1 / range
    grid: vec4<f32>,
    // `gains`: x = level of the first sample, y = samples per stop, z = last index
    gains: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> gains: array<f32>;
@group(0) @binding(4) var grid: texture_3d<f32>;
@group(0) @binding(5) var samp: sampler;

// `scene`: camera pixel -> scene RGB, clipped channels rebuilt (`raw_scene.wgsl`).
//#use SCENE

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
    let lit = max(scene(textureLoad(src, at, 0).rgb) * p.neutral.w, vec3<f32>(0.0));
    // The base: the grid at this pixel's place and own level.
    let level = log2(max(dot(p.luma.xyz, lit), p.clip.z));
    let dims = vec3<f32>(textureDimensions(grid));
    let v = vec3<f32>(
        (f32(gid.x) + 0.5) / p.luma.w + 0.5,
        (f32(gid.y) + 0.5) / p.luma.w + 0.5,
        clamp((level - p.grid.x) * p.grid.y + 0.5, 0.0, dims.z - 1.0));
    let g = textureSampleLevel(grid, samp, (v + vec3<f32>(0.5)) / dims, 0.0);
    let base = (g.x + p.clip.w * g.z) / (g.y + p.clip.w) + p.grid.x;
    textureStore(dst, at, vec4<f32>(stops_of(base), 0.0, 0.0, 1.0));
}
