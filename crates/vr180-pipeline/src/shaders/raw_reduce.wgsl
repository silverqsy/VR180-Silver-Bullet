// The reduced copy a raw frame's Shadows / Highlights base is built from:
// every output texel is log2 of the power mean of the scene luminance over
// its n × n pixels of the eye (scene = the same `scene` as
// `raw_develop.wgsl`, at the frame's exposure; luminance not below the
// floor).
//
// Bindings:
//   (0) src — the eye, camera-linear
//   (1) dst — R32Float storage, ceil(size / n)
//   (2) p   — Params

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // xyz = camera values of the white (largest = 1); w = exposure gain
    neutral: vec4<f32>,
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    // x, y = a channel is trusted up to x and counts as clipped at y; z = luminance floor; w = the mean's power
    clip: vec4<f32>,
    // xyz = luminance weights of the working space; w = pixels per texel (n)
    luma: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;

// `scene`: camera pixel -> scene RGB, clipped channels rebuilt (`raw_scene.wgsl`).
//#use SCENE

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
            let lit = max(scene(textureLoad(src, at, 0).rgb) * p.neutral.w, vec3<f32>(0.0));
            sum = sum + pow(max(dot(p.luma.xyz, lit), p.clip.z), p.clip.w);
        }
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(log2(sum / f32(n * n)) / p.clip.w, 0.0, 0.0, 1.0));
}
