// The reduced copy a raw frame's tone keys are read from: every output
// cell is the mean scene RGB (linear ProPhoto, exposure 0, clipped channels
// rebuilt — the same `scene` as `raw_develop.wgsl`) of its part of the eye.
//
// Bindings:
//   (0) src — the eye, camera-linear
//   (1) dst — Rgba32Float storage, the cells
//   (2) p   — Params

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

struct Params {
    neutral: vec4<f32>,
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    clip: vec4<f32>,
    // x, y = source pixels per cell; z = taps along each side of a cell
    cell: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;

fn scene(camera: vec3<f32>) -> vec3<f32> {
    let n = p.neutral.xyz;
    let c = max(camera, vec3<f32>(0.0));
    let wb = c / n;
    let est = max(wb.r, max(wb.g, wb.b));
    let w = smoothstep(vec3<f32>(p.clip.x), vec3<f32>(p.clip.y), c);
    let rebuilt = (wb + w * (vec3<f32>(est) - wb)) * n;
    return vec3<f32>(
        dot(p.cam_to_work_0.xyz, rebuilt),
        dot(p.cam_to_work_1.xyz, rebuilt),
        dot(p.cam_to_work_2.xyz, rebuilt));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let last = vec2<f32>(textureDimensions(src)) - vec2<f32>(1.0);
    let origin = vec2<f32>(f32(gid.x), f32(gid.y)) * p.cell.xy;
    let taps = i32(p.cell.z);
    var sum = vec3<f32>(0.0);
    for (var j: i32 = 0; j < taps; j = j + 1) {
        for (var i: i32 = 0; i < taps; i = i + 1) {
            let at = origin + (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5)) * p.cell.xy / f32(taps);
            let px = vec2<i32>(clamp(floor(at), vec2<f32>(0.0), last));
            sum = sum + scene(textureLoad(src, px, 0).rgb);
        }
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sum / f32(taps * taps), 1.0));
}
