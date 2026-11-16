// Raw development with Shadows / Highlights: `raw_develop.wgsl` with a
// per-pixel gain on the black-subtracted signal. The GPU twin of
// `raw_tone::DevelopConstants::develop_local`.
//
//   scene (ProPhoto, exposure applied) → its luminance's level
//   → base = the frame's bilateral grid at the pixel's place and own level
//     (where the grid holds nothing at that level, the local average)
//   → gain = 2^gains(base)
//   → tone(x) = out(gain · pre(x)), hue kept (the extremes through it, the
//     middle channel at its place) → Rec.709 primaries → sRGB encoding.
//
// Bindings:
//   (0) src   — the eye, camera-linear
//   (1) dst   — Rgba16Unorm storage, same size
//   (2) p     — Params
//   (3) pre   — black-subtracted value over log2(scene)
//   (4) outc  — output light over log2(black-subtracted value)
//   (5) gains — log2 gain over the base level
//   (6) grid  — the smoothed bilateral grid (Rgba16Float, 3D)
//   (7) samp  — linear, clamped

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16unorm, write>;

struct Params {
    neutral: vec4<f32>,
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    work_to_out_0: vec4<f32>,
    work_to_out_1: vec4<f32>,
    work_to_out_2: vec4<f32>,
    // `pre`: x = log2 of the first sample, y = samples per stop, z = last index, w = scene value of the first sample
    pre: vec4<f32>,
    // `outc`: the same for the black-subtracted value
    outc: vec4<f32>,
    // `gains`: x = level of the first sample, y = samples per stop, z = last index
    gains: vec4<f32>,
    // x, y = a channel counts as clipped from x to y; z = luminance floor; w = fallback weight
    clip: vec4<f32>,
    // xyz = luminance weights; w = grid cell in pixels of the eye
    luma: vec4<f32>,
    // x = the grid's first level, y = 1 / range; z, w unused
    grid: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> pre: array<f32>;
@group(0) @binding(4) var<storage, read> outc: array<f32>;
@group(0) @binding(5) var<storage, read> gains: array<f32>;
@group(0) @binding(6) var grid: texture_3d<f32>;
@group(0) @binding(7) var samp: sampler;

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

fn pre_of(x: f32) -> f32 {
    if (x <= 0.0) { return 0.0; }
    if (x <= p.pre.w) { return pre[0] * x / p.pre.w; }
    let y = min((log2(x) - p.pre.x) * p.pre.y, p.pre.z);
    let i = min(floor(y), p.pre.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return pre[k] * (1.0 - f) + pre[k + 1u] * f;
}

fn out_of(v: f32) -> f32 {
    if (v <= 0.0) { return 0.0; }
    if (v <= p.outc.w) { return outc[0] * v / p.outc.w; }
    let y = min((log2(v) - p.outc.x) * p.outc.y, p.outc.z);
    let i = min(floor(y), p.outc.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return outc[k] * (1.0 - f) + outc[k + 1u] * f;
}

fn gain_of(base: f32) -> f32 {
    let y = clamp((base - p.gains.x) * p.gains.y, 0.0, p.gains.z);
    let i = min(floor(y), p.gains.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return exp2(gains[k] * (1.0 - f) + gains[k + 1u] * f);
}

fn srgb(x: f32) -> f32 {
    if (x <= 0.0031308) { return 12.92 * x; }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(src, at, 0);
    let lit = max(scene(texel.rgb) * p.neutral.w, vec3<f32>(0.0));

    // The base: the grid at this pixel's place and own level.
    let level = log2(max(dot(p.luma.xyz, lit), p.clip.z));
    let dims = vec3<f32>(textureDimensions(grid));
    let v = vec3<f32>(
        (f32(gid.x) + 0.5) / p.luma.w + 0.5,
        (f32(gid.y) + 0.5) / p.luma.w + 0.5,
        clamp((level - p.grid.x) * p.grid.y + 0.5, 0.0, dims.z - 1.0));
    let g = textureSampleLevel(grid, samp, (v + vec3<f32>(0.5)) / dims, 0.0);
    let base = (g.x + p.clip.w * g.z) / (g.y + p.clip.w) + p.grid.x;
    let gain = gain_of(base);

    let hi = max(lit.r, max(lit.g, lit.b));
    let lo = min(lit.r, min(lit.g, lit.b));
    let thi = out_of(min(gain * pre_of(hi), 1.0));
    var toned = vec3<f32>(thi);
    if (hi > lo) {
        let tlo = out_of(min(gain * pre_of(lo), 1.0));
        toned = vec3<f32>(tlo) + (thi - tlo) * (lit - vec3<f32>(lo)) / (hi - lo);
    }
    let out = clamp(vec3<f32>(
        dot(p.work_to_out_0.xyz, toned),
        dot(p.work_to_out_1.xyz, toned),
        dot(p.work_to_out_2.xyz, toned)), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(dst, at, vec4<f32>(srgb(out.r), srgb(out.g), srgb(out.b), texel.a));
}
