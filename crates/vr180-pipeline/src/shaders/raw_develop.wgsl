// Raw development: camera-linear RGB (sensor white = 1) → display-referred
// sRGB-encoded Rec.709. The GPU twin of `raw_tone::DevelopConstants::develop`:
//
//   white-balance, rebuild clipped channels → camera → linear ProPhoto
//   → exposure gain → the frame's tone curve over log2(scene), hue kept
//   (the extremes through the curve, the middle channel at its place
//   between them) → Rec.709 primaries → sRGB encoding.
//
// Bindings:
//   (0) src   — the eye, camera-linear
//   (1) dst   — Rgba16Unorm storage, same size
//   (2) p     — Params
//   (3) curve — the frame's curve: output light over log2(scene)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16unorm, write>;

struct Params {
    // xyz = camera values of the white (largest = 1); w = exposure gain
    neutral: vec4<f32>,
    // Rows of the two matrices.
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    work_to_out_0: vec4<f32>,
    work_to_out_1: vec4<f32>,
    work_to_out_2: vec4<f32>,
    // x = log2 of the first sample, y = samples per stop, z = last index,
    // w = scene value of the first sample
    curve: vec4<f32>,
    // x, y = a channel counts as clipped from x to y (of the sensor's white)
    clip: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> curve: array<f32>;

// Scene RGB (linear ProPhoto) of a camera pixel, clipped channels rebuilt:
// white-balanced, a neutral reads the same in every channel, so a channel
// near its clip point is raised to what the brightest channel says.
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

fn tone(x: f32) -> f32 {
    if (x <= 0.0) { return 0.0; }
    if (x <= p.curve.w) { return curve[0] * x / p.curve.w; }
    let y = min((log2(x) - p.curve.x) * p.curve.y, p.curve.z);
    let i = min(floor(y), p.curve.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return curve[k] * (1.0 - f) + curve[k + 1u] * f;
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
    let hi = max(lit.r, max(lit.g, lit.b));
    let lo = min(lit.r, min(lit.g, lit.b));
    let thi = tone(hi);
    var toned = vec3<f32>(thi);
    if (hi > lo) {
        let tlo = tone(lo);
        toned = vec3<f32>(tlo) + (thi - tlo) * (lit - vec3<f32>(lo)) / (hi - lo);
    }
    let out = clamp(vec3<f32>(
        dot(p.work_to_out_0.xyz, toned),
        dot(p.work_to_out_1.xyz, toned),
        dot(p.work_to_out_2.xyz, toned)), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(dst, at, vec4<f32>(srgb(out.r), srgb(out.g), srgb(out.b), texel.a));
}
