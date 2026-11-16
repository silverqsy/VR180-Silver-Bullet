// Raw development: camera-linear RGB (sensor white = 1) → display-referred
// sRGB-encoded Rec.709. The GPU twin of `raw_tone::DevelopConstants::develop`:
//
//   clip to the camera white → camera → linear ProPhoto (0..1)
//   → exposure ramp → tone curve (hue kept: extremes through the curve,
//   the middle channel at its place between them) → Rec.709 primaries
//   → sRGB encoding.
//
// Bindings:
//   (0) src   — the eye, camera-linear
//   (1) dst   — Rgba16Unorm storage, same size
//   (2) p     — Params
//   (3) curve — tone curve samples over 0..1

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16unorm, write>;

struct Params {
    camera_white: vec4<f32>,
    // Rows of the two matrices.
    cam_to_work_0: vec4<f32>,
    cam_to_work_1: vec4<f32>,
    cam_to_work_2: vec4<f32>,
    work_to_out_0: vec4<f32>,
    work_to_out_1: vec4<f32>,
    work_to_out_2: vec4<f32>,
    // slope, black, radius, q_scale
    ramp: vec4<f32>,
    // slope, a, b, c
    shoulder: vec4<f32>,
    // x = last index of the curve
    misc: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> curve: array<f32>;

fn ramp(x: f32) -> f32 {
    let black = p.ramp.y;
    let radius = p.ramp.z;
    if (x <= black - radius) { return 0.0; }
    if (x >= black + radius) { return min((x - black) * p.ramp.x, 1.0); }
    let y = x - (black - radius);
    return p.ramp.w * y * y;
}

fn shoulder(x: f32) -> f32 {
    if (x <= 0.25) { return x * p.shoulder.x; }
    return (p.shoulder.y * x + p.shoulder.z) * x + p.shoulder.w;
}

fn tone(x: f32) -> f32 {
    let last = p.misc.x;
    let y = clamp(shoulder(x), 0.0, 1.0) * last;
    let i = min(floor(y), last - 1.0);
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
    let cam = clamp(texel.rgb, vec3<f32>(0.0), p.camera_white.xyz);
    let work = clamp(vec3<f32>(
        dot(p.cam_to_work_0.xyz, cam),
        dot(p.cam_to_work_1.xyz, cam),
        dot(p.cam_to_work_2.xyz, cam)), vec3<f32>(0.0), vec3<f32>(1.0));
    let lit = vec3<f32>(ramp(work.r), ramp(work.g), ramp(work.b));
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
