// Raw sharpening, last pass: the kernel's change of each pixel's level,
// limited and faded as Camera Raw limits and fades it, put back into the
// picture as one gain on its linear RGB.
//
//   d  = fade(phi) · mask · soft(delta)          clamped to what the pixel's
//   phi' = phi + d                                own level allows
//   rgb' = rgb · (phi' / phi)^gamma
//
// Bindings:
//   (0) src    — the developed picture (sRGB-encoded)
//   (1) dst    — Rgba16Unorm
//   (2) p      — Params
//   (3) level  — phi
//   (4) delta  — the kernel's change of phi
//   (5) edge   — the gradient's size, spread (read when the mask is on)
//   (6) tables — the largest darkening, the largest lightening (each
//                p.table.x + 1 values over phi 0..1), then the mask's curve
//                (p.table.y + 1 values)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16unorm, write>;

struct Params {
    // xyz = luminance weights (unused here); w = gamma
    luma: vec4<f32>,
    // x = the soft limit's scale; y = the mix of its two shapes; w = 1 with a mask
    soft: vec4<f32>,
    // none of the change at x, all of it from y to z, none at w
    fade: vec4<f32>,
    // x = what the spread gradient is multiplied by; y = 1 / threshold;
    // z = log2 of the curve's first sample; w = samples per unit of log2
    mask: vec4<f32>,
    // x = last index of a limit table; y = last index of the mask's curve
    table: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var level: texture_2d<f32>;
@group(0) @binding(4) var delta: texture_2d<f32>;
@group(0) @binding(5) var edge: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> tables: array<f32>;

fn decode(e: vec3<f32>) -> vec3<f32> {
    let lo = e / 12.92;
    let hi = pow((max(e, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, e <= vec3<f32>(0.04045));
}

fn encode(x: vec3<f32>) -> vec3<f32> {
    let lo = 12.92 * x;
    let hi = 1.055 * pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// A table of `last + 1` values starting at `first`, at position x in 0..last.
fn table_at(first: u32, last: f32, x: f32) -> f32 {
    let c = clamp(x, 0.0, last);
    let i = min(floor(c), last - 1.0);
    let f = c - i;
    let k = first + u32(i);
    return tables[k] * (1.0 - f) + tables[k + 1u] * f;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(src, at, 0);
    let phi = textureLoad(level, at, 0).r;
    let d = textureLoad(delta, at, 0).r;

    let x = abs(d) / p.soft.x;
    let limited = sign(d) * p.soft.x * ((1.0 - p.soft.y) * (1.0 - exp(-x)) + p.soft.y * x / (1.0 + x));
    let fade = clamp((phi - p.fade.x) / (p.fade.y - p.fade.x), 0.0, 1.0)
             * clamp((p.fade.w - phi) / (p.fade.w - p.fade.z), 0.0, 1.0);

    let strength = max(p.mask.x * textureLoad(edge, at, 0).r, 1e-12);
    let n = u32(p.table.x) + 1u;
    let share = table_at(2u * n, p.table.y, (log2(strength * p.mask.y) - p.mask.z) * p.mask.w);
    let mask = select(1.0, share, p.soft.w > 0.5);

    let at_level = phi * p.table.x;
    let change = clamp(fade * mask * limited, -table_at(0u, p.table.x, at_level), table_at(n, p.table.x, at_level));
    let out = clamp(phi + change, 0.0, 1.0);
    let gain = pow(out / phi, p.luma.w);
    let rgb = clamp(decode(texel.rgb) * gain, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(dst, at, vec4<f32>(encode(rgb), texel.a));
}
