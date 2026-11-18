// The reduced copy a frame's linear-light tone is derived from
// (`tone_linear.rs`): every output cell is the mean light of its part of the
// eye (rgb, before the exposure) and, in alpha, the mean of log2(luminance)
// over the same part. The counterpart of `raw_measure.wgsl` for a picture
// that arrives as code values: a code becomes light through the `decode`
// table.
//
// Bindings:
//   (0) src    — the eye, code values
//   (1) dst    — Rgba32Float storage, the cells
//   (2) p      — Params
//   (3) decode — light over the code

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

struct Params {
    // x, y = source pixels per cell; z = taps along each side of a cell
    cell: vec4<f32>,
    // xyz = luminance weights; w = luminance floor
    luma: vec4<f32>,
    // x = last index of `decode`
    table: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> decode: array<f32>;

fn light1(code: f32) -> f32 {
    let y = clamp(code, 0.0, 1.0) * p.table.x;
    let i = min(floor(y), p.table.x - 1.0);
    let f = y - i;
    let k = u32(i);
    return decode[k] * (1.0 - f) + decode[k + 1u] * f;
}

fn light(code: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(light1(code.r), light1(code.g), light1(code.b));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let last = vec2<f32>(textureDimensions(src)) - vec2<f32>(1.0);
    let origin = vec2<f32>(f32(gid.x), f32(gid.y)) * p.cell.xy;
    let taps = i32(p.cell.z);
    var sum = vec3<f32>(0.0);
    var levels = 0.0;
    for (var j: i32 = 0; j < taps; j = j + 1) {
        for (var i: i32 = 0; i < taps; i = i + 1) {
            let at = origin + (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5)) * p.cell.xy / f32(taps);
            let px = vec2<i32>(clamp(floor(at), vec2<f32>(0.0), last));
            let lit = max(light(textureLoad(src, px, 0).rgb), vec3<f32>(0.0));
            sum = sum + lit;
            levels = levels + log2(max(dot(p.luma.xyz, lit), p.luma.w));
        }
    }
    let n = f32(taps * taps);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sum / n, levels / n));
}
