// Tone controls in linear light for a picture that arrives as code values:
// the GPU twin of `tone_linear::ToneConstants::apply`.
//
//   codes → light (the `decode` table), × the exposure
//   → base = the frame's bilateral grid at the pixel's place and own level
//     (where the grid holds nothing at that level, the local average)
//   → Shadows / Highlights: gain = gains(base), in stops; above zero it is
//     limited by the room r the pixel has below white: taken whole up to
//     r/2, beyond that r − r²/(4·gain)
//   → Whites, then Blacks, then Contrast: each what a light becomes (kept
//     as the change of it), either a gain keyed to the pixel's luminance
//     (the channels keep their ratios) or a curve its largest and smallest
//     channel go through; the middle channel keeps its place between them
//   → codes again (the `encode` table, laid out over the sRGB encoding of
//     the light).
//
// A code under a log curve's black is light below none (the curve's
// straight toe runs on there: a clip's noise). It keeps its sign through
// the exposure and the gain, its distance from what black becomes through
// the three stages, and goes back to a code along the toe.
//
// A frame without Shadows / Highlights is given a grid of one empty cell
// and gains of zero: the gain is then exactly 1.
//
// The 16-bit variant is produced at pipeline-creation time by the usual
// `rgba8unorm, write` → `rgba16unorm, write` replacement.
//
// Bindings:
//   (0) src      — the eye, code values
//   (1) dst      — storage, same size
//   (2) p        — Params
//   (3) decode   — light over the code
//   (4) gains    — log2 gain over the base level
//   (5) grid     — the smoothed bilateral grid (Rgba16Float, 3D)
//   (6) samp     — linear, clamped
//   (7) whites, (8) blacks, (9) contrast — the change of a light, over its log2
//   (10) encode  — the code over the sRGB encoding of the light

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

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
    // The stages' tables: x = log2 of the first sample, y = samples per
    // stop, z = last index, w = the light of the first sample
    stage: vec4<f32>,
    // xyz = the change of no light in Whites, Blacks, Contrast;
    // w = last index of `encode`
    floors: vec4<f32>,
    // x, y = 1 when Whites, Blacks are a gain keyed to the luminance;
    // z = codes to a unit of light below none
    keyed: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> decode: array<f32>;
@group(0) @binding(4) var<storage, read> gains: array<f32>;
@group(0) @binding(5) var grid: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> whites: array<f32>;
@group(0) @binding(8) var<storage, read> blacks: array<f32>;
@group(0) @binding(9) var<storage, read> contrast: array<f32>;
@group(0) @binding(10) var<storage, read> encode: array<f32>;

fn light1(code: f32) -> f32 {
    let y = clamp(code, 0.0, 1.0) * p.light.x;
    let i = min(floor(y), p.light.x - 1.0);
    let f = y - i;
    let k = u32(i);
    return decode[k] * (1.0 - f) + decode[k + 1u] * f;
}

// The gain of a base level, in stops, before the white limiter.
fn stops_of(base: f32) -> f32 {
    let y = clamp((base - p.gains.x) * p.gains.y, 0.0, p.gains.z);
    let i = min(floor(y), p.gains.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return gains[k] * (1.0 - f) + gains[k + 1u] * f;
}

// Where a light falls in the stages' tables: x = index of the sample
// below, y = fraction to the next. Only for lights above the first sample;
// under it a stage's change runs in a straight line to what no light
// becomes, and on along that line for a light below none.
fn stage_at(v: f32) -> vec2<f32> {
    let y = min((log2(v) - p.stage.x) * p.stage.y, p.stage.z);
    let i = min(floor(y), p.stage.z - 1.0);
    return vec2<f32>(i, y - i);
}

fn whites_of(v: f32) -> f32 {
    if (v <= p.stage.w) { return v + mix(p.floors.x, whites[0], v / p.stage.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return v + whites[k] * (1.0 - at.y) + whites[k + 1u] * at.y;
}

fn blacks_of(v: f32) -> f32 {
    if (v <= p.stage.w) { return v + mix(p.floors.y, blacks[0], v / p.stage.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return v + blacks[k] * (1.0 - at.y) + blacks[k + 1u] * at.y;
}

fn contrast_of(v: f32) -> f32 {
    if (v <= p.stage.w) { return v + mix(p.floors.z, contrast[0], v / p.stage.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return v + contrast[k] * (1.0 - at.y) + contrast[k + 1u] * at.y;
}

// A keyed stage: the luminance `y` of the pixel goes to `to`, its extremes
// `x` (largest, smallest) keep their ratio to it (next to black: their
// difference from it).
fn keyed_extremes(x: vec2<f32>, y: f32, to: f32) -> vec2<f32> {
    // Below no light: the differences as they are.
    if (y <= 0.0) { return x + vec2<f32>(to - y); }
    let over = max(y, p.stage.w);
    return x * (to / over) + vec2<f32>(to * (1.0 - y / over));
}

fn srgb(x: f32) -> f32 {
    if (x <= 0.0031308) { return 12.92 * x; }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn code_of(x: f32) -> f32 {
    // Below no light: along the curve's toe.
    if (x < 0.0) { return max(encode[0] + x * p.keyed.z, 0.0); }
    let y = srgb(min(x, 1.0)) * p.floors.w;
    let i = min(floor(y), p.floors.w - 1.0);
    let f = y - i;
    let k = u32(i);
    return encode[k] * (1.0 - f) + encode[k + 1u] * f;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(src, at, 0);
    let lit = vec3<f32>(light1(texel.r), light1(texel.g), light1(texel.b)) * p.light.y;

    // The base: the grid at this pixel's place and own level (as the grid
    // was gathered: a channel below no light counts as none).
    let luminance = dot(p.luma.xyz, lit);
    let level = log2(max(dot(p.luma.xyz, max(lit, vec3<f32>(0.0))), p.light.z));
    let dims = vec3<f32>(textureDimensions(grid));
    let v = vec3<f32>(
        (f32(gid.x) + 0.5) / p.luma.w + 0.5,
        (f32(gid.y) + 0.5) / p.luma.w + 0.5,
        clamp((level - p.grid.x) * p.grid.y + 0.5, 0.0, dims.z - 1.0));
    let g = textureSampleLevel(grid, samp, (v + vec3<f32>(0.5)) / dims, 0.0);
    let base = (g.x + p.light.w * g.z) / (g.y + p.light.w) + p.grid.x;
    // The room below white limits a gain above zero.
    let wanted = stops_of(base);
    let room = max(-log2(max(luminance, 1e-12)), 0.0);
    let limited = room - room * room / (4.0 * max(wanted, 1e-6));
    let gain = exp2(select(wanted, limited, wanted > 0.5 * room));

    let hi = max(lit.r, max(lit.g, lit.b));
    let lo = min(lit.r, min(lit.g, lit.b));
    var between = 0.0;
    if (hi > lo) { between = (luminance - lo) / (hi - lo); }
    // Whites, then Blacks, then Contrast. `between` (where the luminance
    // lies between the extremes) is kept by every stage.
    var x = vec2<f32>(min(gain * hi, 1.0), min(gain * lo, 1.0));
    if (p.keyed.x > 0.5) {
        let y = x.y + (x.x - x.y) * between;
        x = keyed_extremes(x, y, whites_of(y));
    } else {
        x = vec2<f32>(whites_of(x.x), whites_of(x.y));
    }
    if (p.keyed.y > 0.5) {
        let y = x.y + (x.x - x.y) * between;
        x = keyed_extremes(x, y, blacks_of(y));
    } else {
        x = vec2<f32>(blacks_of(x.x), blacks_of(x.y));
    }
    x = vec2<f32>(contrast_of(x.x), contrast_of(x.y));
    // The middle channel keeps its place between the two, in light.
    var out = vec3<f32>(x.x);
    if (hi > lo) {
        out = vec3<f32>(x.y) + (x.x - x.y) * (lit - vec3<f32>(lo)) / (hi - lo);
    }
    textureStore(dst, at, vec4<f32>(code_of(out.r), code_of(out.g), code_of(out.b), texel.a));
}
