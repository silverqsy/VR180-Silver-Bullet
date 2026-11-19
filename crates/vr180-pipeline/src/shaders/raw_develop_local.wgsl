// The staged raw development: `raw_develop.wgsl` with what sits between
// the default black and the tone curve. The GPU twin of
// `raw_tone::DevelopConstants::develop_gained`.
//
//   scene (ProPhoto, exposure applied)
//   → Shadows / Highlights: the gain of the pixel's base (`raw_gain.wgsl`),
//     smoothed: the mean over the cells it lies between (`gain_cells.wgsl`),
//     held between zero and the pixel's own gain lightly smoothed
//     (`raw_local::smooth_gain`); above zero it is limited by the room r
//     the pixel has below the white point: taken whole up to r/2, beyond
//     that r − r²/(4·gain)
//   → fed(x) = white point (of the gained frame) of 2^gain · black(x), for
//     the largest and the smallest channel
//   → Whites, then Blacks: each either a gain keyed to the pixel's fed
//     luminance (the channels keep their ratios) or a curve the two
//     extremes go through
//   → the tone curve (Contrast in), hue kept (the extremes through it, the
//     middle channel at its place) → Rec.709 primaries → sRGB encoding.
//
// Bindings:
//   (0) src   — the eye, camera-linear
//   (1) dst   — Rgba16Unorm storage, same size
//   (2) p     — Params
//   (3) black — black-subtracted value over log2(scene)
//   (4) fed   — the value the tone curve is fed over log2(gained value)
//   (5) gain  — the gain of every pixel's base, in stops (R32Float, the eye's size)
//   (6) cells — its means over cells (Rgba16Float, in x)
//   (7) samp  — linear, clamped
//   (8) whites, (9) blacks — what a fed value becomes, over its log2
//   (10) tone — output light over log2(fed value)

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
    // `black`: x = log2 of the first sample, y = samples per stop, z = last index, w = scene value of the first sample
    black: vec4<f32>,
    // `fed`: the same for the gained value
    fed: vec4<f32>,
    // x = pixels per cell of `cells`; y = pixels to either side the limit is
    // taken over; z = log2 of the white point; w = output light of a fed value of zero
    spread: vec4<f32>,
    // x, y = a channel is trusted up to x and counts as clipped at y
    clip: vec4<f32>,
    // xyz = luminance weights
    luma: vec4<f32>,
    // `whites`, `blacks` and `tone` (from the first sample of `fed` up to 1):
    // x = samples per stop, y = last index
    stage: vec4<f32>,
    // Whites, Blacks: x, y = what a fed value of zero becomes; z, w = 1 when
    // the stage is a gain keyed to the luminance
    keyed: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> black: array<f32>;
@group(0) @binding(4) var<storage, read> fed: array<f32>;
@group(0) @binding(5) var gain: texture_2d<f32>;
@group(0) @binding(6) var cells: texture_2d<f32>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var<storage, read> whites: array<f32>;
@group(0) @binding(9) var<storage, read> blacks: array<f32>;
@group(0) @binding(10) var<storage, read> tone: array<f32>;

// `scene`: camera pixel -> scene RGB, clipped channels rebuilt (`raw_scene.wgsl`).
//#use SCENE

fn black_of(x: f32) -> f32 {
    if (x <= 0.0) { return 0.0; }
    if (x <= p.black.w) { return black[0] * x / p.black.w; }
    let y = min((log2(x) - p.black.x) * p.black.y, p.black.z);
    let i = min(floor(y), p.black.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return black[k] * (1.0 - f) + black[k + 1u] * f;
}

fn fed_of(v: f32) -> f32 {
    if (v <= 0.0) { return 0.0; }
    if (v <= p.fed.w) { return fed[0] * v / p.fed.w; }
    let y = min((log2(v) - p.fed.x) * p.fed.y, p.fed.z);
    let i = min(floor(y), p.fed.z - 1.0);
    let f = y - i;
    let k = u32(i);
    return fed[k] * (1.0 - f) + fed[k + 1u] * f;
}

// Where a fed value falls in the tables over its log2: x = index of the
// sample below, y = fraction to the next. Only for values above the first
// sample.
fn stage_at(v: f32) -> vec2<f32> {
    let y = min((log2(v) - p.fed.x) * p.stage.x, p.stage.y);
    let i = min(floor(y), p.stage.y - 1.0);
    return vec2<f32>(i, y - i);
}

fn whites_of(v: f32) -> f32 {
    if (v <= 0.0) { return p.keyed.x; }
    if (v <= p.fed.w) { return mix(p.keyed.x, whites[0], v / p.fed.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return whites[k] * (1.0 - at.y) + whites[k + 1u] * at.y;
}

fn blacks_of(v: f32) -> f32 {
    if (v <= 0.0) { return p.keyed.y; }
    if (v <= p.fed.w) { return mix(p.keyed.y, blacks[0], v / p.fed.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return blacks[k] * (1.0 - at.y) + blacks[k + 1u] * at.y;
}

// A keyed stage: the luminance `y` of the pixel goes to `to`, its extremes
// `x` (largest, smallest) keep their ratio to it (next to black: their
// difference from it).
fn keyed_extremes(x: vec2<f32>, y: f32, to: f32) -> vec2<f32> {
    let over = max(y, p.fed.w);
    return x * (to / over) + vec2<f32>(to * (1.0 - y / over));
}

fn tone_of(v: f32) -> f32 {
    if (v <= 0.0) { return p.spread.w; }
    if (v <= p.fed.w) { return mix(p.spread.w, tone[0], v / p.fed.w); }
    let at = stage_at(v);
    let k = u32(at.x);
    return tone[k] * (1.0 - at.y) + tone[k + 1u] * at.y;
}

// The binomial weight of the pixel `d` away, of 2r + 1 (r up to 2).
fn limit_weight(d: i32, r: i32) -> f32 {
    let a = abs(d);
    if (r == 2) { return select(select(1.0, 4.0, a == 1), 6.0, a == 0) / 16.0; }
    if (r == 1) { return select(0.25, 0.5, a == 0); }
    return 1.0;
}

// The gain the picture gives the pixel `at`, in stops, before the white
// limiter: the cells' means, linear between their centres, held between
// zero and the pixel's own gain lightly smoothed.
fn spread_gain(at: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(gain)) - vec2<i32>(1);
    let r = i32(p.spread.y);
    var own = 0.0;
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            let g = textureLoad(gain, clamp(at + vec2<i32>(dx, dy), vec2<i32>(0), last), 0).r;
            own = own + limit_weight(dx, r) * limit_weight(dy, r) * g;
        }
    }
    let size = vec2<f32>(textureDimensions(cells));
    let around = textureSampleLevel(cells, samp, (vec2<f32>(at) + vec2<f32>(0.5)) / (p.spread.x * size), 0.0).r;
    return clamp(around, min(own, 0.0), max(own, 0.0));
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

    let luminance = dot(p.luma.xyz, lit);
    // The room below the white point limits a gain above zero.
    let wanted = spread_gain(at);
    let room = max(p.spread.z - log2(max(black_of(luminance), 1e-12)), 0.0);
    let limited = room - room * room / (4.0 * max(wanted, 1e-6));
    let gain_of = exp2(select(wanted, limited, wanted > 0.5 * room));

    let hi = max(lit.r, max(lit.g, lit.b));
    let lo = min(lit.r, min(lit.g, lit.b));
    let fhi = fed_of(gain_of * black_of(hi));
    var flo = fhi;
    var between = 0.0;
    if (hi > lo) {
        flo = fed_of(gain_of * black_of(lo));
        between = (luminance - lo) / (hi - lo);
    }
    // Whites, then Blacks. `between` (where the luminance lies between the
    // extremes) is kept by every stage.
    var x = vec2<f32>(fhi, flo);
    if (p.keyed.z > 0.5) {
        let y = x.y + (x.x - x.y) * between;
        x = keyed_extremes(x, y, whites_of(y));
    } else {
        x = vec2<f32>(whites_of(x.x), whites_of(x.y));
    }
    if (p.keyed.w > 0.5) {
        let y = x.y + (x.x - x.y) * between;
        x = keyed_extremes(x, y, blacks_of(y));
    } else {
        x = vec2<f32>(blacks_of(x.x), blacks_of(x.y));
    }
    let thi = tone_of(x.x);
    var toned = vec3<f32>(thi);
    if (hi > lo) {
        let tlo = tone_of(x.y);
        toned = vec3<f32>(tlo) + (thi - tlo) * (lit - vec3<f32>(lo)) / (hi - lo);
    }
    let out = clamp(vec3<f32>(
        dot(p.work_to_out_0.xyz, toned),
        dot(p.work_to_out_1.xyz, toned),
        dot(p.work_to_out_2.xyz, toned)), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(dst, at, vec4<f32>(srgb(out.r), srgb(out.g), srgb(out.b), texel.a));
}
