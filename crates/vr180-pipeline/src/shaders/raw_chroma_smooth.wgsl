// Colour noise reduction, second pass (run along the rows, then along the
// columns): every pixel of the half-size picture takes the weighted sum of
// its neighbours along the pass's direction — the spatial taps times a
// Gaussian of how far the neighbour's colour (its values over its
// luminance) is from the pixel's own, so a colour edge is not smeared,
// times a Gaussian of how many stops its luminance is from the pixel's own,
// so no colour crosses an edge of light (the sum is of values: a bright
// neighbour would weigh by its brightness). The rows' pass puts the sum
// back at the pixel's own luminance.
//
// Bindings:
//   (0) src  — Rgba16Float, half size
//   (1) dst  — Rgba16Float storage, half size
//   (2) p    — Params (as raw_chroma_apply.wgsl)
//   (3) taps — the spatial weights from the centre outward

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

struct Params {
    balance: vec4<f32>,
    luma: vec4<f32>,
    dir: vec4<i32>,
    range: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> taps: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let last = vec2<i32>(size) - vec2<i32>(1);
    let c = textureLoad(src, at, 0).rgb;
    let lc = max(dot(p.luma.xyz, c), p.luma.w);
    let qc = c / lc;
    var sum = taps[0] * c;
    for (var i = 1; i <= p.dir.z; i = i + 1) {
        for (var s = -1; s <= 1; s = s + 2) {
            let n = textureLoad(src, clamp(at + s * i * p.dir.xy, vec2<i32>(0), last), 0).rgb;
            let ln = max(dot(p.luma.xyz, n), p.luma.w);
            let d = n / ln - qc;
            let stops = log2(ln / lc);
            sum = sum + taps[i] * exp(-dot(d, d) * p.range.x - stops * stops * p.range.y) * n;
        }
    }
    let own = lc / max(dot(p.luma.xyz, sum), p.luma.w);
    textureStore(dst, at, vec4<f32>(sum * select(1.0, own, p.dir.w == 1), 1.0));
}
