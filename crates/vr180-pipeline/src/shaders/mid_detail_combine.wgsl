// Mid-detail clarity combine — final pass of the mid-detail pipeline.
//
// Port of `apply_mid_detail` in vr180_gui.py:6424-6491. The preceding
// passes downsample the input 4× and Gaussian-blur it; this shader
// upsamples that blurred-small image back to full resolution (via
// bilinear sample) and combines it with the original:
//
//   blur_up  = sample(blur_small, uv)                  (bilinear upsample)
//   detail   = blur_up - orig                          (signed difference)
//   luma     = 0.299*R + 0.587*G + 0.114*B             (BT.601)
//   weight   = max(0, 1 - 4·(luma - 0.5)²)             (bell curve in midtones)
//   out_rgb  = orig + amount * detail * weight         (clamped 0..1)
//
// The bell-curve weight peaks at 1.0 for luma=0.5 and falls to 0
// at the black/white extremes. This protects shadows + highlights
// from clarity contamination — the Python app added this after
// users reported "blown highlights getting muddier with clarity."
//
// Note: this is the Python `detail = blurred - original` convention.
// A *positive* `amount` pushes pixels toward the low-pass blur (soft
// "haze removal" feel); negative amounts boost local contrast.
// (Confirmed against vr180_gui.py:6481 — `frame += amount * detail`.)

struct MidDetailCombineUniforms {
    amount: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var orig_tex:       texture_2d<f32>;
@group(0) @binding(1) var blur_small_tex: texture_2d<f32>;
@group(0) @binding(2) var bl_sampler:     sampler;
@group(0) @binding(3) var out_tex:        texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> u:     MidDetailCombineUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);
    let orig = textureLoad(orig_tex, coord, 0);

    // Bilinear upsample of the blurred-small image. The small texture
    // covers the same scene at 1/4 resolution; sampling at the full-res
    // pixel-center UV gives us the smooth low-pass version at full res.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let blur_up = textureSampleLevel(blur_small_tex, bl_sampler, uv, 0.0);

    let detail = blur_up - orig;
    let luma = 0.299 * orig.r + 0.587 * orig.g + 0.114 * orig.b;
    let centered = luma - 0.5;
    let weight = max(0.0, 1.0 - 4.0 * centered * centered);

    let scaled = orig + u.amount * detail * weight;
    let rgb = clamp(scaled.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, coord, vec4<f32>(rgb, orig.a));
}
