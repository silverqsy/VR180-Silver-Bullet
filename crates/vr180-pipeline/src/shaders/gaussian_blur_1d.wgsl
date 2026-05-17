// Separable 1-D Gaussian blur. Dispatched twice per blur:
// once with direction=0 (horizontal pass), then again with
// direction=1 (vertical pass), with the intermediate texture
// feeding into the second dispatch.
//
// Shared by sharpen.wgsl (unsharp mask LP component) and
// mid_detail.wgsl (downsampled-image blur). Keeping it one
// shader cuts compile time + binary size — the cost is the
// `direction` branch per pixel, which on modern GPUs is a
// uniform branch (every workgroup takes the same arm).
//
// Kernel: 9-tap (radius 4). Sigma comes in via uniform; weights
// are computed inline as exp(-i²/2σ²) and normalized to sum=1.
// 9 taps cover ~98% of a σ=1.4 Gaussian and ~85% of a σ=2.0 one,
// which is plenty for the bandlimited blur used by USM and
// clarity. Larger σ values lose accuracy past the kernel edge
// but the perceptual effect saturates anyway.

struct GaussianBlurUniforms {
    sigma:     f32,
    direction: u32,    // 0 = horizontal, 1 = vertical
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: GaussianBlurUniforms;

const RADIUS: i32 = 4;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);
    let max_xy = vec2<i32>(dims) - vec2<i32>(1);

    let sigma = max(u.sigma, 0.0001);
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -RADIUS; i <= RADIUS; i = i + 1) {
        let fi = f32(i);
        let w = exp(-fi * fi * inv_2sigma2);
        var sc: vec2<i32>;
        if u.direction == 0u {
            sc = vec2<i32>(clamp(coord.x + i, 0, max_xy.x), coord.y);
        } else {
            sc = vec2<i32>(coord.x, clamp(coord.y + i, 0, max_xy.y));
        }
        acc  = acc + w * textureLoad(in_tex, sc, 0);
        wsum = wsum + w;
    }
    textureStore(out_tex, coord, acc / max(wsum, 0.0001));
}
