// Unsharp mask combine — final pass of the sharpen pipeline.
//
// Port of `apply_equirect_sharpen` in vr180_gui.py:6699-6744. The
// preceding 2 passes (horizontal + vertical Gaussian blur) produce
// `blur_tex`. This shader computes:
//
//   detail   = orig - blur                   (high-frequency residual)
//   lat_w    = cos(π · (0.5 - y/H))  clip(0.02..1.0)   if apply_lat_weight else 1.0
//   out_rgb  = orig + lat_w * amount * detail        (USM, clamped 0..1)
//
// `apply_lat_weight=1` is the equirect path: the cos(latitude) weight
// attenuates sharpening near the poles where the projection compresses
// pixels heavily (over-sharpening there produces angular ringing).
// `apply_lat_weight=0` is for non-equirect surfaces (e.g. fisheye preview
// or downsampled diagnostic frames) where uniform sharpening is correct.

struct SharpenCombineUniforms {
    amount:           f32,
    apply_lat_weight: u32,   // 0 = uniform 1.0, 1 = cos(latitude)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var orig_tex: texture_2d<f32>;
@group(0) @binding(1) var blur_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> u: SharpenCombineUniforms;

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);
    let orig = textureLoad(orig_tex, coord, 0);
    let blur = textureLoad(blur_tex, coord, 0);
    let detail = orig - blur;

    var lat_w = 1.0;
    if u.apply_lat_weight != 0u {
        let h_f = f32(dims.y);
        let y_norm = (f32(coord.y) + 0.5) / h_f;
        lat_w = clamp(cos(PI * (0.5 - y_norm)), 0.02, 1.0);
    }

    let scaled = orig + lat_w * u.amount * detail;
    let rgb = clamp(scaled.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, coord, vec4<f32>(rgb, orig.a));
}
