// 16-bit 3D LUT apply (trilinear). Same math + half-texel correction as
// `lut3d.wgsl`. The LUT itself is still 8-bit storage (33³ entries — the
// sampling math runs in floats so the precision loss is bounded to the
// LUT's quantization, which is not relevant to keeping the rest of the
// pipeline at 10-bit). Output is `rgba16unorm` so the grade stays at
// 10-bit through to the encoder.

@group(0) @binding(0) var in_tex:  texture_2d<f32>;
@group(0) @binding(1) var lut_tex: texture_3d<f32>;
@group(0) @binding(2) var lut_smp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16unorm, write>;

struct LutUniforms {
    size: u32,
    intensity: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> uni: LutUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(out_tex);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let pixel = textureLoad(in_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let rgb = clamp(pixel.rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    let s = f32(uni.size);
    let uvw = (rgb * (s - 1.0) + 0.5) / s;
    let graded = textureSampleLevel(lut_tex, lut_smp, uvw, 0.0).rgb;

    let out_rgb = mix(rgb, graded, uni.intensity);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out_rgb, pixel.a));
}
