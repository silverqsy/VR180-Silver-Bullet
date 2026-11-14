// 16-bit 3D LUT apply (trilinear). Same arithmetic as `lut3d.wgsl`; the
// output is `rgba16unorm`, so the grade stays at 10-bit precision through
// to the encoder. The cube is stored at 16 bits a channel and interpolated
// here, not by a sampler: both halves of that matter at this depth (8-bit
// storage alone is two 10-bit codes off).

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

    // The cube's own values, interpolated here: a hardware sampler blends
    // with weights of 8 fractional bits, which is up to 0.8 of a 10-bit
    // code off where neighbouring lattice values are far apart. Colour 0.0
    // is lattice point 0 and 1.0 the last one.
    let s = f32(uni.size);
    let pos = rgb * (s - 1.0);
    let cell = clamp(floor(pos), vec3<f32>(0.0), vec3<f32>(s - 2.0));
    let t = pos - cell;
    let i = vec3<i32>(cell);
    let c000 = textureLoad(lut_tex, i, 0).rgb;
    let c100 = textureLoad(lut_tex, i + vec3<i32>(1, 0, 0), 0).rgb;
    let c010 = textureLoad(lut_tex, i + vec3<i32>(0, 1, 0), 0).rgb;
    let c110 = textureLoad(lut_tex, i + vec3<i32>(1, 1, 0), 0).rgb;
    let c001 = textureLoad(lut_tex, i + vec3<i32>(0, 0, 1), 0).rgb;
    let c101 = textureLoad(lut_tex, i + vec3<i32>(1, 0, 1), 0).rgb;
    let c011 = textureLoad(lut_tex, i + vec3<i32>(0, 1, 1), 0).rgb;
    let c111 = textureLoad(lut_tex, i + vec3<i32>(1, 1, 1), 0).rgb;
    let graded = mix(
        mix(mix(c000, c100, t.x), mix(c010, c110, t.x), t.y),
        mix(mix(c001, c101, t.x), mix(c011, c111, t.x), t.y),
        t.z);

    let out_rgb = mix(rgb, graded, uni.intensity);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out_rgb, pixel.a));
}
