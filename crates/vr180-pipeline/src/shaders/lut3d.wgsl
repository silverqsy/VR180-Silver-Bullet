// 3D LUT apply (trilinear).
//
// Inputs:
//   binding(0): equirect input texture (rgba8unorm)
//   binding(1): the LUT as a 3D texture (rgba8unorm, size×size×size)
//   binding(2): trilinear sampler
//   binding(3): output equirect texture (rgba8unorm storage)
//   binding(4): uniforms (lut size, intensity)
//
// One thread per output pixel. For each input RGB:
//   - convert pixel to [0,1] color
//   - sample the LUT at the half-texel-corrected coord
//   - lerp(orig, lut, intensity) so the user can dial the grade in/out
//   - write back
//
// The half-texel correction matters: WGSL's `textureSample` returns the
// linearly-interpolated value at the given uv, where texel centers sit
// at `(idx + 0.5) / size`. To map color value 0.0 → texel 0 center and
// 1.0 → texel (size-1) center we need `uv = (c * (size - 1) + 0.5) / size`.
// Without it, the LUT samples between black texel and black-edge, washing
// out the grade.

@group(0) @binding(0) var in_tex:  texture_2d<f32>;
@group(0) @binding(1) var lut_tex: texture_3d<f32>;
@group(0) @binding(2) var lut_smp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct LutUniforms {
    // u32 size of one axis of the cube (e.g. 33 for a 33³ LUT).
    size: u32,
    // Blend factor: 0.0 = original color, 1.0 = full LUT effect. The
    // Python pipeline calls this `lut_intensity`.
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

    // Half-texel-corrected LUT coordinate.
    let s = f32(uni.size);
    let uvw = (rgb * (s - 1.0) + 0.5) / s;
    let graded = textureSampleLevel(lut_tex, lut_smp, uvw, 0.0).rgb;

    // Blend toward LUT result.
    let out_rgb = mix(rgb, graded, uni.intensity);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out_rgb, pixel.a));
}
