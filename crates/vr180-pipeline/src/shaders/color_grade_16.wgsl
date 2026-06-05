// 16-bit color grade — same math as `color_grade.wgsl` but writes to
// `rgba16unorm` storage so the 10-bit export path keeps full precision
// through this stage.

struct ColorGradeUniforms {
    temperature: f32,
    tint:        f32,
    saturation:  f32,
    _pad: f32,
};

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba16unorm, write>;
@group(0) @binding(2) var<uniform> u: ColorGradeUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let px = textureLoad(in_tex, vec2<i32>(gid.xy), 0);

    let r_scale = 1.0 + 0.30 * u.temperature;
    let g_scale = 1.0 - 0.30 * u.tint;
    let b_scale = 1.0 - 0.30 * u.temperature;
    var rgb = vec3<f32>(px.r * r_scale, px.g * g_scale, px.b * b_scale);

    let luma = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    let gray = vec3<f32>(luma);
    rgb = mix(gray, rgb, u.saturation);

    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, px.a));
}
