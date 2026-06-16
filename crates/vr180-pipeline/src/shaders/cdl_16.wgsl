// 16-bit CDL (ASC Color Decision List) + shadow / highlight zone
// adjustments. Same math as `cdl.wgsl`, but the storage output is
// `rgba16unorm` so the intermediate stays at 10-bit precision when
// targeting the 10-bit export path.

struct CdlUniforms {
    lift:      f32,
    gamma:     f32,
    gain:      f32,
    shadow:    f32,
    highlight: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba16unorm, write>;
@group(0) @binding(2) var<uniform> u: CdlUniforms;

fn hermite(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }

fn cdl_channel(x_in: f32) -> f32 {
    var x = x_in;
    x = x + u.lift * (1.0 - x);
    x = x * u.gain;
    let s_t = clamp(1.0 - 2.0 * x, 0.0, 1.0);
    x = x + u.shadow * hermite(s_t) * 0.6;
    let h_t = clamp(2.0 * x - 1.0, 0.0, 1.0);
    x = x + u.highlight * hermite(h_t) * 0.6;
    // Clip to [0,1] BEFORE gamma — matches the Python build_color_1d_lut.
    x = clamp(x, 0.0, 1.0);
    let inv_g = 1.0 / max(u.gamma, 0.0001);
    x = pow(x, inv_g);
    return x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let px = textureLoad(in_tex, vec2<i32>(gid.xy), 0);
    let r = cdl_channel(px.r);
    let g = cdl_channel(px.g);
    let b = cdl_channel(px.b);
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(r, g, b, px.a));
}
