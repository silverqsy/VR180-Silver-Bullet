// Exposure change on log-encoded footage, made in the light the sensor
// recorded.
//
// Each channel is taken back to scene-linear with the camera's own log
// curve, multiplied by the gain, and encoded again:
//
//     u' = E(D(u) * gain)
//
// On the logarithmic part of the curve that is close to a constant shift of
// the code; on the straight toe near black it is not, and a shift there
// would lift or crush the shadows. Black itself (D = 0) stays where it is.
//
// The curve is a straight toe joined to a logarithm (`LogCurve` on the Rust
// side, where the same arithmetic is written out in f64):
//
//     E(x) = lin_slope * x + lin_offset                     x <= lin_cut
//     E(x) = log_offset + log_scale * log10(x + log_bias)   x >  lin_cut
//
// The stage sits ahead of every creative one (CDL, 3D LUT): they then see
// the picture a normal exposure would have given them.
//
// The 16-bit variant is produced at pipeline-creation time by the usual
// `rgba8unorm, write` → `rgba16unorm, write` replacement.

struct LogExposureParams {
    lin_slope:  f32,
    lin_offset: f32,
    lin_cut:    f32,   // scene value where the toe ends
    code_cut:   f32,   // the code there
    log_scale:  f32,
    log_offset: f32,
    log_bias:   f32,
    gain:       f32,   // 2^stops
};

@group(0) @binding(0) var in_tex:  texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: LogExposureParams;

const LN_10: f32 = 2.302585093;

fn decode(c: vec3<f32>) -> vec3<f32> {
    let toe = (c - vec3<f32>(u.lin_offset)) / u.lin_slope;
    let lg = exp((c - vec3<f32>(u.log_offset)) * (LN_10 / u.log_scale)) - vec3<f32>(u.log_bias);
    return select(lg, toe, c < vec3<f32>(u.code_cut));
}

fn encode(x: vec3<f32>) -> vec3<f32> {
    let toe = x * u.lin_slope + vec3<f32>(u.lin_offset);
    // The argument is only ever small on the toe's side of the cut, where
    // this branch is not taken; the floor keeps it finite all the same.
    let lg = vec3<f32>(u.log_offset)
        + log(max(x + vec3<f32>(u.log_bias), vec3<f32>(1e-6))) * (u.log_scale / LN_10);
    return select(lg, toe, x <= vec3<f32>(u.lin_cut));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let px = textureLoad(in_tex, vec2<i32>(gid.xy), 0);
    let rgb = clamp(encode(decode(px.rgb) * u.gain), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(rgb, px.a));
}
