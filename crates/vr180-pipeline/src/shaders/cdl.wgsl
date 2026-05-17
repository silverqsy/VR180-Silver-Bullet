// CDL (ASC Color Decision List) + shadow / highlight zone adjustments.
//
// Port of `build_color_1d_lut` in vr180_gui.py (lines 6355-6384). The
// Python version builds a 1-D LUT and applies it via `cv2.LUT`; we do
// the math per-pixel on the GPU since the work fits in one shader pass
// and we avoid the upload of a 256-entry table.
//
// Operation order per channel (RGB independently):
//   1. lift     :  x = x + lift * (1 - x)
//   2. gain     :  x = x * gain
//   3. shadow   :  x += shadow    * smoothstep(clamp(1 - 2x, 0, 1)) * 0.6
//   4. highlight:  x += highlight * smoothstep(clamp(2x - 1, 0, 1)) * 0.6
//   5. gamma    :  x = x ^ (1 / gamma)
//
// The 0.6 scale on the shadow / highlight masks matches the Python
// pivot=0.5 / strength=0.6 defaults — limits each band to a ±60%
// adjustment range so the user knob travels [-1..+1] without saturating.
//
// All math is in 0..1 float space. Alpha is passed through unchanged.

struct CdlUniforms {
    lift:      f32,
    gamma:     f32,
    gain:      f32,
    shadow:    f32,
    highlight: f32,
    // std140 round-up to 32 bytes (Metal / Vulkan happier with multiples of 16):
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: CdlUniforms;

// Hermite smoothstep: 3t² - 2t³.  Input is already clamped to [0..1].
fn hermite(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Apply CDL math to one channel.
fn cdl_channel(x_in: f32) -> f32 {
    var x = x_in;
    // 1. lift — pushes blacks toward 1.0 when lift > 0
    x = x + u.lift * (1.0 - x);
    // 2. gain — multiplicative scaling
    x = x * u.gain;
    // 3. shadow mask — bell-curve centered in lower half
    let s_t = clamp(1.0 - 2.0 * x, 0.0, 1.0);
    x = x + u.shadow * hermite(s_t) * 0.6;
    // 4. highlight mask — bell-curve centered in upper half
    let h_t = clamp(2.0 * x - 1.0, 0.0, 1.0);
    x = x + u.highlight * hermite(h_t) * 0.6;
    // 5. gamma — pow(x, 1/gamma). Clamp x ≥ 0 so pow doesn't NaN on
    //    a negative residue from extreme lift / shadow params.
    let inv_g = 1.0 / max(u.gamma, 0.0001);
    x = pow(max(x, 0.0), inv_g);
    return clamp(x, 0.0, 1.0);
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
