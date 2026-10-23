// P210 textures → planar yuv422p10le (LSB-aligned) storage buffers: the
// exact layout the ProRes encoders consume. Runs after the unchanged
// compose_sbs_p010_y / compose_sbs_p210_uv passes and does on the GPU what
// the encode thread used to do on the CPU after reading those textures
// back: `>> 6` and split U/V. Reading the stored unorm16 values back
// (u/65535 → ×65535 → round is exact for 16-bit) keeps the codes
// bit-identical to that path — recomputing them from float here would not
// be, because the storage write's float→unorm rounding differs from WGSL's.
//
// One thread per 4 horizontal pixels of one row: 2 packed Y words, 1 U
// word, 1 V word. WGSL has no 16-bit storage, so samples are packed in
// pairs; the left sample goes in the low half (little-endian = the byte
// order of the CPU planes). Requires out_w % 4 == 0.

@group(0) @binding(0) var y_tex:  texture_2d<f32>;   // R16Unorm, out_w × out_h
@group(0) @binding(1) var uv_tex: texture_2d<f32>;   // Rg16Unorm, out_w/2 × out_h
@group(0) @binding(2) var<storage, read_write> y_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> u_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> v_out: array<u32>;

struct Uniforms { out_w: u32, out_h: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(5) var<uniform> u: Uniforms;

// MSB-aligned unorm16 sample → its 10-bit code.
fn code10(v: f32) -> u32 {
    return u32(round(v * 65535.0)) >> 6u;
}

fn y_at(x: u32, y: u32) -> u32 {
    return code10(textureLoad(y_tex, vec2<i32>(i32(x), i32(y)), 0).r);
}

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.out_w / 4u || gid.y >= u.out_h) { return; }
    let x0 = gid.x * 4u;
    let y = gid.y;

    let yw = (y * u.out_w + x0) / 2u;
    y_out[yw]      = y_at(x0, y)      | (y_at(x0 + 1u, y) << 16u);
    y_out[yw + 1u] = y_at(x0 + 2u, y) | (y_at(x0 + 3u, y) << 16u);

    let c0 = textureLoad(uv_tex, vec2<i32>(i32(x0 / 2u), i32(y)), 0);
    let c1 = textureLoad(uv_tex, vec2<i32>(i32(x0 / 2u + 1u), i32(y)), 0);
    let cw = (y * (u.out_w / 2u) + x0 / 2u) / 2u;
    u_out[cw] = code10(c0.r) | (code10(c1.r) << 16u);
    v_out[cw] = code10(c0.g) | (code10(c1.g) << 16u);
}
