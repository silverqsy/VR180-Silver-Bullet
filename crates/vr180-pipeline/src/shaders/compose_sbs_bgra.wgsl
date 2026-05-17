// Compose two half-equirect textures (left + right eye, RGBA8) into one
// side-by-side BGRA8 texture suitable for the VideoToolbox encoder. The
// destination is an IOSurface-backed wgpu::Texture viewed as Rgba8Unorm —
// the BGRA byte order is achieved by writing channels in swapped order
// (storing `(b, g, r, a)` so when VT reads the underlying bytes as BGRA
// it gets correct colors).
//
// Output layout: output[x ∈ [0, eye_w),     y] = left[x, y]
//                output[x ∈ [eye_w, 2eye_w), y] = right[x - eye_w, y]
//
// One dispatch covers the full 2·eye_w × eye_h frame. Branch on x
// uniformly per workgroup (workgroup_size.x = 8 means each workgroup
// straddles exactly one boundary at x = eye_w only at workgroups where
// `eye_w % 8 != 0`; in practice eye_w is a power of two for our exports
// so it never straddles).

struct SbsComposeUniforms {
    eye_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var left_tex:  texture_2d<f32>;
@group(0) @binding(1) var right_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex:   texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> u: SbsComposeUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);

    var rgba: vec4<f32>;
    if gid.x < u.eye_w {
        rgba = textureLoad(left_tex, coord, 0);
    } else {
        let r_coord = vec2<i32>(coord.x - i32(u.eye_w), coord.y);
        rgba = textureLoad(right_tex, r_coord, 0);
    }

    // R↔B swap: the wgpu texture is viewed as Rgba8Unorm but its
    // bytes land in BGRA byte order in the underlying IOSurface, which
    // the VideoToolbox encoder reads as `kCVPixelFormatType_32BGRA`.
    // Writing (b, g, r, a) here therefore produces correct colors at
    // the encoder.
    textureStore(out_tex, coord, vec4<f32>(rgba.b, rgba.g, rgba.r, rgba.a));
}
