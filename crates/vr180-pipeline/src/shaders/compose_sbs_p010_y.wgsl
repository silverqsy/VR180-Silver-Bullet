// SBS RGBA16 → P010 Y plane.
//
// One thread per output Y pixel. Reads left or right eye texture
// depending on x position, computes Rec.709 luma in 10-bit range
// (stored in the top 10 bits of a u16, bottom 6 zero — standard P010
// convention), writes to R16Unorm.
//
// The Y plane is at full SBS resolution, so this dispatches at
// (out_w, out_h).

@group(0) @binding(0) var left_tex:  texture_2d<f32>;
@group(0) @binding(1) var right_tex: texture_2d<f32>;
@group(0) @binding(2) var y_out:     texture_storage_2d<r16unorm, write>;

// `full_range` (0 = video/limited [64,940], 1 = full [0,1023]).
struct Uniforms { eye_w: u32, full_range: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(3) var<uniform> u: Uniforms;

// Rec.709 luma + video-range YCbCr scaling. P010 video-range Y goes
// from 16/256 (black) to 235/256 (white) when stored in the top 8
// bits; the 10-bit version is 64/1024 to 940/1024. After we write to
// R16Unorm (which scales 0..1 to 0..65535), we need the final 16-bit
// value to have the 10-bit Y in its top 10 bits.
//
// Easier: target the 10-bit value directly (range 64..940 for video
// range), then map to R16Unorm's [0,1] by multiplying by 64 / 65535
// (since the bottom 6 bits should be zero, we shift left by 6 by
// multiplying the 10-bit value by 64 before dividing by 65535 — net:
// scale by 64/65535 ≈ 0.000976544).
const Y_SCALE: f32 = 64.0 / 65535.0;

fn rgb_to_y10_videorange(r: f32, g: f32, b: f32) -> f32 {
    // sRGB → Rec.709 luma. Assumes inputs in [0,1] linear-ish (we
    // skip the gamma conversion; this is a perceptual approximation
    // that matches what swscale does for "RGB→YUV" without a
    // dedicated colorspace flag).
    let y_norm = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if (u.full_range != 0u) {
        // Full range: [0,1] → [0, 1023] in 10-bit.
        return y_norm * 1023.0;
    }
    // Video range: [0,1] → [64, 940] in 10-bit.
    return 64.0 + y_norm * (940.0 - 64.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(y_out);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    // Decide which eye + per-eye coords.
    let eye_w = u.eye_w;
    let is_right = gid.x >= eye_w;
    let local_x = select(gid.x, gid.x - eye_w, is_right);

    let src_uv = vec2<f32>(
        (f32(local_x) + 0.5) / f32(eye_w),
        (f32(gid.y) + 0.5) / f32(out_dim.y),
    );
    let pixel = select(
        textureLoad(left_tex,  vec2<i32>(i32(local_x), i32(gid.y)), 0),
        textureLoad(right_tex, vec2<i32>(i32(local_x), i32(gid.y)), 0),
        is_right,
    );
    let y10 = rgb_to_y10_videorange(pixel.r, pixel.g, pixel.b);
    textureStore(y_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(y10 * Y_SCALE, 0.0, 0.0, 0.0));
}
