// SBS RGBA16 → P210 UV plane (4:2:2 — full vertical chroma).
//
// One thread per output UV pixel, which covers a 2×1 HORIZONTAL pair of
// the full-res image (unlike the P010 variant's 2×2 block — P210 keeps
// every chroma row). Averages the pair, computes Cb/Cr in 10-bit video
// range, writes interleaved U V to Rg16Unorm at the top 10 bits. Used
// to feed the VideoToolbox hardware ProRes encoder true 4:2:2.

@group(0) @binding(0) var left_tex:  texture_2d<f32>;
@group(0) @binding(1) var right_tex: texture_2d<f32>;
@group(0) @binding(2) var uv_out:    texture_storage_2d<rg16unorm, write>;

// `full_range` (0 = video/limited [64,960], 1 = full [0,1023]).
struct Uniforms { eye_w: u32, full_range: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(3) var<uniform> u: Uniforms;

const CHROMA_SCALE: f32 = 64.0 / 65535.0;

fn sample_full(local_x: u32, y: u32, is_right: bool) -> vec4<f32> {
    if (is_right) {
        return textureLoad(right_tex, vec2<i32>(i32(local_x), i32(y)), 0);
    }
    return textureLoad(left_tex, vec2<i32>(i32(local_x), i32(y)), 0);
}

fn rgb_to_uv10_videorange(r: f32, g: f32, b: f32) -> vec2<f32> {
    // Rec.709 chroma. Output in [0,1] then mapped to [64, 960] for
    // 10-bit video range.
    let cb_norm = -0.1146 * r - 0.3854 * g + 0.5 * b + 0.5;
    let cr_norm =  0.5    * r - 0.4542 * g - 0.0458 * b + 0.5;
    if (u.full_range != 0u) {
        // Full range: [0,1] → [0, 1023], chroma centered at 512.
        return vec2<f32>(cb_norm * 1023.0, cr_norm * 1023.0);
    }
    let cb10 = 64.0 + cb_norm * (960.0 - 64.0);
    let cr10 = 64.0 + cr_norm * (960.0 - 64.0);
    return vec2<f32>(cb10, cr10);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let uv_dim = textureDimensions(uv_out);
    if (gid.x >= uv_dim.x || gid.y >= uv_dim.y) { return; }

    // This UV pixel covers full-res pixels (2*gid.x, gid.y) and
    // (2*gid.x+1, gid.y) — horizontal pair, same row.
    let full_w = uv_dim.x * 2u;
    let eye_w = u.eye_w;
    let y = gid.y;

    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var n = 0.0;
    for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
        let x = gid.x * 2u + dx;
        if (x >= full_w) { continue; }
        let is_right = x >= eye_w;
        let local_x = select(x, x - eye_w, is_right);
        let p = sample_full(local_x, y, is_right);
        sum = sum + p.rgb;
        n = n + 1.0;
    }
    let avg = sum / max(n, 1.0);
    let uv10 = rgb_to_uv10_videorange(avg.r, avg.g, avg.b);
    textureStore(uv_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(uv10.x * CHROMA_SCALE, uv10.y * CHROMA_SCALE, 0.0, 0.0));
}
