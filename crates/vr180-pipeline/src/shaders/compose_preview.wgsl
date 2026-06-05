// Compose two per-eye half-equirect textures into one preview frame
// using one of three modes. The shape of the output frame depends on
// the mode:
//
//   mode == 0 (SBS):       output is (2·eye_w × eye_h); left → x<eye_w, right → x≥eye_w
//   mode == 1 (Anaglyph):  output is (eye_w × eye_h); R = left.R, GB = right.GB
//   mode == 2 (Overlay):   output is (eye_w × eye_h); 50% blend of left + right
//
// The caller picks output_w / output_h to match the mode so the dispatch
// covers the whole result. Output format is rgba8unorm — this shader is
// used for the **preview** path only (egui's swapchain is 8-bit).

struct PreviewUniforms {
    eye_w: u32,
    mode:  u32,   // 0 sbs, 1 anaglyph, 2 overlay
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var left_tex:  texture_2d<f32>;
@group(0) @binding(1) var right_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex:   texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> u: PreviewUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let coord = vec2<i32>(gid.xy);

    if u.mode == 0u {
        // SBS: pick from left or right depending on x position.
        var rgba: vec4<f32>;
        if gid.x < u.eye_w {
            rgba = textureLoad(left_tex, coord, 0);
        } else {
            let r_coord = vec2<i32>(coord.x - i32(u.eye_w), coord.y);
            rgba = textureLoad(right_tex, r_coord, 0);
        }
        textureStore(out_tex, coord, rgba);
        return;
    }

    // Modes 1 and 2 sample both eyes at the same coord and combine.
    let l = textureLoad(left_tex,  coord, 0);
    let r = textureLoad(right_tex, coord, 0);

    if u.mode == 1u {
        // Anaglyph (red / cyan). Standard "true colour" anaglyph keeps
        // the left eye in red and the right eye in green+blue — fits
        // anyone with red/cyan glasses without colour scrambling.
        // Use BT.709 luminance for the left to avoid pure-red push.
        let l_lum = 0.2126 * l.r + 0.7152 * l.g + 0.0722 * l.b;
        let rgba = vec4<f32>(l_lum, r.g, r.b, 1.0);
        textureStore(out_tex, coord, rgba);
    } else {
        // Overlay (50% blend). The Python app calls it "50 overlay".
        // Plain alpha-50 blend; useful for verifying convergence /
        // disparity at a glance without glasses.
        let rgba = vec4<f32>(0.5 * (l.rgb + r.rgb), 1.0);
        textureStore(out_tex, coord, rgba);
    }
}
