// RGBA stream textures → EAC cross assembly (one GPU pass).
//
// Windows zero-copy sibling of `nv12_to_eac_cross.wgsl`. On Windows the
// d3d11va/NVDEC P010 frames are converted to single-plane **Rgba16Unorm**
// on the D3D11 side (`interop_windows::P010Converter`, BT.709 limited-range
// YUV→RGB) before import into Vulkan — the multi-plane P010 imports with a
// broken chroma offset, so we never hand wgpu the raw planes. That means
// the EAC cross assembly reads ALREADY-RGB textures here, not YUV planes:
// the only difference from `nv12_to_eac_cross.wgsl` is the per-stream
// sampler (no YUV matrix). The cross→stream geometry is byte-identical, so
// the assembled cross matches the macOS NV12-plane path pixel-for-pixel.
//
// Inputs (per stream, native EAC stream res, Rgba16Unorm):
//   s0_rgba — stream_w × stream_h
//   s4_rgba — stream_w × stream_h
//
// Output:
//   cross_tex — rgba8unorm / rgba16unorm storage, cross_w × cross_w
//   (the rgba16 variant is format-patched at pipeline creation)

@group(0) @binding(0) var s0_rgba: texture_2d<f32>;
@group(0) @binding(1) var s4_rgba: texture_2d<f32>;
@group(0) @binding(2) var eac_smp:  sampler;
@group(0) @binding(3) var cross_tex: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    stream_w:  u32,
    stream_h:  u32,
    tile_w:    u32,
    center_w:  u32,    // fixed at 1920 for GoPro EAC
    cross_w:   u32,
    lens:      u32,    // 0 = Lens A, 1 = Lens B
    _pad0: u32, _pad1: u32,
}
@group(0) @binding(4) var<uniform> uni: Uniforms;

// Sample an already-RGB stream texture at a sub-pixel stream coordinate
// (bilinear). `stream_xy` is in stream-pixel space.
fn sample_rgba(tex: texture_2d<f32>, stream_xy: vec2<f32>) -> vec3<f32> {
    let sw = f32(uni.stream_w);
    let sh = f32(uni.stream_h);
    let uv = (stream_xy + vec2<f32>(0.5)) / vec2<f32>(sw, sh);
    return textureSampleLevel(tex, eac_smp, uv, 0.0).rgb;
}

// Per-(lens, face) inverse mapping from cross-pixel → stream-pixel.
// Returns `vec3<f32>(stream_x, stream_y, stream_idx)` where stream_idx
// is 0.0 for s0, 1.0 for s4. Returns negative stream_idx if the pixel
// is outside any face (caller writes the corner-replicate color).
//
// IDENTICAL to `nv12_to_eac_cross.wgsl::cross_to_stream` — see that file
// for the per-lens tile-layout derivation (it mirrors the Python app's
// _assemble_lensA / _assemble_lensB and the CPU `eac::assemble`).
fn cross_to_stream(cx: u32, cy: u32) -> vec3<f32> {
    let tw   = i32(uni.tile_w);
    let cn   = i32(uni.center_w);
    let sw   = i32(uni.stream_w);
    let sh   = i32(uni.stream_h);
    let cw   = i32(uni.cross_w);
    let i_cx = i32(cx);
    let i_cy = i32(cy);

    let center_top = tw;
    let center_bot = tw + cn;

    // ── Middle row band (cy ∈ [tw, tw+cn)) — Lens-dependent ──
    if (i_cy >= center_top && i_cy < center_bot) {
        let y_in = i_cy - tw;
        if (uni.lens == 0u) {
            if (i_cx < cw) {
                return vec3<f32>(f32(i_cx + tw), f32(y_in), 0.0);
            }
        } else {
            if (i_cx < tw) {
                return vec3<f32>(f32((sw - tw) + i_cx), f32(y_in), 0.0);
            } else if (i_cx < tw + cn) {
                let s_x = tw + (cw - 1 - i_cy);
                let s_y = i_cx - tw;
                return vec3<f32>(f32(s_x), f32(s_y), 1.0);
            } else if (i_cx < cw) {
                return vec3<f32>(f32(i_cx - tw - cn), f32(y_in), 0.0);
            }
        }
    }

    // ── TOP band (cy < tw, cx ∈ [tw, tw+cn)) — rotated s4 tile ──
    if (i_cy < center_top && i_cx >= center_top && i_cx < center_bot) {
        if (uni.lens == 0u) {
            let s_y = sh - 1 - (i_cx - tw);
            let s_x = (sw - tw) + i_cy;
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        } else {
            let s_y = i_cx - tw;
            let s_x = tw + (cw - 1 - i_cy);
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        }
    }

    // ── BOTTOM band (cy ≥ tw+cn, cx ∈ [tw, tw+cn)) — rotated s4 tile ──
    if (i_cy >= center_bot && i_cx >= center_top && i_cx < center_bot) {
        let local_row = i_cy - center_bot;
        if (uni.lens == 0u) {
            let s_y = sh - 1 - (i_cx - tw);
            let s_x = local_row;
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        } else {
            let s_y = i_cx - tw;
            let s_x = tw + (cw - 1 - i_cy);
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        }
    }

    // ── Corner regions: signal "out of face" via negative stream_idx ──
    return vec3<f32>(0.0, 0.0, -1.0);
}

// Find the nearest valid-face pixel for a corner, by clamping the
// cross coord into the closest side band. Mirrors `fill_cross_corners`.
fn clamp_corner_to_side(cx: u32, cy: u32) -> vec2<u32> {
    let tw = uni.tile_w;
    let cn = uni.center_w;
    let center_top = tw;
    let center_bot = tw + cn;
    var ny: u32 = cy;
    if (cy < center_top) { ny = center_top; }
    else if (cy >= center_bot) { ny = center_bot - 1u; }
    return vec2<u32>(cx, ny);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = uni.cross_w;
    if (gid.x >= cw || gid.y >= cw) {
        return;
    }
    var probe = cross_to_stream(gid.x, gid.y);
    if (probe.z < 0.0) {
        let clamped = clamp_corner_to_side(gid.x, gid.y);
        probe = cross_to_stream(clamped.x, clamped.y);
    }
    let stream_xy = vec2<f32>(probe.x, probe.y);
    var rgb: vec3<f32>;
    if (probe.z < 0.5) {
        rgb = sample_rgba(s0_rgba, stream_xy);
    } else {
        rgb = sample_rgba(s4_rgba, stream_xy);
    }
    textureStore(cross_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(rgb, 1.0));
}
