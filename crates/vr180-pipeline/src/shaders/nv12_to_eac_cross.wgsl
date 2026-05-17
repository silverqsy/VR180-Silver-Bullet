// P010 (10-bit YUV) stream planes → EAC cross assembly (one GPU pass).
//
// Phase 0.6.6 — replaces the CPU `vr180-core::eac::assemble::*`
// functions when the input frames come from IOSurface-backed
// `wgpu::Texture`s (Phase 0.6.5 substrate).
//
// GoPro Max records HEVC Main10 (10-bit) so VideoToolbox produces
// `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` IOSurfaces:
//   Plane 0 = Y, uint16 per pixel (10 bits in upper 10 of 16) → R16Unorm
//   Plane 1 = CbCr interleaved, 2× uint16 per chroma-pair → Rg16Unorm
//
// 8-bit input (NV12 / R8Unorm) would need a different shader entry +
// different limited-range constants. Defer that to a Phase 0.6.6.5
// follow-up if anyone shows up with 8-bit GoPro footage.
//
// One thread per output cross pixel. For each `(cx, cy)`:
//   - Decide which EAC face this pixel belongs to (TOP / LEFT / CENTER
//     / RIGHT / BOTTOM / corner) based on the cross-layout offsets.
//   - For TOP/BOTTOM (the rotated tiles from s4), un-rotate to recover
//     the source stream pixel.
//   - For LEFT/CENTER/RIGHT, it's a simple shift into s0.
//   - For corners, replicate from the adjacent side face's nearest edge
//     pixel (matches the CPU `fill_cross_corners`).
//   - Sample NV12: Y from R8Unorm at the stream pixel, UV from
//     Rg8Unorm at half resolution. BT.709 YUV → RGB matrix.
//   - Write Rgba8Unorm to the cross.
//
// Two dispatches per frame: one for Lens A (uniforms.lens = 0), one for
// Lens B (lens = 1). The shader picks the right tile-source mapping
// from the `lens` uniform.
//
// Inputs (per stream):
//   s0_y  — R8Unorm,  stream_w × stream_h
//   s0_uv — Rg8Unorm, stream_w/2 × stream_h/2
//   s4_y  — R8Unorm,  stream_w × stream_h
//   s4_uv — Rg8Unorm, stream_w/2 × stream_h/2
//
// Output:
//   cross_tex — Rgba8Unorm storage, cross_w × cross_w

@group(0) @binding(0) var s0_y:  texture_2d<f32>;
@group(0) @binding(1) var s0_uv: texture_2d<f32>;
@group(0) @binding(2) var s4_y:  texture_2d<f32>;
@group(0) @binding(3) var s4_uv: texture_2d<f32>;
@group(0) @binding(4) var nv12_smp: sampler;
@group(0) @binding(5) var cross_tex: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    stream_w:  u32,
    stream_h:  u32,
    tile_w:    u32,
    center_w:  u32,    // fixed at 1920 for GoPro EAC
    cross_w:   u32,
    lens:      u32,    // 0 = Lens A, 1 = Lens B
    _pad0: u32, _pad1: u32,
}
@group(0) @binding(6) var<uniform> uni: Uniforms;

// BT.709 limited-range YUV → RGB for **P010** 10-bit content.
//
// P010 stores each 10-bit Y / UV value in the UPPER 10 bits of a u16
// (low 6 bits zero). When sampled as R16Unorm / Rg16Unorm, wgpu hands
// us `s ∈ [0, 1]` mapping from u16 [0, 65535]. To recover the 10-bit
// value: `t10 = s * 65535 / 64`. Then BT.709 limited-range expansion
// is `(t10 - 64) / 876` for Y and `(t10 - 512) / 896` for UV.
// Pre-fused into one mul-add per channel:
//   y_l = s * (65535 / 56064) - (64 / 876)
//   u_l = s * (65535 / 57344) - (512 / 896)
//   v_l = same scale as u_l
fn yuv_to_rgb_bt709_p010(y: f32, u: f32, v: f32) -> vec3<f32> {
    let y_l = y * (65535.0 / 56064.0) - (64.0 / 876.0);
    let u_l = u * (65535.0 / 57344.0) - (512.0 / 896.0);
    let v_l = v * (65535.0 / 57344.0) - (512.0 / 896.0);
    let r = y_l + 1.5748 * v_l;
    let g = y_l - 0.1873 * u_l - 0.4681 * v_l;
    let b = y_l + 1.8556 * u_l;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Sample NV12 at a sub-pixel stream coordinate using the bound sampler
// (bilinear). `stream_xy` is in stream-pixel space; we convert to UV.
fn sample_nv12(
    y_tex:  texture_2d<f32>,
    uv_tex: texture_2d<f32>,
    stream_xy: vec2<f32>,
) -> vec3<f32> {
    let sw = f32(uni.stream_w);
    let sh = f32(uni.stream_h);
    let y_uv  = (stream_xy + vec2<f32>(0.5)) / vec2<f32>(sw, sh);
    // UV plane is half resolution in NV12 — sample at the matching
    // half-pixel position. Bilinear filtering handles the in-between.
    let uv_uv = (stream_xy * 0.5 + vec2<f32>(0.5)) / vec2<f32>(sw * 0.5, sh * 0.5);
    let y = textureSampleLevel(y_tex, nv12_smp, y_uv,  0.0).r;
    let uv = textureSampleLevel(uv_tex, nv12_smp, uv_uv, 0.0).rg;
    return yuv_to_rgb_bt709_p010(y, uv.r, uv.g);
}

// Per-(lens, face) inverse mapping from cross-pixel → stream-pixel.
// Returns `vec3<f32>(stream_x, stream_y, stream_idx)` where stream_idx
// is 0.0 for s0, 1.0 for s4. Returns negative stream_idx if the pixel
// is outside any face (caller writes the corner-replicate color).
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

    // Middle row band: LEFT / CENTER / RIGHT faces from s0.
    if (i_cy >= center_top && i_cy < center_bot) {
        let y_in = i_cy - tw;   // row inside the face band
        if (i_cx < tw) {
            // LEFT face: s0[y_in, cx + tw]
            return vec3<f32>(f32(i_cx + tw), f32(y_in), 0.0);
        } else if (i_cx < tw + cn) {
            // CENTER face: s0[y_in, (cx - tw) + 2*tw] = s0[y_in, cx + tw]
            return vec3<f32>(f32(i_cx + tw), f32(y_in), 0.0);
        } else if (i_cx < cw) {
            // RIGHT face: s0[y_in, (cx - (tw+cn)) + 2*tw + cn] = s0[y_in, cx + tw]
            return vec3<f32>(f32(i_cx + tw), f32(y_in), 0.0);
        }
    }
    // TOP band (rows < tw): rotated tile from s4.
    if (i_cy < center_top && i_cx >= center_top && i_cx < center_bot) {
        // Lens A: source is s4[:, sw-tw .. sw] rotated 90 CW.
        //   In cross local coords (row=cy, col=cx-tw),
        //     dst[r, c] = src[sh-1-c, r]  (forward rot)
        //   so src local = (sh-1-(cx-tw), cy)
        //   src abs = s4[sh-1-(cx-tw), (sw-tw)+cy]
        // Lens B: source is s4[:, tw .. sw-tw] (middle 3*tw+cn=cw wide)
        //   rotated 90 CCW. Inverse:
        //     dst[r, c] = src[c, w-1-r]  (forward CCW for src (h=sh, w=sw-2tw=cw))
        //   so src local = (cx-tw, cw-1-r)? Actually for Lens B's TOP,
        //   the cross top tile is s4_rot[0..tw, :] where s4_rot is the
        //   90-CCW rotation. Let me re-derive in shader using the
        //   forward direction: s4_rot has shape (sw-2tw, sh) = (cw, sh).
        //   s4_rot[i_cy, i_cx-tw] is the top tile pixel. After inverse
        //   rotation (CCW → CW): src[r, c] = s4_rot_src[h-1-c, r]
        //     where s4_rot_src is shape (sh, cw=sw-2tw).
        //   That src is s4[:, tw..sw-tw] with rows = sh, cols = cw.
        //   So s4_rot[i_cy, i_cx-tw] = s4[sh-1-(i_cx-tw), tw + i_cy].
        //   Wait that's the same formula as Lens A. Let me recheck.
        if (uni.lens == 0u) {
            let s_y = sh - 1 - (i_cx - tw);
            let s_x = (sw - tw) + i_cy;
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        } else {
            // Lens B TOP — rotated 90 CCW from s4[:, tw..sw-tw]
            // s4_rot (CCW) of (h=sh, w=cw): dst[r, c] = src[c, w-1-r]
            // So inverse: src[r, c] = dst[w-1-c, r] where dst dims (w, h) = (cw, sh)
            // dst is rows of cross top = [0..tw], cols [tw..tw+cn]
            // dst local (r=cy, c=cx-tw): src local (cy_src = (cw-1)-(cx-tw), cx_src = cy)
            // Wait — this is getting tangled. The cleanest direction:
            //   Forward CCW: dst[w-1-c, r] = src[r, c]
            //   So dst[i, j] = src[j, w-1-i]  (substituting r=j, c=w-1-i)
            //   For our case w = cw (source width before rot), j is col in dst, i is row.
            //   Source local: (row, col) = (j, w-1-i) = (cx-tw, cw-1-cy)
            //   Then absolute s4: s4[(cx-tw), tw + (cw-1-cy)]
            //   But cx-tw needs to map to a stream Y coord... and cw-1-cy to a stream X coord.
            //   Hmm, the source slice for Lens B is s4[:, tw..sw-tw] which has dim (sh, cw).
            //   So "row in source" maps to s4_y, "col in source" maps to s4_x - tw.
            //   s4_y = cx - tw  (which is in [0, cn) = [0, sh) since cn==sh)
            //   s4_x = tw + (cw - 1 - cy)
            let s_y = i_cx - tw;
            let s_x = tw + (cw - 1 - i_cy);
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        }
    }
    // BOTTOM band (rows >= center_bot): rotated tile from s4.
    if (i_cy >= center_bot && i_cx >= center_top && i_cx < center_bot) {
        let local_row = i_cy - center_bot;  // in [0, tw)
        if (uni.lens == 0u) {
            // Lens A BOTTOM: s4[:, 0..tw] rotated 90 CW
            //   dst[r, c] = src[sh-1-c, r]
            //   src local = (sh-1-(cx-tw), local_row)
            //   src abs = s4[sh-1-(cx-tw), 0 + local_row]
            let s_y = sh - 1 - (i_cx - tw);
            let s_x = local_row;
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        } else {
            // Lens B BOTTOM: s4_rot[tw+sh .. tw+sh+tw, :] which is
            //   s4_rot rows [center_bot, center_bot+tw) of the CCW rotation.
            //   Using the same forward-CCW formula as above with the local
            //   row shifted: dst[i, j] = src[j, w-1-i]  but i is offset by
            //   (tw+sh) in s4_rot indexing.
            //   For our slice: dst local i = local_row + (tw + sh)?
            //   Wait — s4_rot is shape (cw, sh) total. The cross uses
            //     top: s4_rot[0..tw, :]
            //     center: s4_rot[tw..tw+sh, :]
            //     bottom: s4_rot[tw+sh..tw+sh+tw, :]
            //   So our bottom-tile dst row = local_row + (tw + sh)
            //   src local: (j, w-1-i) = (cx-tw, cw-1-(local_row+tw+sh))
            //   s4_y = cx - tw
            //   s4_x = tw + (cw - 1 - (local_row + tw + sh))
            let s_y = i_cx - tw;
            let s_x = tw + (cw - 1 - (local_row + tw + sh));
            return vec3<f32>(f32(s_x), f32(s_y), 1.0);
        }
    }
    // Corner regions: signal "out of face" via negative stream_idx.
    return vec3<f32>(0.0, 0.0, -1.0);
}

// Find the nearest valid-face pixel for a corner, by clamping the
// cross coord into the closest side band. Mirrors `fill_cross_corners`
// from `vr180-core::eac::assemble`.
fn clamp_corner_to_side(cx: u32, cy: u32) -> vec2<u32> {
    let tw = uni.tile_w;
    let cn = uni.center_w;
    let center_top = tw;
    let center_bot = tw + cn;
    // Clamp Y into the middle band (so we land in LEFT/CENTER/RIGHT).
    var ny: u32 = cy;
    if (cy < center_top) { ny = center_top; }
    else if (cy >= center_bot) { ny = center_bot - 1u; }
    // X can stay where it is — the row pick will land us on a real face.
    return vec2<u32>(cx, ny);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = uni.cross_w;
    if (gid.x >= cw || gid.y >= cw) {
        return;
    }
    var probe = cross_to_stream(gid.x, gid.y);
    // If we landed in a corner, re-aim at the nearest valid side face
    // (matches CPU `fill_cross_corners` edge-replication).
    if (probe.z < 0.0) {
        let clamped = clamp_corner_to_side(gid.x, gid.y);
        probe = cross_to_stream(clamped.x, clamped.y);
    }
    let stream_xy = vec2<f32>(probe.x, probe.y);
    var rgb: vec3<f32>;
    if (probe.z < 0.5) {
        rgb = sample_nv12(s0_y, s0_uv, stream_xy);
    } else {
        rgb = sample_nv12(s4_y, s4_uv, stream_xy);
    }
    textureStore(cross_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(rgb, 1.0));
}
