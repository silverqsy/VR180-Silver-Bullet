// P010 (YCbCr 10-bit limited-range) → RGBA16 resolve + box downscale.
//
// This is the GPU equivalent of the CPU path's "swscale P010 → RGBA, then
// downscale" step, done in the CORRECT order: upsample chroma to full-res
// RGB *first* (one bilinear UV tap per source position), then box-average
// down to the working resolution. Doing it this way (vs. sampling the
// half-res chroma plane at scattered minified positions during projection)
// is what kills the orange/blue chroma moiré in the live preview — the
// chroma is resolved before it can alias.
//
// Output is a plain Rgba16Unorm fisheye image at `out_w × out_h`, which the
// downstream `project_fisheye_rgba16_texture_to_equirect_16` then projects
// with a single bilinear tap (the box prefilter here means the projection's
// minification is small enough that one tap no longer aliases).
//
// Bindings:
//   (0) src_y  — R16Unorm  (full-res 10-bit-in-top-10 Y plane)
//   (1) src_uv — Rg16Unorm (half-res interleaved Cb/Cr)
//   (2) smp    — bilinear sampler (ClampToEdge)
//   (3) out_tex— Rgba16Unorm storage (downscaled fisheye)
//   (4) dims   — src/out dimensions

@group(0) @binding(0) var src_y:  texture_2d<f32>;
@group(0) @binding(1) var src_uv: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16unorm, write>;

struct Dims { src_w: f32, src_h: f32, out_w: f32, out_h: f32 }
@group(0) @binding(4) var<uniform> d: Dims;

// Max box taps per axis. 4 covers up to a 4:1 downscale per pass; the
// working res is chosen so the real ratio is ≤3:1 (native 3840 → ~1280).
const MAX_K: i32 = 4;

// BT.709 limited-range YUV → RGB for P010 (same constants as
// `fisheye_p010_to_hequirect.wgsl::yuv_to_rgb_bt709_p010`).
fn yuv_to_rgb_bt709_p010(y: f32, u: f32, v: f32) -> vec3<f32> {
    let y_l = y * (65535.0 / 56064.0) - (64.0 / 876.0);
    let u_l = u * (65535.0 / 57344.0) - (512.0 / 896.0);
    let v_l = v * (65535.0 / 57344.0) - (512.0 / 896.0);
    let r = y_l + 1.5748 * v_l;
    let g = y_l - 0.1873 * u_l - 0.4681 * v_l;
    let b = y_l + 1.8556 * u_l;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(d.out_w) || gid.y >= u32(d.out_h)) { return; }

    let scale_x = d.src_w / d.out_w;
    let scale_y = d.src_h / d.out_h;
    // Centre of this output texel's footprint in source pixels.
    let cx = (f32(gid.x) + 0.5) * scale_x;
    let cy = (f32(gid.y) + 0.5) * scale_y;
    // One tap per source pixel the footprint covers (clamped).
    let kx = clamp(i32(ceil(scale_x)), 1, MAX_K);
    let ky = clamp(i32(ceil(scale_y)), 1, MAX_K);

    var acc = vec3<f32>(0.0);
    for (var j: i32 = 0; j < ky; j = j + 1) {
        for (var i: i32 = 0; i < kx; i = i + 1) {
            // Spread taps evenly across the footprint, centred on it.
            let ox = ((f32(i) + 0.5) / f32(kx) - 0.5) * scale_x;
            let oy = ((f32(j) + 0.5) / f32(ky) - 0.5) * scale_y;
            let sxy = vec2<f32>(cx + ox, cy + oy);
            // Y at full res; UV bilinear from the half-res plane = full-res
            // chroma upsample at this exact source position.
            let y_uv  = (sxy + vec2<f32>(0.5)) / vec2<f32>(d.src_w, d.src_h);
            let uv_uv = (sxy * 0.5 + vec2<f32>(0.5)) / vec2<f32>(d.src_w * 0.5, d.src_h * 0.5);
            let yv  = textureSampleLevel(src_y,  smp, y_uv,  0.0).r;
            let uvv = textureSampleLevel(src_uv, smp, uv_uv, 0.0).rg;
            acc = acc + yuv_to_rgb_bt709_p010(yv, uvv.r, uvv.g);
        }
    }
    let rgb = acc / f32(kx * ky);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
