// Fisheye P010 (YCbCr 10-bit limited-range) → half-equirect projection.
//
// Zero-copy variant of `fisheye_to_hequirect.wgsl`. Instead of taking a
// pre-converted RGBA texture (which costs an FFmpeg P010LE → RGBA64LE
// swscale pass plus a CPU→GPU upload — ~840 MB/frame at full OSV res),
// this shader binds the Y and UV planes of the VideoToolbox-decoded
// IOSurface directly and does YCbCr→RGB inline alongside the KB
// projection.
//
// Math is byte-for-byte identical to `fisheye_to_hequirect.wgsl`; the
// only changes are:
//   - two input textures (Y at full res, UV at half res) instead of one RGBA
//   - YCbCr→RGB BT.709 expansion baked into the per-pixel kernel
//     (same constants as `nv12_to_eac_cross.wgsl::yuv_to_rgb_bt709_p010`)
//
// Bindings:
//   (0) fisheye_y  — R16Unorm  (full-res 10-bit-in-top-10 Y plane)
//   (1) fisheye_uv — Rg16Unorm (half-res interleaved Cb/Cr)
//   (2) fisheye_smp— bilinear sampler
//   (3) out_tex    — Rgba16Unorm storage (half-equirect output)
//   (4) equ        — per-frame stab rotation
//   (5) cal        — per-eye KB calibration

@group(0) @binding(0) var fisheye_y:  texture_2d<f32>;
@group(0) @binding(1) var fisheye_uv: texture_2d<f32>;
@group(0) @binding(2) var fisheye_smp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16unorm, write>;

struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(4) var<uniform> equ: EquirectUniforms;

struct FisheyeCalibUniforms {
    fx: f32, fy: f32, cx: f32, cy: f32,                              // vec4 #0
    k1: f32, k2: f32, k3: f32, k4: f32,                              // vec4 #1
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,        // vec4 #2
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,        // vec4 #3
    p1: f32, p2: f32, _pad3: f32, _pad4: f32,
}
@group(0) @binding(5) var<uniform> cal: FisheyeCalibUniforms;

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

// BT.709 limited-range YUV → RGB for **P010** 10-bit content.
//
// P010 stores each 10-bit Y / UV value in the UPPER 10 bits of a u16
// (low 6 bits zero). When sampled as R16Unorm / Rg16Unorm, wgpu hands
// us `s ∈ [0, 1]` mapping from u16 [0, 65535]. To recover the 10-bit
// value: `t10 = s * 65535 / 64`. Then BT.709 limited-range expansion
// is `(t10 - 64) / 876` for Y and `(t10 - 512) / 896` for UV.
// Pre-fused into one mul-add per channel; matches
// `nv12_to_eac_cross.wgsl::yuv_to_rgb_bt709_p010`.
fn yuv_to_rgb_bt709_p010(y: f32, u: f32, v: f32) -> vec3<f32> {
    let y_l = y * (65535.0 / 56064.0) - (64.0 / 876.0);
    let u_l = u * (65535.0 / 57344.0) - (512.0 / 896.0);
    let v_l = v * (65535.0 / 57344.0) - (512.0 / 896.0);
    let r = y_l + 1.5748 * v_l;
    let g = y_l - 0.1873 * u_l - 0.4681 * v_l;
    let b = y_l + 1.8556 * u_l;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// KB forward + cubic Hermite extension. Copied verbatim from
// `fisheye_to_hequirect.wgsl` — see comments there for the math.
fn kb_forward(theta: f32) -> f32 {
    let t2 = theta * theta;
    let inner = 1.0 + t2 * (cal.k1 + t2 * (cal.k2 + t2 * (cal.k3 + t2 * (cal.k4 + t2 * cal.k5))));
    return cal.fx * theta * inner;
}
fn kb_forward_deriv(theta: f32) -> f32 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    let t10 = t8 * t2;
    return cal.fx * (1.0
        + 3.0 * cal.k1 * t2
        + 5.0 * cal.k2 * t4
        + 7.0 * cal.k3 * t6
        + 9.0 * cal.k4 * t8
        + 11.0 * cal.k5 * t10);
}
fn kb_cubic_extension(theta: f32) -> f32 {
    let span = cal.theta_max - cal.theta_trans;
    let u = clamp((theta - cal.theta_trans) / span, 0.0, 1.0);
    let r_trans = kb_forward(cal.theta_trans);
    let r_trans_deriv = kb_forward_deriv(cal.theta_trans);
    let h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0;
    let h10 = u * u * u - 2.0 * u * u + u;
    let h01 = -2.0 * u * u * u + 3.0 * u * u;
    return h00 * r_trans + h10 * span * r_trans_deriv + h01 * cal.r_max;
}
fn kb_radius(theta: f32) -> f32 {
    if (theta <= cal.theta_trans) { return kb_forward(theta); }
    return kb_cubic_extension(theta);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    // `output_hfov_rad` is the output half-FOV (radians); π/2 = 180°
    // VR180, ~1.812 = 207.68° OSV-lens-full.
    let u = (f32(gid.x) + 0.5) / f32(out_dim.x);
    let v = (f32(gid.y) + 0.5) / f32(out_dim.y);
    let lon = (u - 0.5) * 2.0 * cal.output_hfov_rad;
    let lat = (0.5 - v) * PI;

    let cos_lat = cos(lat);
    let dir = vec3<f32>(
        cos_lat * sin(lon),
        sin(lat),
        cos_lat * cos(lon),
    );

    let xn = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    let yn = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    let zn = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

    // Don't reject `zn <= 0` — Python's MLX kernel
    // (`vr180_gui.py:1387`) accepts the full range and lets the
    // theta_max clamp sample the rim. Without this we get black blocks
    // in the periphery whenever stab rotates the view.

    let cos_theta = clamp(zn, -1.0, 1.0);
    var theta = acos(cos_theta);
    let sin_theta = sqrt(max(0.0, 1.0 - zn * zn));

    // Match Python MLX kernel `vr180_gui.py:1388`:
    // `theta = metal::min(theta, theta_max);` — clamp, don't reject.
    theta = min(theta, cal.theta_max);

    let r_px = kb_radius(theta);
    var cos_phi: f32 = 1.0;
    var sin_phi: f32 = 0.0;
    if (sin_theta > 1e-6) {
        cos_phi = xn / sin_theta;
        sin_phi = yn / sin_theta;
    }

    // Brown-Conrady tangential distortion (p1,p2) — DJI applies this to the
    // normalized point AFTER the radial KB. p1=p2=0 → identical to before.
    let theta_d = r_px / cal.fx;          // r_px = fx · θ_d
    let u0 = theta_d * cos_phi;
    let v0 = theta_d * sin_phi;
    let r2 = u0 * u0 + v0 * v0;            // = θ_d²
    let ut = u0 + 2.0 * cal.p1 * u0 * v0 + cal.p2 * (r2 + 2.0 * u0 * u0);
    let vt = v0 + cal.p1 * (r2 + 2.0 * v0 * v0) + 2.0 * cal.p2 * u0 * v0;
    let src_x = cal.cx + cal.fx * ut;
    let src_y = cal.cy - cal.fy * vt;

    // Out-of-frame coords are clamped by the ClampToEdge sampler
    // (matches Python MLX kernel's per-pixel clamp at lines 1447-1450).

    // ── Sample Y at full res, UV at half res ────────────────────────
    //
    // Match the chroma-siting convention used in `nv12_to_eac_cross.wgsl`:
    //   y_uv  = (sx + 0.5) / sw             — pixel-centre sampling on Y
    //   uv_uv = (sx * 0.5 + 0.5) / (sw / 2) — bilinear of the 2×2 block
    //                                         centred on the matching
    //                                         half-res chroma sample.
    // The normalised UV coords are the same space regardless of plane
    // resolution, but the half-pixel offset differs because the UV
    // plane is half-res in stream-pixel terms.
    let sw = cal.src_w;
    let sh = cal.src_h;
    let stream_xy = vec2<f32>(src_x, src_y);
    let y_uv  = (stream_xy + vec2<f32>(0.5)) / vec2<f32>(sw, sh);
    let uv_uv = (stream_xy * 0.5 + vec2<f32>(0.5)) / vec2<f32>(sw * 0.5, sh * 0.5);

    let y_sample  = textureSampleLevel(fisheye_y,  fisheye_smp, y_uv,  0.0).r;
    let uv_sample = textureSampleLevel(fisheye_uv, fisheye_smp, uv_uv, 0.0).rg;
    let rgb = yuv_to_rgb_bt709_p010(y_sample, uv_sample.r, uv_sample.g);

    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(rgb, 1.0));
}
