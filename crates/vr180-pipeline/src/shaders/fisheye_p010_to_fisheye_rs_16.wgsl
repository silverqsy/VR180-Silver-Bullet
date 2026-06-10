// P010 → stabilized fisheye output (Rgba16Unorm) WITH per-row rolling-
// shutter correction. Zero-copy export path for OSV fisheye output at
// 10-bit.
//
// Combines the equidistant fisheye-output parametrisation of
// `fisheye_p010_to_fisheye_16.wgsl` with the per-row RS warp of
// `fisheye_p010_to_hequirect_rs.wgsl`. RS operates on the source-frame
// direction after per-frame stab, independent of the output mapping.
//
// Bindings:
//   (0) fisheye_y  — R16Unorm  (full-res Y)
//   (1) fisheye_uv — Rg16Unorm (half-res Cb/Cr)
//   (2) fisheye_smp— bilinear sampler
//   (3) out_tex    — Rgba16Unorm storage (fisheye output)
//   (4) equ        — per-frame stab rotation
//   (5) cal        — per-eye KB calibration
//   (6) rs_rows    — storage buffer: array<RsRowR>, one per scanline

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
    fx: f32, fy: f32, cx: f32, cy: f32,
    k1: f32, k2: f32, k3: f32, k4: f32,
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,
    p1: f32, p2: f32, _pad3: f32, _pad4: f32,
}
@group(0) @binding(5) var<uniform> cal: FisheyeCalibUniforms;

struct RsRowR {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(6) var<storage, read> rs_rows: array<RsRowR>;

const PI: f32 = 3.14159265359;

fn yuv_to_rgb_bt709_p010(y: f32, u: f32, v: f32) -> vec3<f32> {
    let y_l = y * (65535.0 / 56064.0) - (64.0 / 876.0);
    let u_l = u * (65535.0 / 57344.0) - (512.0 / 896.0);
    let v_l = v * (65535.0 / 57344.0) - (512.0 / 896.0);
    let r = y_l + 1.5748 * v_l;
    let g = y_l - 0.1873 * u_l - 0.4681 * v_l;
    let b = y_l + 1.8556 * u_l;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

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
fn project_kb(xn: f32, yn: f32, zn: f32) -> vec2<f32> {
    let cos_theta = clamp(zn, -1.0, 1.0);
    var theta = acos(cos_theta);
    let sin_theta = sqrt(max(0.0, 1.0 - zn * zn));
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
    return vec2<f32>(src_x, src_y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    // NORMALIZED equidistant circular-fisheye output: the inscribed circle
    // of the (square) output frame maps LINEARLY to ray angle, with the
    // circle edge at the output half-FOV (cal.output_hfov_rad). A canonical
    // projection — independent of the source lens's own distortion — so the
    // result is a standard equidistant fisheye (195° full FOV). The output
    // is centered on the optical axis; cal.cx/cal.cy locate that axis in the
    // SOURCE during sampling below, so changing them shifts the source tap.
    let half = min(f32(out_dim.x), f32(out_dim.y)) * 0.5;
    let dx = (f32(gid.x) + 0.5) - f32(out_dim.x) * 0.5;
    let dy = f32(out_dim.y) * 0.5 - (f32(gid.y) + 0.5);
    let r_norm = sqrt(dx * dx + dy * dy) / max(half, 1.0);
    if (r_norm > 1.0) {
        textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let phi_out = atan2(dy, dx);
    let theta_out = r_norm * cal.output_hfov_rad;   // equidistant: θ = ρ · θ_max
    let s_t = sin(theta_out);
    let c_t = cos(theta_out);
    let dir = vec3<f32>(s_t * cos(phi_out), s_t * sin(phi_out), c_t);

    var rx = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    var ry = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    var rz = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

    var src = project_kb(rx, ry, rz);

    let fish_h_i = i32(cal.src_h);
    let row_idx = clamp(i32(round(src.y)), 0, fish_h_i - 1);
    let rsm = rs_rows[row_idx];
    if (rsm.r00 != 0.0) {
        let rsx = rsm.r00 * rx + rsm.r01 * ry + rsm.r02 * rz;
        let rsy = rsm.r10 * rx + rsm.r11 * ry + rsm.r12 * rz;
        let rsz = rsm.r20 * rx + rsm.r21 * ry + rsm.r22 * rz;
        src = project_kb(rsx, rsy, rsz);
    }

    let sw = cal.src_w;
    let sh = cal.src_h;
    let stream_xy = src;
    let y_uv  = (stream_xy + vec2<f32>(0.5)) / vec2<f32>(sw, sh);
    let uv_uv = (stream_xy * 0.5 + vec2<f32>(0.5)) / vec2<f32>(sw * 0.5, sh * 0.5);

    let y_sample  = textureSampleLevel(fisheye_y,  fisheye_smp, y_uv,  0.0).r;
    let uv_sample = textureSampleLevel(fisheye_uv, fisheye_smp, uv_uv, 0.0).rg;
    let rgb = yuv_to_rgb_bt709_p010(y_sample, uv_sample.r, uv_sample.g);

    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(rgb, 1.0));
}
