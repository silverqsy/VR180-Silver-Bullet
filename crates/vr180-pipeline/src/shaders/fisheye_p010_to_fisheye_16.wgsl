// P010 → fisheye-output (Rgba16Unorm) zero-copy projection.
//
// Same per-pixel math as `fisheye_to_fisheye_16.wgsl` (equidistant
// fisheye output, KB source projection), with Y+UV plane sampling and
// BT.709 limited-range YCbCr→RGB inline so VT-decoded IOSurface bytes
// flow straight into the Rgba16Unorm output. Used by the 10-bit OSV
// fisheye-output export path.

@group(0) @binding(0) var fisheye_y:   texture_2d<f32>;
@group(0) @binding(1) var fisheye_uv:  texture_2d<f32>;
@group(0) @binding(2) var fisheye_smp: sampler;
@group(0) @binding(3) var out_tex:     texture_storage_2d<rgba16unorm, write>;

struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(4) var<uniform> equ: EquirectUniforms;

struct FisheyeCalibUniforms {
    fx: f32, fy: f32, cx: f32, cy: f32,
    k1: f32, k2: f32, k3: f32, k4: f32,
    theta_trans: f32, theta_max: f32, r_max: f32, _pad0: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,
}
@group(0) @binding(5) var<uniform> cal: FisheyeCalibUniforms;

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
    let inner = 1.0 + t2 * (cal.k1 + t2 * (cal.k2 + t2 * (cal.k3 + t2 * cal.k4)));
    return cal.fx * theta * inner;
}
fn kb_forward_deriv(theta: f32) -> f32 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    return cal.fx * (1.0
        + 3.0 * cal.k1 * t2
        + 5.0 * cal.k2 * t4
        + 7.0 * cal.k3 * t6
        + 9.0 * cal.k4 * t8);
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
// Invert KB: given a source-pixel radius, solve for the ray angle theta
// via Newton-Raphson. Used by the fisheye-output parametrisation so the
// OUTPUT uses the source lens's own projection (identity rotation → 1:1).
fn kb_inverse(r_target: f32) -> f32 {
    if (r_target <= 0.0) { return 0.0; }
    var theta = r_target / max(cal.fx, 1.0);   // paraxial seed
    for (var i = 0; i < 8; i = i + 1) {
        let f = kb_forward(theta) - r_target;
        let d = kb_forward_deriv(theta);
        theta = theta - f / max(d, 1e-4);
        theta = clamp(theta, 0.0, cal.theta_max);
    }
    return theta;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    // Output uses the SOURCE lens's own KB projection (not a synthetic
    // equidistant one): map this output pixel into source-pixel space,
    // invert KB to get the ray angle, so an identity rotation reproduces
    // the source fisheye 1:1 — exactly the original projection and FOV.
    let scale_x = cal.src_w / f32(out_dim.x);
    let scale_y = cal.src_h / f32(out_dim.y);
    // Output is a CENTERED fisheye whose geometric center represents the
    // lens optical axis. Distances are measured from the output center
    // (converted to source pixels) and KB-inverted with the source
    // focal — so the projection + FOV match the source lens. The applied
    // principal point (cal.cx/cal.cy) is honored via the source sampling
    // below: it sets WHERE in the source the optical axis sits, so
    // changing cx/cy shifts the output accordingly.
    let dx = ((f32(gid.x) + 0.5) - f32(out_dim.x) * 0.5) * scale_x;
    let dy = (f32(out_dim.y) * 0.5 - (f32(gid.y) + 0.5)) * scale_y;
    let dyp = dy * (cal.fx / max(cal.fy, 1e-6));
    let r_out = sqrt(dx * dx + dyp * dyp);
    if (r_out > cal.r_max) {
        textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let phi_out = atan2(dyp, dx);
    let theta_out = kb_inverse(r_out);
    let s_t = sin(theta_out);
    let c_t = cos(theta_out);
    let dir = vec3<f32>(s_t * cos(phi_out), s_t * sin(phi_out), c_t);

    let xn = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    let yn = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    let zn = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

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
    let src_x = cal.cx + r_px * cos_phi;
    let src_y = cal.cy - r_px * sin_phi * (cal.fy / cal.fx);

    // Two-plane sampling — Y at full res, UV at half res. Same
    // chroma-siting convention as fisheye_p010_to_hequirect.wgsl.
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
