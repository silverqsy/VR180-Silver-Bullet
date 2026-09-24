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
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, src_x0: f32,
    p1: f32, p2: f32, xi: f32, src_proj: f32,     // xi > 0 selects the unified camera model; src_proj > 0.5 = half-equirect input (no lens model)
    ta: f32, tb: f32, tc: f32, te: f32,            // UCM tangential: (r²+2x²)(ta + tc r²) + 2xy(tb + te r²)
    s1: f32, s2: f32, s3: f32, s4: f32,            // UCM thin prism: x += s1 r² + s2 r⁴, y += s3 r² + s4 r⁴
    // vec4 #7 — reframed-view output; unused by this kernel, declared so
    // vec4 #8 lands at the same offset as in the Rust struct.
    proj_mode: f32, defish_k: f32, edge_x: f32, edge_y: f32,
    // vec4 #8 — source sample range expansion (y_scale, y_off, c_scale, c_off).
    yuv_range: vec4<f32>,
}
@group(0) @binding(5) var<uniform> cal: FisheyeCalibUniforms;

const PI: f32 = 3.14159265359;

fn yuv_to_rgb_bt709_p010(y: f32, u: f32, v: f32) -> vec3<f32> {
    // Range expansion comes from the uniform — (y_scale, y_off, c_scale,
    // c_off) for P010-in-16 or NV12-in-8, limited or full range; see
    // `yuv_range_constants` in gpu.rs. The BT.709 matrix below is unchanged.
    let y_l = y * cal.yuv_range.x + cal.yuv_range.y;
    let u_l = u * cal.yuv_range.z + cal.yuv_range.w;
    let v_l = v * cal.yuv_range.z + cal.yuv_range.w;
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
    // Brown-Conrady tangential distortion (p1,p2) — DJI applies this to the
    // normalized point AFTER the radial KB. p1=p2=0 → identical to before.
    var src_x: f32;
    var src_y: f32;
    if (cal.src_proj > 0.5) {
        // Half-equirect input (a VR180 SBS file that is already dewarped —
        // the fisheye dewarp toggle is off): the eye spans 180° × 180°, so
        // the rotated ray maps straight to (lon, lat) with no lens model.
        // Identity resample for the default 180° output + identity rotation.
        let lon = atan2(xn, zn);
        let lat = asin(clamp(yn, -1.0, 1.0));
        src_x = (0.5 + lon / PI) * cal.src_w;
        src_y = (0.5 - lat / PI) * cal.src_h;
    } else if (cal.xi > 0.0) {
        // Unified camera model (Insta360 factory calibration): the ray is
        // projected onto the plane n = sinθ / (ξ + cosθ); then an even radial
        // polynomial in n² (k1..k5), tangential terms whose strength grows
        // with r² (ta,tb ; tc,te) and thin-prism terms (s1..s4) are applied
        // in image (y-down) coordinates. Matches Insta360 Studio's output.
        let ct = cos(theta);
        let st = sin(theta);
        let n = st / (cal.xi + ct);
        let x = n * cos_phi;
        let y = -n * sin_phi;                  // image y points down
        let r2 = n * n;
        let dd = 1.0 + r2 * (cal.k1 + r2 * (cal.k2 + r2 * (cal.k3 + r2 * (cal.k4 + r2 * cal.k5))));
        let xy2 = 2.0 * x * y;
        let xd = x * dd + (r2 + 2.0 * x * x) * (cal.ta + cal.tc * r2) + xy2 * (cal.tb + cal.te * r2)
               + cal.s1 * r2 + cal.s2 * r2 * r2;
        let yd = y * dd + (r2 + 2.0 * y * y) * (cal.tb + cal.te * r2) + xy2 * (cal.ta + cal.tc * r2)
               + cal.s3 * r2 + cal.s4 * r2 * r2;
        src_x = cal.cx + cal.fx * xd;
        src_y = cal.cy + cal.fy * yd;
    } else {
        // Kannala-Brandt radial (+ rim extension), then Brown-Conrady
        // tangential distortion (p1,p2) on the normalized point — DJI applies
        // this AFTER the radial KB. p1=p2=0 → pure KB.
        let theta_d = r_px / cal.fx;          // r_px = fx · θ_d
        let u0 = theta_d * cos_phi;
        let v0 = theta_d * sin_phi;
        let r2 = u0 * u0 + v0 * v0;            // = θ_d²
        let ut = u0 + 2.0 * cal.p1 * u0 * v0 + cal.p2 * (r2 + 2.0 * u0 * u0);
        let vt = v0 + cal.p1 * (r2 + 2.0 * v0 * v0) + 2.0 * cal.p2 * u0 * v0;
        src_x = cal.cx + cal.fx * ut;
        src_y = cal.cy - cal.fy * vt;
    }

    // Two-plane sampling — Y at full res, UV at half res. Same
    // chroma-siting convention as fisheye_p010_to_hequirect.wgsl.
    // Eye sub-rect: `src_w`/`src_h` are the EYE's dims (the lens model and
    // the half-equirect mapping normalise by them); the planes may be wider —
    // a generic side-by-side frame holds both eyes in ONE texture and this
    // eye starts at `src_x0`. Clamp in TEXEL space to the eye's own rect
    // BEFORE offsetting: the ClampToEdge sampler only knows the texture's
    // edge, and an out-of-rect tap — or a chroma tap on the eye's last
    // column, which sits exactly on the boundary — would otherwise
    // bilinear-blend the OTHER eye in. Dual-stream sources pass src_x0 = 0
    // with eye == plane, so the clamp reduces to ClampToEdge: bit-identical.
    let sw = cal.src_w;
    let sh = cal.src_h;
    let stream_xy = vec2<f32>(src_x, src_y);
    let plane_y  = vec2<f32>(textureDimensions(fisheye_y));
    let plane_uv = vec2<f32>(textureDimensions(fisheye_uv));
    let y_px  = clamp(stream_xy + vec2<f32>(0.5),
                      vec2<f32>(0.5), vec2<f32>(sw - 0.5, sh - 0.5))
              + vec2<f32>(cal.src_x0, 0.0);
    let uv_px = clamp(stream_xy * 0.5 + vec2<f32>(0.5),
                      vec2<f32>(0.5), vec2<f32>(sw * 0.5 - 0.5, sh * 0.5 - 0.5))
              + vec2<f32>(cal.src_x0 * 0.5, 0.0);
    let y_uv  = y_px / plane_y;
    let uv_uv = uv_px / plane_uv;

    let y_sample  = textureSampleLevel(fisheye_y,  fisheye_smp, y_uv,  0.0).r;
    let uv_sample = textureSampleLevel(fisheye_uv, fisheye_smp, uv_uv, 0.0).rg;
    let rgb = yuv_to_rgb_bt709_p010(y_sample, uv_sample.r, uv_sample.g);

    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(rgb, 1.0));
}
