// Fisheye source → stabilized fisheye output (one eye).
//
// Same as `fisheye_to_hequirect.wgsl` from step 3 onward — only step 1
// differs: instead of unpacking (u, v) → (lon, lat) for an equirect
// output we unpack (u, v) → (θ_out, φ_out) for a circular EQUIDISTANT
// fisheye output of user-chosen FOV (`output_hfov_rad`). Every output
// pixel within the inscribed unit circle maps to one ray; output corners
// (r_norm > 1) are written as black so the result looks like a real
// circular-fisheye frame.
//
// Stabilization + per-eye view adjustment work exactly as they do in the
// half-equirect path: the caller composes the per-frame stab matrix with
// the per-eye view-adjust matrix CPU-side and passes the product in
// `equ`. The shader applies it to the output ray before the source KB
// forward-projection.

@group(0) @binding(0) var fisheye_tex: texture_2d<f32>;
@group(0) @binding(1) var fisheye_smp: sampler;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(3) var<uniform> equ: EquirectUniforms;

struct FisheyeCalibUniforms {
    fx: f32, fy: f32, cx: f32, cy: f32,
    k1: f32, k2: f32, k3: f32, k4: f32,
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,
    p1: f32, p2: f32, _pad3: f32, _pad4: f32,
}
@group(0) @binding(4) var<uniform> cal: FisheyeCalibUniforms;

const PI: f32 = 3.14159265359;

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

    // Normalize output pixel position to (-1, +1) on each axis. The
    // inscribed unit circle maps to the visible fisheye disk; r > 1
    // produces black corners (real-fisheye look).
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

    // Apply per-frame stab × per-eye view-adjust rotation.
    let xn = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    let yn = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    let zn = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

    // Source elevation / azimuth.
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

    let uv = vec2<f32>(src_x / cal.src_w, src_y / cal.src_h);
    let color = textureSampleLevel(fisheye_tex, fisheye_smp, uv, 0.0);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
