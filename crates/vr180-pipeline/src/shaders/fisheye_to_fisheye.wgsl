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
    src_w: f32, src_h: f32, output_hfov_rad: f32, src_x0: f32,
    p1: f32, p2: f32, xi: f32, src_proj: f32,     // xi > 0 selects the unified camera model; src_proj > 0.5 = half-equirect input (no lens model)
    ta: f32, tb: f32, tc: f32, te: f32,            // UCM tangential: (r²+2x²)(ta + tc r²) + 2xy(tb + te r²)
    s1: f32, s2: f32, s3: f32, s4: f32,            // UCM thin prism: x += s1 r² + s2 r⁴, y += s3 r² + s4 r⁴
    proj_mode: f32, defish_k: f32, edge_x: f32, edge_y: f32,  // reframed-view output (proj_mode 1 → k-projection; 0 → half-equirect)
    // Source sample range expansion (y_scale, y_off, c_scale, c_off) — read by the P010 kernels.
    yuv_range: vec4<f32>,
    // This eye's y offset in the plane (stacked top/bottom frames); frame_mask
    // > 0.5 = the frame is a CROP of the image circle (see `frame_cover`).
    src_y0: f32, frame_mask: f32, _p9b: f32, _p9c: f32,
    // YCbCr→RGB coefficients (r_v, g_u, g_v, b_u) — read by the P010 kernels.
    yuv_mat: vec4<f32>,
    // View mask (see `view_mask`): half-extents (radians; 0 = off), corner radius, feather.
    mask_hx: f32, mask_hy: f32, mask_round: f32, mask_feather: f32,
}
@group(0) @binding(4) var<uniform> cal: FisheyeCalibUniforms;

// Coverage of a source tap for a frame that is a crop of the image circle
// (`frame_mask`): 1 inside the frame, 0 outside it, 0 for a ray past the
// lens' clamp angle — there is no data there. A full-circle source
// (frame_mask = 0) keeps replicating its rim instead: always 1. So does a
// crop in the reframed (flat) view: its viewport sits inside the recorded
// view, and the repeated edge is what covers the border the stabilizer
// exposes when it runs out of picture. `px_rad` is one output pixel in
// radians (see `view_mask`).
fn frame_cover(src: vec2<f32>, cos_theta: f32, px_rad: f32) -> f32 {
    if (cal.frame_mask < 0.5 || cal.proj_mode > 0.5) { return 1.0; }
    if (acos(clamp(cos_theta, -1.0, 1.0)) > cal.theta_max) { return 0.0; }
    let d = min(min(src.x, src.y), min(cal.src_w - src.x, cal.src_h - src.y));
    // Soft edge one OUTPUT pixel wide: source pixels per output pixel at the
    // lens centre (paraxial focal = fx / (1 + ξ); ξ = 0 for the KB model).
    let edge = max(px_rad * cal.fx / (1.0 + max(cal.xi, 0.0)), 1.0);
    return clamp(d / edge + 0.5, 0.0, 1.0);
}

// View mask: a rounded rectangle fixed in the OUTPUT, on the equidistant
// angle plane θ·(cos φ, sin φ) around the output's forward axis. `dir` is the
// output ray BEFORE the stabilization rotation, so the window stays put
// while the stabilized image moves under it. `px_rad` is one output pixel in
// radians: a hard edge is antialiased over it. Off in the reframed view.
// CPU twin: `gpu::view_mask_cover`.
fn view_mask(dir: vec3<f32>, px_rad: f32) -> f32 {
    if (cal.mask_hx <= 0.0 || cal.proj_mode > 0.5) { return 1.0; }
    let s = sqrt(max(dir.x * dir.x + dir.y * dir.y, 1e-12));
    let th = acos(clamp(dir.z, -1.0, 1.0));
    let p = abs(vec2<f32>(dir.x, dir.y) * (th / s));
    let q = p - (vec2<f32>(cal.mask_hx, cal.mask_hy) - vec2<f32>(cal.mask_round));
    let sd = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - cal.mask_round;
    let f = max(cal.mask_feather, px_rad);
    return 1.0 - smoothstep(-f, 0.0, sd);
}

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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    // Normalize output pixel position to (-1, +1) on each axis. The
    // inscribed unit circle maps to the visible fisheye disk; r > 1
    // produces black corners (real-fisheye look).
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

    let uv = vec2<f32>(src_x / cal.src_w, src_y / cal.src_h);
    let color = textureSampleLevel(fisheye_tex, fisheye_smp, uv, 0.0);
    let px_rad = cal.output_hfov_rad / max(half, 1.0);
    let cover = frame_cover(vec2<f32>(src_x, src_y), zn, px_rad) * view_mask(dir, px_rad);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color.rgb * cover, color.a));
}
