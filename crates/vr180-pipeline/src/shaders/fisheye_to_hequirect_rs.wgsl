// Fisheye → half-equirect WITH per-row rolling-shutter correction.
//
// This is the RGBA-input variant of `fisheye_p010_to_hequirect_rs.wgsl`,
// used by the GUI preview path (which decodes to RGBA via the older
// CPU/GPU pipeline rather than the export's P010 zero-copy path).
//
// Math is identical to `fisheye_to_hequirect.wgsl` plus an extra
// per-row direction warp:
//
//   1. Output pixel → equirect direction → R_stab · dir (per-frame
//      stabilization).
//   2. KB-project to find the initial source-pixel `v` — this tells
//      us which scanline the sensor was reading at the moment that
//      pixel's photons arrived.
//   3. Look up that row's R_rs matrix from a storage buffer. R_rs is
//      `R_cam(q_row⁻¹ ⊗ q_mid)` (mid-frame → this-row orientation),
//      already in camera frame.
//   4. Apply R_rs to the stab-rotated direction and re-project.
//   5. Sample the fisheye RGBA at the resulting (u, v).
//
// "RS off" convention: the buffer's first matrix has R00 == 0.0 →
// shader skips the per-row pass.

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
    theta_trans: f32, theta_max: f32, r_max: f32, _pad0: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,
}
@group(0) @binding(4) var<uniform> cal: FisheyeCalibUniforms;

// Per-row rotation matrix in std430 layout.
struct RsRowR {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(5) var<storage, read> rs_rows: array<RsRowR>;

const PI: f32 = 3.14159265359;

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
    let src_x = cal.cx + r_px * cos_phi;
    let src_y = cal.cy - r_px * sin_phi * (cal.fy / cal.fx);
    return vec2<f32>(src_x, src_y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }

    let u = (f32(gid.x) + 0.5) / f32(out_dim.x);
    let v = (f32(gid.y) + 0.5) / f32(out_dim.y);
    let lon = (u - 0.5) * 2.0 * cal.output_hfov_rad;
    let lat = (0.5 - v) * PI;
    let cos_lat = cos(lat);
    let dir = vec3<f32>(cos_lat * sin(lon), sin(lat), cos_lat * cos(lon));

    // Per-frame stab rotation.
    var rx = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    var ry = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    var rz = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

    // First projection — find which scanline this pixel maps to.
    var src = project_kb(rx, ry, rz);

    // Per-row RS correction.
    let fish_h_i = i32(cal.src_h);
    let row_idx = clamp(i32(round(src.y)), 0, fish_h_i - 1);
    let rsm = rs_rows[row_idx];
    if (rsm.r00 != 0.0) {
        let rsx = rsm.r00 * rx + rsm.r01 * ry + rsm.r02 * rz;
        let rsy = rsm.r10 * rx + rsm.r11 * ry + rsm.r12 * rz;
        let rsz = rsm.r20 * rx + rsm.r21 * ry + rsm.r22 * rz;
        src = project_kb(rsx, rsy, rsz);
    }

    // Sample as normalized texture coords.
    let uv = vec2<f32>(src.x / cal.src_w, src.y / cal.src_h);
    let color = textureSampleLevel(fisheye_tex, fisheye_smp, uv, 0.0);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
