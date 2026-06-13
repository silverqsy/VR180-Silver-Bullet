// EAC → half-equirect projection (one lens, one frame).
//
// Inputs:
//   binding(0): the EAC cross texture (cross_w × cross_w, rgba8unorm)
//   binding(1): bilinear sampler
//   binding(2): storage texture for the equirect output (rgba8unorm)
//   binding(3): EquirectUniforms — per-frame R matrix for stabilization
//   binding(4): RsUniforms — per-frame RS correction data (Phase D)
//
// One thread per output pixel. The math mirrors the Python
// `FrameExtractor.build_cross_remap`:
//
//   * convert (u, v) ∈ [0,1]² to an equidistant fisheye direction
//   * recover unit direction vector (xn, yn, zn)
//   * **apply per-frame rotation R** (Phase A stabilization).
//   * **apply per-pixel RS small-angle rotation** (Phase D)
//   * pick one of 5 EAC faces (front / right / left / top / bottom)
//     by max-axis test
//   * within the chosen face, recover (u_eac, v_eac) via arctan
//   * scale to the absolute pixel coordinate in the assembled cross
//   * sample with bilinear filtering
//
// Per-frame R uniform: when stabilization is off, R = identity and
// this is a no-op. When on, R = quat_to_mat3(CORI_frame_i) — rotates
// the output direction into the camera frame at frame i, so the EAC
// face sample lands where the scene actually was when that pixel was
// captured. Net effect: scene appears stationary while the camera
// pans/tilts/rolls.
//
// Phase D RS uniform: when `rs.srot_s == 0` the RS correction collapses
// to a no-op. When non-zero, for each pixel we:
//   1. Compute the polar angle θ from the optical axis (+Z).
//   2. Evaluate the KLNS polynomial `r = c0·θ + c1·θ³ + ... + c4·θ⁹`
//      to recover the original fisheye radius (in sensor pixels).
//   3. Recover the sensor Y coordinate from the direction's Y
//      component: `sensor_y = (cal_dim/2 + ctry) - r·y/sin(θ)`.
//   4. Normalize to a time offset in `[-SROT/2, +SROT/2]` seconds.
//   5. Apply a small-angle 3D rotation by `ω · t_offset` (the
//      pre-multiplied effective angular velocity in shader frame).
// This compensates for the yaw-modded right eye's reversed firmware
// RS — same math as `apply_rs_correction` in the Python reference.

@group(0) @binding(0) var cross_tex: texture_2d<f32>;
@group(0) @binding(1) var cross_smp: sampler;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
// std140 layout for a row-major 3×3 rotation matrix. We use 12
// individual `f32` fields (3 rows × 4 — the 4th column per row is
// std140 padding) because naga (wgpu 0.20's WGSL parser) silently
// rejects struct field access on `mat3x3<f32>` and `array<vec4<f32>, 3>`
// uniforms. Plain scalar fields parse fine in every wgpu version
// we've tested.
struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
// Renamed to `equ` to avoid shadowing the local `let u = ...` below
// for the equirect U coordinate.
@group(0) @binding(3) var<uniform> equ: EquirectUniforms;

// RS uniforms — 12 scalars + 24 vec4s of per-scanline ω groups
// (96 floats, flat-packed 4-per-vec4: group g component c at flat
// index g*3+c → groups[idx/4][idx%4]). `srot_s == 0` disables.
// `n_groups >= 2` → the kernel linearly interpolates the instantaneous
// ω across the readout window by each pixel's sensor-row time
// (Python's per-scanline RS); `n_groups <= 1` → constant `omega_*`
// (Python-preview-equivalent).
// `omega.{x,y,z}` are pre-multiplied effective angular velocities
// (already × pitch/yaw/roll factors) in rad/s, shader frame.
// `klns_c0..c4` are the Kannala-Brandt polynomial coefficients of
// this eye's lens. `ctry` is the principal-point Y offset (pixels),
// `cal_dim` is the calibration sensor dimension (pixels, square).
struct RsUniforms {
    omega_x: f32, omega_y: f32, omega_z: f32, srot_s: f32,
    klns_c0: f32, klns_c1: f32, klns_c2: f32, klns_c3: f32,
    klns_c4: f32, ctry: f32, cal_dim: f32, n_groups: f32,
    groups: array<vec4<f32>, 24>,
}
@group(0) @binding(4) var<uniform> rs: RsUniforms;

// Fetch ω group g (shader-order vec3) from the flat-packed buffer.
fn rs_group(g: i32) -> vec3<f32> {
    let base = g * 3;
    let x = rs.groups[base / 4][base % 4];
    let y = rs.groups[(base + 1) / 4][(base + 1) % 4];
    let z = rs.groups[(base + 2) / 4][(base + 2) % 4];
    return vec3<f32>(x, y, z);
}

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;
const TWO_OVER_PI: f32 = 0.63661977236;  // 2/π

// Cross-coordinate constants. All face boundaries derive from these:
//   * `tile_w = cross_dim / 3.90476…` is not stable across camera variants,
//   * but `center_w = 1920` is fixed, so we compute tile_w on-shader:
//        tile_w = (cross_dim - center_w) / 2  // because cross_dim = 2 * tile_w + center_w
const CENTER_W: f32 = 1920.0;
// Output fisheye half-FOV for `.360`: 185° equidistant (92.5° half).
// The GoPro Max captures 184.5° per eye in the EAC; 185° gives the disk
// edge a hair of margin while staying faithful to the lens FOV — NOT the
// OSV/DJI 195° (that lens is wider).
const FISHEYE_HALF_FOV: f32 = 1.61442956;

fn cross_dim() -> f32 {
    let sz = textureDimensions(cross_tex);
    // Square cross; either dim works.
    return f32(sz.x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) {
        return;
    }

    // EQUIDISTANT circular-fisheye output (FISHEYE_FULL_FOV full FOV):
    // the inscribed circle is the visible disk, corners (r_norm > 1) are
    // black so the result is a real fisheye. θ = r_norm · half_fov. Same
    // output→dir mapping as the OSV fisheye-output shader, so a `.360`
    // fisheye export matches the OSV fisheye export's framing.
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
    let theta_out = r_norm * FISHEYE_HALF_FOV;
    let s_t = sin(theta_out);
    // Unit direction in the OUTPUT frame (X=right, Y=up, Z=forward),
    // rotated into the camera frame by R below — same as the equirect
    // variant from here on (RS + EAC face sampling are identical).
    let dir_world = vec3<f32>(
        s_t * cos(phi_out),
        s_t * sin(phi_out),
        cos(theta_out),
    );
    // R · dir — manual row-vector dot products. 12-scalar uniform
    // layout (see EquirectUniforms above); each row is 3 floats +
    // 1 pad.
    let xr = equ.r00 * dir_world.x + equ.r01 * dir_world.y + equ.r02 * dir_world.z;
    let yr = equ.r10 * dir_world.x + equ.r11 * dir_world.y + equ.r12 * dir_world.z;
    let zr = equ.r20 * dir_world.x + equ.r21 * dir_world.y + equ.r22 * dir_world.z;

    // Phase D: per-pixel rolling-shutter small-angle rotation. When
    // `rs.srot_s == 0`, skip entirely — the (xn, yn, zn) below equal
    // (xr, yr, zr) and this is a pure pass-through.
    var xn: f32 = xr;
    var yn: f32 = yr;
    var zn: f32 = zr;
    if (rs.srot_s > 0.0) {
        // Polar angle from optical axis (+Z) — computed from the
        // PRE-rotation output direction (`dir_world`), matching the
        // Python pipeline, which precomputes the per-pixel readout
        // time from the identity (unrotated) direction grid ("R is
        // always a small rotation" approximation). Keeping the same
        // basis keeps the two apps' RS per-pixel timing identical.
        let cos_theta = clamp(dir_world.z, -1.0, 1.0);
        let theta = acos(cos_theta);
        let sin_theta = sqrt(max(0.0, 1.0 - dir_world.z * dir_world.z));
        // KLNS polynomial: r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹.
        // Horner form for stability.
        let theta2 = theta * theta;
        let r_klns = theta * (rs.klns_c0
            + theta2 * (rs.klns_c1
                + theta2 * (rs.klns_c2
                    + theta2 * (rs.klns_c3
                        + theta2 * rs.klns_c4))));
        // Sensor Y coordinate (row, 0 = top, cal_dim = bottom).
        // At θ → 0 the limit of r / sin(θ) is c0, so we branch there.
        let center_y = rs.cal_dim * 0.5 + rs.ctry;
        var sensor_y: f32;
        if (sin_theta > 1e-6) {
            sensor_y = center_y - r_klns * dir_world.y / sin_theta;
        } else {
            sensor_y = center_y - rs.klns_c0 * dir_world.y;
        }
        // Normalized time offset ∈ [-0.5, +0.5] → seconds.
        let t_norm = sensor_y / rs.cal_dim - 0.5;
        let t = t_norm * rs.srot_s;
        // Instantaneous ω at this pixel's row-time. Per-scanline mode
        // (n_groups ≥ 2): lerp between the 32 RAW-gyro row groups that
        // span [-srot/2, +srot/2] — captures angular acceleration
        // WITHIN the readout window. Constant mode: single smoothed ω.
        var w: vec3<f32>;
        if (rs.n_groups > 1.5) {
            let n1 = rs.n_groups - 1.0;
            let gf = clamp((t / rs.srot_s + 0.5) * n1, 0.0, n1);
            let g0 = i32(floor(gf));
            let g1 = min(g0 + 1, i32(n1));
            let a = gf - f32(g0);
            w = mix(rs_group(g0), rs_group(g1), a);
        } else {
            w = vec3<f32>(rs.omega_x, rs.omega_y, rs.omega_z);
        }
        // Small-angle 3D rotation by ω · t. `w` already includes the
        // rs factors (multiplied on CPU side).
        let wx = w.x * t;
        let wy = w.y * t;
        let wz = w.z * t;
        xn = xr + wy * zr - wz * yr;
        yn = yr + wz * xr - wx * zr;
        zn = zr - wy * xr + wx * yr;
    }

    let ax = abs(xn);
    let ay = abs(yn);
    let az = abs(zn);

    // Cross layout derived from input texture dimensions:
    //   tile_w = (cross_dim - 1920) / 2
    //   center_top = tile_w,  center_bot = tile_w + 1920
    let cdim = cross_dim();
    let tw = (cdim - CENTER_W) * 0.5;
    let center_top = tw;
    let center_bot = tw + CENTER_W;

    // EDGE FILL: the per-eye EAC cross only stores the captured ~180°
    // (front face + a `tile_w` strip of each side/top/bottom face); rays
    // past the lens FOV land beyond that strip. Rather than writing black
    // there (a hard border), we CLAMP the side-tile coordinate to the
    // last captured column/row and sample the rim — identical to Python's
    // `edge_fill` (the numba/MLX kernels clamp `fc/pc/pr/fr` to
    // `[0, tile_w-1]` and always sample) and to the OSV path's
    // ClampToEdge rim smear. `hit` stays false only if no face matched
    // (impossible for a unit vector — a paranoia fallback).
    var hit = false;
    var sx: f32 = 0.0;  // pixel x in cross (absolute, before 0.5 sample offset)
    var sy: f32 = 0.0;
    let edge = tw - 1.0;  // last captured column/row in a side tile

    // ── Front face ── (z > 0, |x| ≤ z, |y| ≤ z) ───────────────────────
    if (zn > 0.0 && ax <= zn && ay <= zn) {
        let u_eac = TWO_OVER_PI * atan(xn / zn) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(yn / zn);
        sx = center_top + u_eac * CENTER_W;
        sy = center_top + v_eac * CENTER_W;
        hit = true;
    }
    // ── Right face ── (x > 0, z ≤ x, |y| ≤ x) ────────────────────────
    else if (xn > 0.0 && zn <= xn && ay <= xn) {
        let u_eac = TWO_OVER_PI * atan(-zn / xn) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(yn / xn);
        let full_col = clamp(u_eac * CENTER_W, 0.0, edge);
        sx = center_bot + full_col;
        sy = center_top + v_eac * CENTER_W;
        hit = true;
    }
    // ── Left face ── (x < 0, z ≤ |x|, |y| ≤ |x|) ─────────────────────
    else if (xn < 0.0 && zn <= ax && ay <= ax) {
        let u_eac = TWO_OVER_PI * atan(zn / ax) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(yn / ax);
        let partial_col = clamp(u_eac * CENTER_W - (CENTER_W - tw), 0.0, edge);
        sx = partial_col;
        sy = center_top + v_eac * CENTER_W;
        hit = true;
    }
    // ── Top face ── (y > 0, |x| ≤ y, z ≤ y) ──────────────────────────
    else if (yn > 0.0 && ax <= yn && zn <= yn) {
        let u_eac = TWO_OVER_PI * atan(xn / yn) + 0.5;
        let v_eac = TWO_OVER_PI * atan(zn / yn) + 0.5;
        let partial_row = clamp(v_eac * CENTER_W - (CENTER_W - tw), 0.0, edge);
        sx = center_top + u_eac * CENTER_W;
        sy = partial_row;
        hit = true;
    }
    // ── Bottom face ── (y < 0, |x| ≤ |y|, z ≤ |y|) ───────────────────
    else if (yn < 0.0 && ax <= ay && zn <= ay) {
        let u_eac = TWO_OVER_PI * atan(xn / ay) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(zn / ay);
        let full_row = clamp(v_eac * CENTER_W, 0.0, edge);
        sx = center_top + u_eac * CENTER_W;
        sy = center_bot + full_row;
        hit = true;
    }

    var color: vec4<f32>;
    if (hit) {
        let uv = vec2<f32>((sx + 0.5) / cdim, (sy + 0.5) / cdim);
        color = textureSampleLevel(cross_tex, cross_smp, uv, 0.0);
    } else {
        color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
