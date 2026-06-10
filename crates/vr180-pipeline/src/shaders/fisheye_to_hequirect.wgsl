// Fisheye → half-equirect projection (one eye, one frame).
//
// Counterpart of eac_to_equirect.wgsl. This shader is used for the
// non-GoPro family of inputs (DJI Osmo OSV, SBS fisheye .mp4, Blackmagic
// .braw) — anything where the source is a raw fisheye image rather than
// an EAC cross. Math ported from vr180_gui.py:1350-1480 (the Metal
// kernel that processes one fisheye eye).
//
// Per-output-pixel pipeline:
//   1. (u, v) ∈ [0, 1]² → (lon, lat) ∈ [-π/2, +π/2]² (half-equirect)
//   2. Recover unit ray (xn, yn, zn) in the output frame
//   3. Apply per-frame stabilization rotation R (equ.r*)
//   4. Convert to fisheye-image (cx, cy, r, φ):
//        θ = acos(zn)            -- elevation from optical axis
//        φ = atan2(yn, xn)       -- azimuth around it
//        r = KB_forward(θ, fx, k) over the calibrated range, with
//            cubic-Hermite extension past θ_trans
//   5. Sample texture at (cx + r·cos(φ), cy + r·sin(φ)) with bilinear
//
// The KB forward is r = fx·θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸).
// Inversion isn't needed here — we have the world ray, and the KB
// model is forward (ray → pixel). The CPU side does the inverse for
// FOV derivation and tests; the shader stays forward-only.
//
// Cubic Hermite extension: past θ_trans (default 80°) the KB polynomial
// becomes unreliable, especially for ≥190° lenses (DJI Osmo, Blackmagic
// URSA, Canon RF 5.2mm dual). We switch to a cubic that matches r and
// dr/dθ at θ_trans and pins r_max at the image-circle radius (computed
// once on the CPU side from cx/cy + image dimensions). Matches the
// Python kernel at vr180_gui.py:1542-1552.

@group(0) @binding(0) var fisheye_tex: texture_2d<f32>;
@group(0) @binding(1) var fisheye_smp: sampler;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;

// std140 layout for the per-frame stabilization rotation. 12 scalars,
// 3 rows × (3 floats + 1 pad). Same convention as EquirectUniforms in
// eac_to_equirect.wgsl — avoids the naga mat3x3 / array<vec4> bugs.
struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}
@group(0) @binding(3) var<uniform> equ: EquirectUniforms;

// Fisheye calibration uniforms. One block per eye — caller binds the
// left-eye block to the left output, right-eye block to the right.
//
//   fx, fy:       focal length in pixels (KB radial scale)
//   cx, cy:       principal point in pixels (image origin top-left)
//   k1..k4:       KB-4 distortion polynomial coefficients
//   theta_trans:  KB → cubic extension switchover angle (radians,
//                 typically 80°·π/180 ≈ 1.3963)
//   theta_max:    upper bound of the cubic extension (typically
//                 110°·π/180 ≈ 1.9199)
//   r_max:        image-circle radius in pixels (cubic extension pins
//                 r(θ_max) = r_max)
//
// Layout: 4 vec4s = 64 bytes. Caller packs as f32[16].
struct FisheyeCalibUniforms {
    fx: f32, fy: f32, cx: f32, cy: f32,            // vec4 #0
    k1: f32, k2: f32, k3: f32, k4: f32,            // vec4 #1
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,  // vec4 #2
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,  // vec4 #3
    p1: f32, p2: f32, _pad3: f32, _pad4: f32,
}
@group(0) @binding(4) var<uniform> cal: FisheyeCalibUniforms;

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

// ── KB forward polynomial ───────────────────────────────────────────
// r(θ) = fx · θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸).
// Horner form on (θ²) for numerical stability.
fn kb_forward(theta: f32) -> f32 {
    let t2 = theta * theta;
    let inner = 1.0 + t2 * (cal.k1 + t2 * (cal.k2 + t2 * (cal.k3 + t2 * (cal.k4 + t2 * cal.k5))));
    return cal.fx * theta * inner;
}

// dr/dθ for the KB forward. Used to set the Hermite tangent at θ_trans.
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

// Cubic Hermite extension at θ ∈ [θ_trans, θ_max]. Pins r and dr/dθ
// at θ_trans (matching KB) and r at θ_max (= r_max). Leaves dr/dθ
// free at θ_max — most fisheye lenses have very steep falloff there,
// and a zero outgoing tangent would over-flatten the extension. See
// `CubicExtension::forward` in crates/vr180-fisheye/src/projection.rs
// for the matching CPU reference.
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

// End-to-end θ → r dispatch with the switchover at θ_trans.
fn kb_radius(theta: f32) -> f32 {
    if (theta <= cal.theta_trans) {
        return kb_forward(theta);
    }
    return kb_cubic_extension(theta);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = textureDimensions(out_tex);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) {
        return;
    }

    // Per-pixel normalized coords + (lon, lat) for half-equirect.
    // `output_hfov_rad` is the OUTPUT horizontal half-FOV. π/2 gives
    // the standard 180° VR180 layout; larger values (e.g. 1.812 for
    // OSV's 207.68° lens) extend the output to cover the lens periphery
    // that a strict 180° would crop out.
    let u = (f32(gid.x) + 0.5) / f32(out_dim.x);
    let v = (f32(gid.y) + 0.5) / f32(out_dim.y);
    let lon = (u - 0.5) * 2.0 * cal.output_hfov_rad;
    let lat = (0.5 - v) * PI;

    // Unit ray in output frame. Z is the optical axis (forward).
    let cos_lat = cos(lat);
    let dir = vec3<f32>(
        cos_lat * sin(lon),
        sin(lat),
        cos_lat * cos(lon),
    );

    // Apply stabilization rotation R.
    let xn = equ.r00 * dir.x + equ.r01 * dir.y + equ.r02 * dir.z;
    let yn = equ.r10 * dir.x + equ.r11 * dir.y + equ.r12 * dir.z;
    let zn = equ.r20 * dir.x + equ.r21 * dir.y + equ.r22 * dir.z;

    // Don't reject rays past the front hemisphere — Python's MLX
    // kernel at `vr180_gui.py:1387` accepts any rz ∈ [-1, 1] and lets
    // the θ-clamp below cap us at theta_max so we still sample a real
    // (rim) pixel. The previous `if zn <= 0: black` here turned visible
    // peripheral content into black blocks whenever stab rotated the
    // view, contributing to the "different from Python" report.

    // Elevation θ from optical axis, azimuth φ around it.
    let cos_theta = clamp(zn, -1.0, 1.0);
    var theta = acos(cos_theta);
    let sin_theta = sqrt(max(0.0, 1.0 - zn * zn));

    // Past the image circle, clamp θ to theta_max and continue sampling.
    // Matches Python MLX kernel at `vr180_gui.py:1388`:
    //   `theta = metal::min(theta, theta_max);`
    // Out-of-bounds source coords are clamped by the ClampToEdge
    // sampler, so the rim's edge pixel is sampled instead of black.
    theta = min(theta, cal.theta_max);

    // KB forward + extension → pixel radius.
    let r_px = kb_radius(theta);

    // φ azimuth (recovered from xn/yn). Guard the θ → 0 case where
    // sin_theta is tiny: any direction is valid there, snap to +X.
    var cos_phi: f32 = 1.0;
    var sin_phi: f32 = 0.0;
    if (sin_theta > 1e-6) {
        cos_phi = xn / sin_theta;
        sin_phi = yn / sin_theta;
    }

    // Fisheye image coordinates. The Y term is *subtracted* because
    // image coordinates have +Y pointing DOWN while the world frame
    // we computed `yn` in has +Y pointing UP — sampling at (cx + r·cosφ,
    // cy + r·sinφ) without the flip lands every output pixel on the
    // wrong side of the optical axis (upside-down preview, confirmed
    // by user testing). The Python reference at vr180_gui.py:1397 does
    // the same flip explicitly: `v = cy - r * sin(phi)`.
    // fy is applied as a Y-axis scale so anamorphic / non-square
    // calibrations still work.
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

    // Sample. Bilinear via the sampler binding. Out-of-frame coords
    // are clamped by the ClampToEdge sampler (matches Python MLX
    // kernel's `metal::clamp(u0, 0, fish_w - 1)` at lines 1447-1450).
    let uv = vec2<f32>(src_x / cal.src_w, src_y / cal.src_h);
    let color = textureSampleLevel(fisheye_tex, fisheye_smp, uv, 0.0);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
