// EAC → half-equirect projection (one lens, one frame).
//
// Inputs:
//   binding(0): the EAC cross texture (cross_w × cross_w, rgba8unorm)
//   binding(1): bilinear sampler
//   binding(2): storage texture for the equirect output (rgba8unorm)
//   binding(3): EquirectUniforms — per-frame R matrix for stabilization
//
// One thread per output pixel. The math mirrors the Python
// `FrameExtractor.build_cross_remap`:
//
//   * convert (u, v) ∈ [0,1]² to (lon, lat) ∈ [-π/2, π/2]²
//   * recover unit direction vector (xn, yn, zn)
//   * **apply per-frame rotation R** (Phase A stabilization).
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

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;
const TWO_OVER_PI: f32 = 0.63661977236;  // 2/π

// Cross-coordinate constants. All face boundaries derive from these:
//   * `tile_w = cross_dim / 3.90476…` is not stable across camera variants,
//   * but `center_w = 1920` is fixed, so we compute tile_w on-shader:
//        tile_w = (cross_dim - center_w) / 2  // because cross_dim = 2 * tile_w + center_w
const CENTER_W: f32 = 1920.0;

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

    // Per-pixel normalized coords + (lon, lat).
    let u = (f32(gid.x) + 0.5) / f32(out_dim.x);
    let v = (f32(gid.y) + 0.5) / f32(out_dim.y);
    let lon = (u - 0.5) * PI;
    let lat = (0.5 - v) * PI;

    // Unit direction in equirect (output) frame, then rotated into
    // camera (input) frame by the per-frame R uniform. When
    // stabilization is off R = identity and these two frames are the
    // same.
    let cos_lat = cos(lat);
    let dir_world = vec3<f32>(
        cos_lat * sin(lon),
        sin(lat),
        cos_lat * cos(lon),
    );
    // R · dir — manual row-vector dot products. 12-scalar uniform
    // layout (see EquirectUniforms above); each row is 3 floats +
    // 1 pad.
    let xn = equ.r00 * dir_world.x + equ.r01 * dir_world.y + equ.r02 * dir_world.z;
    let yn = equ.r10 * dir_world.x + equ.r11 * dir_world.y + equ.r12 * dir_world.z;
    let zn = equ.r20 * dir_world.x + equ.r21 * dir_world.y + equ.r22 * dir_world.z;

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

    // Sentinel: pixel falls outside any face → write black.
    var hit = false;
    var sx: f32 = 0.0;  // pixel x in cross (absolute, before 0.5 sample offset)
    var sy: f32 = 0.0;

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
        // Python: full_col = u_eac * 1920; valid iff full_col in [0, tile_w)
        let full_col = u_eac * CENTER_W;
        if (full_col >= 0.0 && full_col < tw) {
            sx = center_bot + full_col;
            sy = center_top + v_eac * CENTER_W;
            hit = true;
        }
    }
    // ── Left face ── (x < 0, z ≤ |x|, |y| ≤ |x|) ─────────────────────
    else if (xn < 0.0 && zn <= ax && ay <= ax) {
        let u_eac = TWO_OVER_PI * atan(zn / ax) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(yn / ax);
        // Python: partial_col = u_eac * 1920 - 912; valid iff in [0, tile_w)
        //   On Max: 912 = center_w - tile_w = 1920 - 1008 = 912 ✓
        //   General: 912 = CENTER_W - tile_w
        let partial_col = u_eac * CENTER_W - (CENTER_W - tw);
        if (partial_col >= 0.0 && partial_col < tw) {
            sx = partial_col;
            sy = center_top + v_eac * CENTER_W;
            hit = true;
        }
    }
    // ── Top face ── (y > 0, |x| ≤ y, z ≤ y) ──────────────────────────
    else if (yn > 0.0 && ax <= yn && zn <= yn) {
        let u_eac = TWO_OVER_PI * atan(xn / yn) + 0.5;
        let v_eac = TWO_OVER_PI * atan(zn / yn) + 0.5;
        let partial_row = v_eac * CENTER_W - (CENTER_W - tw);
        if (partial_row >= 0.0 && partial_row < tw) {
            sx = center_top + u_eac * CENTER_W;
            sy = partial_row;
            hit = true;
        }
    }
    // ── Bottom face ── (y < 0, |x| ≤ |y|, z ≤ |y|) ───────────────────
    else if (yn < 0.0 && ax <= ay && zn <= ay) {
        let u_eac = TWO_OVER_PI * atan(xn / ay) + 0.5;
        let v_eac = 0.5 - TWO_OVER_PI * atan(zn / ay);
        let full_row = v_eac * CENTER_W;
        if (full_row >= 0.0 && full_row < tw) {
            sx = center_top + u_eac * CENTER_W;
            sy = center_bot + full_row;
            hit = true;
        }
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
