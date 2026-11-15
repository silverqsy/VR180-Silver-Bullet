// Lens-frame output of the fisheye-output kernels. This file is not a shader
// of its own: `gpu::lens_frame_wgsl` splices it into each of those kernels,
// in place of their equidistant-disc ray generator, to make the lens-frame
// variant. (The disc kernels themselves are left untouched — their output
// must not move by a rounding step.)
//
// The whole frame is the image of a centred Mei-Rives lens `cal.out_lens` =
// (ξ, focal in output pixels, k1, k2). A ray lands on the plane point
// m = (x, y) / (z + ξ) and reaches the frame at
// focal · m · (1 + k1·|m|² + k2·|m|⁴) from its centre; this is the inverse:
// the ray of pixel `gid` (x right, y up, z forward, unit length), with the
// size of one output pixel in radians at the frame's centre in `.w`.
// `.w < 0`: the pixel shows nothing — the last `cal.out_rim` pixels at the
// border stay black, and so does anything past the model's fold.
//
// The file that carries such a frame states the same lens
// (`spherical_inject::apmp_lens_atoms`), so a player shows every pixel at
// this ray. CPU twin: `lens_frame::LensFrame::ray`.
fn lens_frame_ray(gid: vec2<u32>, out_dim: vec2<u32>) -> vec4<f32> {
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;
    if (min(min(px, f32(out_dim.x) - px), min(py, f32(out_dim.y) - py)) < cal.out_rim) {
        return vec4<f32>(0.0, 0.0, 1.0, -1.0);
    }
    let dx = px - f32(out_dim.x) * 0.5;
    let dy = f32(out_dim.y) * 0.5 - py;
    let xi = cal.out_lens.x;
    let k1 = cal.out_lens.z;
    let k2 = cal.out_lens.w;
    let r = sqrt(dx * dx + dy * dy);
    let rd = r / cal.out_lens.y;
    // rd = m · (1 + k1 m² + k2 m⁴), solved for m by Newton from m = rd.
    var m = rd;
    for (var i = 0; i < 8; i = i + 1) {
        let m2 = m * m;
        m = m - (m * (1.0 + m2 * (k1 + k2 * m2)) - rd) / (1.0 + m2 * (3.0 * k1 + 5.0 * k2 * m2));
    }
    let mm = m * m;
    let disc = 1.0 + (1.0 - xi * xi) * mm;
    if (disc < 0.0) { return vec4<f32>(0.0, 0.0, 1.0, -1.0); }
    let g = (xi + sqrt(disc)) / (mm + 1.0);
    var ux: f32 = 0.0;
    var uy: f32 = 0.0;
    if (r > 1e-6) { ux = dx / r; uy = dy / r; }
    return vec4<f32>(g * m * ux, g * m * uy, g - xi, (1.0 + xi) / cal.out_lens.y);
}

