// Bayer mosaic (sensor codes, R16Uint) → camera-linear RGBA16.
//
// Each sample is linearised as `(code − black) × scale` (`scale` =
// 1 / (white − black), so the sensor's white level is 1.0) and the three
// colours of every output pixel are interpolated from the mosaic. The
// result is the camera's own RGB: not white balanced, no colour matrix.
//
// Two kernels, picked by `half`:
//   0 — full size, Malvar-He-Cutler 5×5 (the kernels of `raw_develop::mhc`,
//       borders reflected so the colour of a site is kept);
//   1 — half size, one pixel per 2×2 site group (R, the mean of the two
//       greens, B): a preview's source.
//
// Bindings:
//   (0) src     — R16Uint mosaic
//   (1) out_tex — Rgba16Unorm storage (the eye, or its half-size copy)
//   (2) p       — Params

@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16unorm, write>;

struct Params {
    black: f32,
    scale: f32,
    // Parity (x, y) of the red site; blue is the opposite corner.
    red_x: u32,
    red_y: u32,
    src_w: i32,
    src_h: i32,
    half: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> p: Params;

fn mirror(i: i32, n: i32) -> i32 {
    var k = i;
    if (k < 0) { k = -k; }
    if (k >= n) { k = 2 * (n - 1) - k; }
    return clamp(k, 0, n - 1);
}

fn lin(x: i32, y: i32) -> f32 {
    let v = textureLoad(src, vec2<i32>(mirror(x, p.src_w), mirror(y, p.src_h)), 0).r;
    return (f32(v) - p.black) * p.scale;
}

fn full(x: i32, y: i32) -> vec3<f32> {
    let c = lin(x, y);
    let n = lin(x, y - 1);
    let s = lin(x, y + 1);
    let w = lin(x - 1, y);
    let e = lin(x + 1, y);
    let far_v = lin(x, y - 2) + lin(x, y + 2);
    let far_h = lin(x - 2, y) + lin(x + 2, y);
    let diag = lin(x - 1, y - 1) + lin(x + 1, y - 1) + lin(x - 1, y + 1) + lin(x + 1, y + 1);
    let red_col = (u32(x) & 1u) == p.red_x;
    let red_row = (u32(y) & 1u) == p.red_y;
    if (red_col == red_row) {
        // A red or a blue site.
        let g = (4.0 * c + 2.0 * (n + s + w + e) - (far_v + far_h)) * 0.125;
        let other = (6.0 * c + 2.0 * diag - 1.5 * (far_v + far_h)) * 0.125;
        if (red_row) { return vec3<f32>(c, g, other); }
        return vec3<f32>(other, g, c);
    }
    // A green site: one colour beside it, the other above and below.
    let beside = (5.0 * c + 4.0 * (w + e) - far_h - diag + 0.5 * far_v) * 0.125;
    let above = (5.0 * c + 4.0 * (n + s) - far_v - diag + 0.5 * far_h) * 0.125;
    if (red_row) { return vec3<f32>(beside, c, above); }
    return vec3<f32>(above, c, beside);
}

fn binned(x: i32, y: i32) -> vec3<f32> {
    let x0 = 2 * x;
    let y0 = 2 * y;
    let rx = i32(p.red_x);
    let ry = i32(p.red_y);
    let r = lin(x0 + rx, y0 + ry);
    let b = lin(x0 + 1 - rx, y0 + 1 - ry);
    let g = 0.5 * (lin(x0 + 1 - rx, y0 + ry) + lin(x0 + rx, y0 + 1 - ry));
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var rgb: vec3<f32>;
    if (p.half == 1u) { rgb = binned(x, y); } else { rgb = full(x, y); }
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
