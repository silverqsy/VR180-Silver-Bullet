// Bayer mosaic (sensor codes, R16Uint) → camera-linear RGBA16, last pass.
//
// Each sample is linearised as `(code − black) × scale` (`scale` =
// 1 / (white − black), so the sensor's white level is 1.0). The result is
// the camera's own RGB: not white balanced, no colour matrix.
//
// The interpolation works on the mosaic WHITE-BALANCED (red and blue
// samples divided by the camera's neutral), where a grey edge has no
// colour difference whatever its brightness: the form Adobe Camera Raw's
// own demosaic was found to have (measured on its renders of the same
// mosaics; see CLAUDE.md). Three passes:
//
//   bayer_measure — how much the mosaic changes along rows and columns;
//   bayer_green   — green at every site, interpolated along the direction
//                   that changes less;
//   this one      — red and blue. At its own sites a colour is its
//                   sample; elsewhere it is green plus the colour's
//                   difference from green, carried over bilinearly from
//                   the nearest sites of that colour (two beside or above
//                   and below a green site, four around the opposite
//                   colour's site).
//
// `half` picks the output: 0 — the mosaic's own size; 1 — half size, each
// pixel the mean of the 2×2 full-size pixels it covers (a preview's
// source).
//
// Bindings:
//   (0) src     — R16Uint mosaic
//   (1) out_tex — Rgba16Unorm storage (the eye, or its half-size copy)
//   (2) p       — Params
//   (3) green   — R32Float from the second pass

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
    // x = what a red sample is multiplied by, y = a blue one (1 / the
    // camera's neutral); z, w = the neutral's red and blue themselves.
    balance: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var green: texture_2d<f32>;

fn mirror(i: i32, n: i32) -> i32 {
    var k = i;
    if (k < 0) { k = -k; }
    if (k >= n) { k = 2 * (n - 1) - k; }
    return clamp(k, 0, n - 1);
}

// The white-balanced sample at (x, y); past the border the mosaic is
// mirrored, which keeps a site's colour.
fn bal(x: i32, y: i32) -> f32 {
    let v = textureLoad(src, vec2<i32>(mirror(x, p.src_w), mirror(y, p.src_h)), 0).r;
    let lin = (f32(v) - p.black) * p.scale;
    let red_col = (u32(x) & 1u) == p.red_x;
    let red_row = (u32(y) & 1u) == p.red_y;
    let gain = select(1.0, select(p.balance.y, p.balance.x, red_row), red_col == red_row);
    return lin * gain;
}

fn grn(x: i32, y: i32) -> f32 {
    return textureLoad(green, vec2<i32>(mirror(x, p.src_w), mirror(y, p.src_h)), 0).r;
}

// A red or blue site's difference from green.
fn dif(x: i32, y: i32) -> f32 {
    return bal(x, y) - grn(x, y);
}

fn full(x: i32, y: i32) -> vec3<f32> {
    let g = grn(x, y);
    let red_col = (u32(x) & 1u) == p.red_x;
    let red_row = (u32(y) & 1u) == p.red_y;
    let diag = 0.25 * (dif(x - 1, y - 1) + dif(x + 1, y - 1) + dif(x - 1, y + 1) + dif(x + 1, y + 1));
    let beside = 0.5 * (dif(x - 1, y) + dif(x + 1, y));
    let above = 0.5 * (dif(x, y - 1) + dif(x, y + 1));
    let own = bal(x, y);
    let site = red_col == red_row;
    // Red: its own sample at a red site; around a blue site its four
    // diagonal neighbours; at a green site the two in its row (a red row)
    // or its column.
    let r = select(g + select(above, beside, red_row), select(g + diag, own, red_row), site);
    let b = select(g + select(beside, above, red_row), select(own, g + diag, red_row), site);
    return clamp(vec3<f32>(r * p.balance.z, g, b * p.balance.w), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var rgb: vec3<f32>;
    if (p.half == 1u) {
        rgb = 0.25 * (full(2 * x, 2 * y) + full(2 * x + 1, 2 * y) + full(2 * x, 2 * y + 1) + full(2 * x + 1, 2 * y + 1));
    } else {
        rgb = full(x, y);
    }
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(rgb, 1.0));
}
