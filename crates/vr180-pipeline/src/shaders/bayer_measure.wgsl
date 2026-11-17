// Bayer demosaic, first pass: how much the mosaic changes along its rows
// and along its columns, at every site.
//
// The mosaic is read white-balanced (see bayer_demosaic.wgsl). Along a
// direction, a site's two neighbours are of one colour and so are the site
// and the two samples two away; the measure is
//
//   |a − b|  +  ½ |2 c − c₋₂ − c₊₂|
//
// (a, b the neighbours, c the site): the slope across the site and half its
// own bend. The green pass sums it over a window to decide how much each
// direction is to be trusted.
//
// Bindings:
//   (0) src     — R16Uint mosaic
//   (1) out_tex — Rg32Float storage: r = along the row, g = along the column
//   (2) p       — Params (as bayer_demosaic.wgsl)

@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rg32float, write>;

struct Params {
    black: f32,
    scale: f32,
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = bal(x, y);
    let row = abs(bal(x - 1, y) - bal(x + 1, y)) + 0.5 * abs(2.0 * c - bal(x - 2, y) - bal(x + 2, y));
    let col = abs(bal(x, y - 1) - bal(x, y + 1)) + 0.5 * abs(2.0 * c - bal(x, y - 2) - bal(x, y + 2));
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(row, col, 0.0, 0.0));
}
