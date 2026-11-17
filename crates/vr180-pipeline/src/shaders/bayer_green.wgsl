// Bayer demosaic, second pass: green at every site, white-balanced.
//
// A green site keeps its sample. At a red or blue site green is estimated
// along the row and along the column — the two green neighbours' mean plus
// an eighth of the site's own second difference (its colour bends as green
// does) — and the two estimates are mixed by how much the mosaic changes
// along each direction around the site (bayer_measure.wgsl, summed over
// 5×5): with A along the rows and B along the columns, the row's share is
// B² / (A² + B²). An edge is so interpolated along itself, and flat ground
// from all four neighbours.
//
// Bindings:
//   (0) src     — R16Uint mosaic
//   (1) out_tex — R32Float storage: green
//   (2) p       — Params (as bayer_demosaic.wgsl)
//   (3) measure — Rg32Float from the first pass

@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

struct Params {
    black: f32,
    scale: f32,
    red_x: u32,
    red_y: u32,
    src_w: i32,
    src_h: i32,
    half: u32,
    _pad: u32,
    balance: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var measure: texture_2d<f32>;

fn mirror(i: i32, n: i32) -> i32 {
    var k = i;
    if (k < 0) { k = -k; }
    if (k >= n) { k = 2 * (n - 1) - k; }
    return clamp(k, 0, n - 1);
}

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
    let red_col = (u32(x) & 1u) == p.red_x;
    let red_row = (u32(y) & 1u) == p.red_y;

    var sum = vec2<f32>(0.0);
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            sum = sum + textureLoad(measure, vec2<i32>(mirror(x + dx, p.src_w), mirror(y + dy, p.src_h)), 0).rg;
        }
    }
    let a = sum.x / 25.0 + 1e-5;
    let b = sum.y / 25.0 + 1e-5;
    let share = b * b / (a * a + b * b);
    let row = 0.5 * (bal(x - 1, y) + bal(x + 1, y)) + 0.125 * (2.0 * c - bal(x - 2, y) - bal(x + 2, y));
    let col = 0.5 * (bal(x, y - 1) + bal(x, y + 1)) + 0.125 * (2.0 * c - bal(x, y - 2) - bal(x, y + 2));
    let g = select(c, share * row + (1.0 - share) * col, red_col == red_row);
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(g, 0.0, 0.0, 0.0));
}
