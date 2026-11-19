// The means of a picture's Shadows / Highlights gain (`raw_gain.wgsl`,
// `tone_gain.wgsl`) over cells of c × c pixels: what a pixel's gain is
// smoothed toward (`raw_local::gain_cells`; the last row / column of cells
// repeats the edge pixels).
//
// Bindings:
//   (0) src — the gain, R32Float
//   (1) dst — Rgba16Float storage, ceil(size / c): the mean in x
//   (2) p   — Params

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

struct Params {
    // x = pixels per cell (c)
    cell: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    let c = i32(p.cell.x);
    let origin = vec2<i32>(i32(gid.x), i32(gid.y)) * c;
    var sum = 0.0;
    for (var j: i32 = 0; j < c; j = j + 1) {
        for (var i: i32 = 0; i < c; i = i + 1) {
            sum = sum + textureLoad(src, min(origin + vec2<i32>(i, j), last), 0).r;
        }
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sum / f32(c * c), 0.0, 0.0, 1.0));
}
