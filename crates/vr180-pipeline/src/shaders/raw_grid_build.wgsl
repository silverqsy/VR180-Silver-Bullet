// The bilateral grid of a raw frame's reduced copy (levels = log2 scene
// luminance), gathered: the Shadows / Highlights base is read from it once
// `raw_grid_blur.wgsl` has smoothed it.
//
// A vertex (i, j, k) stands for the place ((i − ½)·cell, (j − ½)·cell) of the
// reduced copy and the level lo + (k − ½)·range. Every vertex gathers the
// texels around it, each weighted by its closeness in place and in level:
// xy = (Σ w·(level − lo), Σ w).
//
// Bindings:
//   (0) red      — the reduced copy (R32Float)
//   (1) out_grid — Rgba16Float storage, 3D
//   (2) p        — Params

const NZ: i32 = 18;

struct Params {
    // x = cell in texels of the reduced copy, y = first level, z = 1 / range
    cell: vec4<f32>,
}

@group(0) @binding(0) var red: texture_2d<f32>;
@group(0) @binding(1) var out_grid: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_grid);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let size = vec2<i32>(textureDimensions(red));
    let cs = p.cell.x;
    let vx = f32(gid.x);
    let vy = f32(gid.y);
    // Texels whose place is within one cell of the vertex.
    let x0 = max(i32(ceil((vx - 1.5) * cs - 0.5)), 0);
    let x1 = min(i32(floor((vx + 0.5) * cs - 0.5)), size.x - 1);
    let y0 = max(i32(ceil((vy - 1.5) * cs - 0.5)), 0);
    let y1 = min(i32(floor((vy + 0.5) * cs - 0.5)), size.y - 1);
    var acc: array<vec2<f32>, 18>;
    for (var k: i32 = 0; k < NZ; k = k + 1) { acc[k] = vec2<f32>(0.0); }
    for (var y: i32 = y0; y <= y1; y = y + 1) {
        let wy = max(1.0 - abs((f32(y) + 0.5) / cs + 0.5 - vy), 0.0);
        for (var x: i32 = x0; x <= x1; x = x + 1) {
            let wx = max(1.0 - abs((f32(x) + 0.5) / cs + 0.5 - vx), 0.0);
            let level = textureLoad(red, vec2<i32>(x, y), 0).r;
            let fz = (level - p.cell.y) * p.cell.z + 0.5;
            let iz = clamp(i32(floor(fz)), 0, NZ - 2);
            let tz = clamp(fz - f32(iz), 0.0, 1.0);
            let w = wx * wy;
            let v = vec2<f32>(level - p.cell.y, 1.0);
            acc[iz] = acc[iz] + w * (1.0 - tz) * v;
            acc[iz + 1] = acc[iz + 1] + w * tz * v;
        }
    }
    for (var k: i32 = 0; k < NZ; k = k + 1) {
        textureStore(out_grid, vec3<i32>(i32(gid.x), i32(gid.y), k), vec4<f32>(acc[k], 0.0, 0.0));
    }
}
