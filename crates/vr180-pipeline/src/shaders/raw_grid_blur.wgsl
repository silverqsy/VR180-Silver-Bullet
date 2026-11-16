// Smooths a raw frame's bilateral grid (`raw_grid_build.wgsl`) [1 2 1] along
// its three axes (edges repeated) and adds, in z, the plain local average of
// the column (Σ over levels of the sums, as an offset from the first level):
// what a pixel unlike everything near it falls back on.
//
// Bindings:
//   (0) in_grid  — the gathered grid
//   (1) out_grid — Rgba16Float storage, 3D

const NZ: i32 = 18;

@group(0) @binding(0) var in_grid: texture_3d<f32>;

@group(0) @binding(1) var out_grid: texture_storage_3d<rgba16float, write>;

fn at(x: i32, y: i32, z: i32, dims: vec3<i32>) -> vec2<f32> {
    let c = clamp(vec3<i32>(x, y, z), vec3<i32>(0), dims - vec3<i32>(1));
    return textureLoad(in_grid, c, 0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<i32>(textureDimensions(out_grid));
    if (i32(gid.x) >= dims.x || i32(gid.y) >= dims.y) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var col: array<vec2<f32>, 18>;
    var total = vec2<f32>(0.0);
    for (var z: i32 = 0; z < NZ; z = z + 1) {
        var sum = vec2<f32>(0.0);
        for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
            let wz = 2.0 - abs(f32(dz));
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                let wy = 2.0 - abs(f32(dy));
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    let wx = 2.0 - abs(f32(dx));
                    sum = sum + wx * wy * wz * at(x + dx, y + dy, z + dz, dims);
                }
            }
        }
        col[z] = sum / 64.0;
        total = total + col[z];
    }
    // The column's plain average, as an offset from the first level.
    let mean = total.x / max(total.y, 1e-9);
    for (var z: i32 = 0; z < NZ; z = z + 1) {
        textureStore(out_grid, vec3<i32>(x, y, z), vec4<f32>(col[z], mean, 0.0));
    }
}
