// 4× downsample via 4×4 box average. First pass of the mid-detail
// pipeline (the Python version uses cv2 `INTER_AREA` resize at 1/4,
// which is also a box-filter average for integer downscale factors).
//
// Output dims are ceil(input / 4). Each output pixel samples the
// 4×4 input block at the same logical location; pixels past the
// right/bottom edge are clamped (`min(coord, max_xy)`) so partial
// blocks at the edge use whatever's available rather than zero-padding.
//
// We deliberately do not use `textureSampleLevel` with a hardware
// bilinear sampler — that would give a 2-tap bilinear average, which
// is the wrong filter (introduces aliasing on high-contrast detail).
// `textureLoad` + manual 4×4 average matches cv2's INTER_AREA.

@group(0) @binding(0) var in_tex:   texture_2d<f32>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(out_tex);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }
    let in_dims = textureDimensions(in_tex);
    let max_xy = vec2<i32>(in_dims) - vec2<i32>(1);

    let base = vec2<i32>(gid.xy) * 4;
    var acc = vec4<f32>(0.0);
    for (var dy = 0; dy < 4; dy = dy + 1) {
        for (var dx = 0; dx < 4; dx = dx + 1) {
            let sc = vec2<i32>(
                min(base.x + dx, max_xy.x),
                min(base.y + dy, max_xy.y),
            );
            acc = acc + textureLoad(in_tex, sc, 0);
        }
    }
    textureStore(out_tex, vec2<i32>(gid.xy), acc / 16.0);
}
