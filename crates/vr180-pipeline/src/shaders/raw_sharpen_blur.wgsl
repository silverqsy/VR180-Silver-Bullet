// Raw sharpening: one pass of a blur along one axis, with the taps given
// (symmetric, the centre first; the picture's edge pixels repeat past its
// border). The pass along the second axis of each of the kernel's blurs
// also adds that blur's share of the sharpening to the sum so far:
//   sum' = sum + coeff · (level − blurred).
//
// Bindings:
//   (0) src   — the plane to blur (R32Float)
//   (1) dst   — R32Float
//   (2) p     — Params
//   (3) taps  — the blur's taps from the centre outward
//   (4) level — the level plane (read when p.dir.w > 0)
//   (5) sum   — the sum so far (read when p.dir.w == 1)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

struct Params {
    // x, y = the step along the blur; z = taps to each side;
    // w = 0: the blur itself; 1: sum + coeff (level − blur); 2: coeff (level − blur)
    dir: vec4<i32>,
    // x = coeff
    coeff: vec4<f32>,
}
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> taps: array<f32>;
@group(0) @binding(4) var level: texture_2d<f32>;
@group(0) @binding(5) var sum: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(dst));
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    if (at.x >= size.x || at.y >= size.y) { return; }
    let step = p.dir.xy;
    let last = size - vec2<i32>(1);
    var acc = taps[0] * textureLoad(src, at, 0).r;
    for (var i: i32 = 1; i <= p.dir.z; i = i + 1) {
        let a = clamp(at - step * i, vec2<i32>(0), last);
        let b = clamp(at + step * i, vec2<i32>(0), last);
        acc = acc + taps[i] * (textureLoad(src, a, 0).r + textureLoad(src, b, 0).r);
    }
    let share = p.coeff.x * (textureLoad(level, at, 0).r - acc);
    let so_far = textureLoad(sum, at, 0).r;
    let out = select(select(share, so_far + share, p.dir.w == 1), acc, p.dir.w == 0);
    textureStore(dst, at, vec4<f32>(out, 0.0, 0.0, 1.0));
}
