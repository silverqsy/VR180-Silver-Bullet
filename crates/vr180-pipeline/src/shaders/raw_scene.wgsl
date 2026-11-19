// The one `scene` of the raw shaders: composed into each of them at its
// `use SCENE` line (`raw_gpu::with_scene`). It reads `p.neutral`,
// `p.cam_to_work_*` and `p.clip` of the shader it lands in. The CPU twin is
// `raw_tone::DevelopGeometry::scene`; the rule and where it comes from are
// described there.

//#define SCENE
// Scene RGB (linear ProPhoto) of a camera pixel, its clipped channels
// rebuilt. White-balanced, a channel is trusted less the nearer it is to
// its clip point: weight ((c - from) / (to - from))², c its camera value.
// The channel of the white that clips first rises toward the mean of the
// other two by its weight and never falls; of those two the lower one rises
// toward the higher by its own.
fn scene(camera: vec3<f32>) -> vec3<f32> {
    let n = p.neutral.xyz;
    let c = max(camera, vec3<f32>(0.0));
    let v = c / n;
    let t = clamp((c - vec3<f32>(p.clip.x)) / (p.clip.y - p.clip.x), vec3<f32>(0.0), vec3<f32>(1.0));
    let w = t * t;
    // The channel that clips first is the largest of the white's camera
    // values: green before red before blue where two are equal.
    let top = max(n.r, max(n.g, n.b));
    let fg = step(top, n.g);
    let fr = (1.0 - fg) * step(top, n.r);
    let first = vec3<f32>(fr, fg, (1.0 - fg) * (1.0 - fr));
    // The higher of the other two (no value is below zero, so -1 stands for
    // the one left out).
    let others = mix(v, vec3<f32>(-1.0), first);
    let hi = max(others.r, max(others.g, others.b));
    let later = v + w * (vec3<f32>(hi) - v);
    let mean = dot(later, vec3<f32>(1.0) - first) * 0.5;
    let own = dot(v, first);
    let risen = own + dot(w, first) * max(mean - own, 0.0);
    let rebuilt = mix(later, vec3<f32>(risen), first) * n;
    return vec3<f32>(
        dot(p.cam_to_work_0.xyz, rebuilt),
        dot(p.cam_to_work_1.xyz, rebuilt),
        dot(p.cam_to_work_2.xyz, rebuilt));
}
//#enddef
