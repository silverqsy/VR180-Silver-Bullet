//! The preview as Apple's players will show the export.
//!
//! The app's exports are tagged Rec.709, and on a Mac everything that plays
//! one through the system (QuickTime Player, Safari, Photos, the headset)
//! reads the code values with Apple's Rec.709 video curve before it
//! colour-matches the light to the display. The preview window is
//! colour-matched too, but as sRGB: that is the colour space the system
//! gives a wgpu surface's Metal layer when it is configured, and the picture
//! is drawn into it as it is. Same primaries, another curve: the same frame
//! is about 10 of 255 darker in the mid-tones than in QuickTime.
//!
//! With the match on, the preview picture is drawn by [`PreviewPainter`]
//! instead of egui's image widget, and a video frame's code values are
//! converted from Apple's Rec.709 video curve to sRGB on the way
//! ([`PictureEncoding`]). Rec.709 and sRGB share their primaries and white
//! point, so the curve is the whole conversion and the result is what the
//! system produces from the tagged file. The curve itself is ColorSync's,
//! read from the system: see [`rec709_video_to_srgb`].
//!
//! That rests on the window being sRGB to the system, so while the match is
//! on [`set_window_colour_matching`] makes sure of it. Where the layer
//! already is (macOS 27: every surface, from its first `configure`), it
//! writes nothing, ever.
//!
//! Nothing here touches a rendered frame, an export or a snapshot. It is the
//! last step in front of the screen, and it is off unless the user turns it
//! on. macOS only: [`available`] is false elsewhere and no code path below
//! runs.

/// What a picture's code values stand for, which decides how they are
/// turned into the window's (sRGB) code values when the match is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureEncoding {
    /// A frame of a video clip: the app exports these tagged Rec.709, and
    /// Apple's players read that with their Rec.709 video curve.
    Rec709Video,
    /// A developed photograph: sRGB already, shown as it is.
    Srgb,
}

/// Apple's Rec.709 video curve (`kCGColorSpaceCoreMedia709`), as ColorSync
/// states it for a GPU: a power of 1.9609375 (the profile's `curv` tag,
/// 502 / 256) with a straight toe of slope 1 / 16 below 0.05583828, where
/// the two meet. Not the Rec.709 camera curve turned around, and not 2.4.
pub const REC709_VIDEO_GAMMA: f64 = 1.960_937_5;
pub const REC709_VIDEO_TOE_SLOPE: f64 = 0.0625;
pub const REC709_VIDEO_TOE_END: f64 = 0.055_838_28;

/// One channel of a Rec.709 video frame, as the sRGB code value that shows
/// the same light: the system's own conversion from `kCGColorSpaceCoreMedia709`
/// to sRGB (ColorSync's parametric stages are this curve and the sRGB
/// encoding, with no matrix between them).
pub fn rec709_video_to_srgb(v: f64) -> f64 {
    let light = if v < REC709_VIDEO_TOE_END {
        v * REC709_VIDEO_TOE_SLOPE
    } else {
        v.powf(REC709_VIDEO_GAMMA)
    };
    if light < 0.003_130_805 {
        12.92 * light
    } else {
        1.055 * light.powf(1.0 / 2.4) - 0.055
    }
}

/// Whether this build can match the preview to Apple playback at all.
pub fn available() -> bool {
    cfg!(target_os = "macos")
}

/// With `on`, make sure the Metal layer of every window of the app is sRGB
/// to the system, which then colour-matches what is drawn in it to the
/// display it is on. A layer that carries anything else (or nothing) is
/// tagged sRGB and what it carried is remembered; with `on` false exactly
/// those layers get it back. A layer that is sRGB already is never written.
/// Returns how many layers it wrote. Cheap enough to call every frame. Main
/// thread only; a no-op off macOS.
pub fn set_window_colour_matching(on: bool) -> usize {
    platform::set_window_colour_matching(on)
}

/// Which of the preview's draws a paint is: each has its own small uniform
/// buffer, so the draws of one frame do not overwrite each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintSlot {
    /// The picture in the main window.
    Main = 0,
    /// The left and right halves on the 3D display.
    ThreeDLeft = 1,
    ThreeDRight = 2,
}

const SLOT_COUNT: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawUniforms {
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    /// 1 = the texture holds Rec.709 video code values, 0 = sRGB.
    rec709_video: u32,
    /// 1 = the target stores linear light (an `*Srgb` surface format).
    linear_target: u32,
    _pad: [u32; 2],
}

const SHADER: &str = r#"
struct Draw {
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    rec709_video: u32,
    linear_target: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var picture_sampler: sampler;
@group(1) @binding(0) var<uniform> draw: Draw;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// A strip over the whole viewport, which the caller sets to the picture's
// rectangle: corner 0 is the top left.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    var out: VertexOutput;
    out.position = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = mix(draw.uv_min, draw.uv_max, corner);
    return out;
}

// Apple's Rec.709 video curve to light, then light to sRGB code values.
fn srgb_from_rec709_video(v: vec3<f32>) -> vec3<f32> {
    let light = select(pow(v, vec3<f32>(1.9609375)), v * 0.0625, v < vec3<f32>(0.05583828));
    return select(
        1.055 * pow(light, vec3<f32>(1.0 / 2.4)) - 0.055,
        light * 12.92,
        light < vec3<f32>(0.003130805));
}

fn linear_from_srgb(v: vec3<f32>) -> vec3<f32> {
    return select(
        pow((v + 0.055) / 1.055, vec3<f32>(2.4)),
        v / 12.92,
        v < vec3<f32>(0.04045));
}

// egui's own dither (interleaved gradient noise, just under half a code),
// so a value between two codes is spread over them instead of banding.
fn dither(rgb: vec3<f32>, position: vec4<f32>) -> vec3<f32> {
    let f = 0.06711056 * position.x + 0.00583715 * position.y;
    let noise = (fract(52.9829189 * fract(f)) - 0.5) * 0.95;
    return rgb + noise / 255.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(picture, picture_sampler, in.uv);
    var rgb = texel.rgb;
    if draw.rec709_video == 1u {
        rgb = srgb_from_rec709_video(rgb);
    }
    rgb = dither(rgb, in.position);
    if draw.linear_target == 1u {
        rgb = linear_from_srgb(max(rgb, vec3<f32>(0.0)));
    }
    return vec4<f32>(rgb, texel.a);
}
"#;

/// Draws the preview picture into egui's pass with its own shader: the
/// place where a video frame's code values become the window's. Lives in
/// the renderer's callback resources; the app keeps a clone to make the
/// per-texture bind groups.
#[derive(Clone)]
pub struct PreviewPainter {
    pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    slots: [(wgpu::Buffer, wgpu::BindGroup); SLOT_COUNT],
    linear_target: bool,
}

impl PreviewPainter {
    /// `target_format` is the format of the pass egui draws in (the window
    /// surface's). The pass is egui's: one sample, no depth.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview painter"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview painter picture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview painter draw"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview painter"),
            bind_group_layouts: &[Some(&texture_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview painter"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // egui's own blend (premultiplied alpha), so a texel
                    // that is not opaque lands as it does in its image.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        // egui's sampler for a picture registered with `FilterMode::Linear`.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preview painter"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let slots = std::array::from_fn(|i| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("preview painter draw {i}")),
                size: std::mem::size_of::<DrawUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("preview painter draw"),
                layout: &draw_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            });
            (buffer, group)
        });
        Self { pipeline, texture_layout, sampler, slots, linear_target: target_format.is_srgb() }
    }

    /// The bind group that hands one picture to the painter: made once per
    /// displayed texture, kept with it.
    pub fn bind_picture(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview painter picture"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }
}

/// One draw of the preview picture: the part `uv` of it into the rectangle
/// the paint callback is given, over the pixels egui's image widget would
/// cover.
pub struct PreviewPaint {
    pub picture: wgpu::BindGroup,
    pub slot: PaintSlot,
    pub uv: egui::Rect,
    pub encoding: PictureEncoding,
}

impl PreviewPaint {
    /// The shape to add to a painter: `rect` in points, as for an image.
    pub fn shape(self, rect: egui::Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(rect, self))
    }
}

impl egui_wgpu::CallbackTrait for PreviewPaint {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(painter) = resources.get::<PreviewPainter>() {
            let uniforms = DrawUniforms {
                uv_min: [self.uv.min.x, self.uv.min.y],
                uv_max: [self.uv.max.x, self.uv.max.y],
                rec709_video: u32::from(self.encoding == PictureEncoding::Rec709Video),
                linear_target: u32::from(painter.linear_target),
                _pad: [0; 2],
            };
            queue.write_buffer(&painter.slots[self.slot as usize].0, 0, bytemuck::bytes_of(&uniforms));
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(painter) = resources.get::<PreviewPainter>() else { return; };
        // egui's image widget snaps its rectangle to whole pixels (each edge
        // to the nearest one) and stretches the picture over that; the same
        // rectangle here. The viewport egui set beforehand is that one too,
        // but cut off at the screen's edge, which would squeeze the picture
        // where the widget only clips it.
        let scale = info.pixels_per_point;
        let rect = info.viewport;
        let (left, top) = ((rect.min.x * scale).round(), (rect.min.y * scale).round());
        let (w, h) = ((rect.max.x * scale).round() - left, (rect.max.y * scale).round() - top);
        if !(w > 0.0 && h > 0.0) { return; }
        render_pass.set_viewport(left, top, w, h, 0.0, 1.0);
        render_pass.set_pipeline(&painter.pipeline);
        render_pass.set_bind_group(0, &self.picture, &[]);
        render_pass.set_bind_group(1, &painter.slots[self.slot as usize].1, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use objc2::encode::{Encoding, RefEncode};
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use std::cell::RefCell;
    use std::ffi::c_void;

    /// `CGColorSpace`, opaque: the pointee of `CGColorSpaceRef`, with the
    /// type encoding the Objective-C runtime reports for it.
    #[repr(C)]
    pub(super) struct CGColorSpace {
        _private: [u8; 0],
    }
    unsafe impl RefEncode for CGColorSpace {
        const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("CGColorSpace", &[]));
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        static kCGColorSpaceSRGB: *const c_void;
        fn CGColorSpaceCreateWithName(name: *const c_void) -> *mut CGColorSpace;
        fn CGColorSpaceRetain(space: *mut CGColorSpace) -> *mut CGColorSpace;
        fn CGColorSpaceRelease(space: *mut CGColorSpace);
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFEqual(a: *const c_void, b: *const c_void) -> u8;
    }

    /// A layer this module tagged sRGB, with what it carried before
    /// (retained; null for nothing).
    pub(super) struct Retagged {
        layer: usize,
        before: *mut CGColorSpace,
    }

    thread_local! {
        /// The layers tagged while the match is on. Main thread only.
        static RETAGGED: RefCell<Vec<Retagged>> = const { RefCell::new(Vec::new()) };
    }

    pub fn set_window_colour_matching(on: bool) -> usize {
        // AppKit and its layer tree belong to the main thread.
        if objc2::MainThreadMarker::new().is_none() { return 0; }
        RETAGGED.with(|retagged| {
            let mut retagged = retagged.borrow_mut();
            // Off with nothing to put back: the usual case, and no reason
            // to walk the windows.
            if !on && retagged.is_empty() { return 0; }
            unsafe {
                let layers = window_metal_layers();
                apply(&mut retagged, &layers, on)
            }
        })
    }

    /// The Metal layers of every window of the app.
    unsafe fn window_metal_layers() -> Vec<*mut AnyObject> {
        let mut layers = Vec::new();
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        if app.is_null() { return layers; }
        let windows: *mut AnyObject = msg_send![app, windows];
        if windows.is_null() { return layers; }
        let count: usize = msg_send![windows, count];
        for i in 0..count {
            let window: *mut AnyObject = msg_send![windows, objectAtIndex: i];
            if window.is_null() { continue; }
            let view: *mut AnyObject = msg_send![window, contentView];
            if view.is_null() { continue; }
            let layer: *mut AnyObject = msg_send![view, layer];
            collect_metal_layers(layer, 0, &mut layers);
        }
        layers
    }

    /// Every `CAMetalLayer` at or just below `layer`. wgpu draws into one it
    /// adds as a sublayer of the view's own layer (or into the view's layer
    /// when that already is a Metal layer).
    pub(super) unsafe fn collect_metal_layers(layer: *mut AnyObject, depth: u32, out: &mut Vec<*mut AnyObject>) {
        if layer.is_null() || depth > 2 { return; }
        let is_metal: bool = msg_send![layer, isKindOfClass: class!(CAMetalLayer)];
        if is_metal {
            out.push(layer);
            return;
        }
        let sublayers: *mut AnyObject = msg_send![layer, sublayers];
        if sublayers.is_null() { return; }
        let count: usize = msg_send![sublayers, count];
        for i in 0..count {
            let sub: *mut AnyObject = msg_send![sublayers, objectAtIndex: i];
            collect_metal_layers(sub, depth + 1, out);
        }
    }

    /// On: every layer that is not sRGB becomes it, and what it carried is
    /// kept in `retagged`. Off: the layers in `retagged` that still exist get
    /// theirs back. Returns the number of layers written.
    pub(super) unsafe fn apply(retagged: &mut Vec<Retagged>, layers: &[*mut AnyObject], on: bool) -> usize {
        let mut written = 0;
        if on {
            // A layer that is gone needs nothing put back.
            retagged.retain(|r| {
                let alive = layers.iter().any(|l| *l as usize == r.layer);
                if !alive && !r.before.is_null() { CGColorSpaceRelease(r.before); }
                alive
            });
            let srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            if srgb.is_null() { return 0; }
            for &layer in layers {
                let current: *mut CGColorSpace = msg_send![layer, colorspace];
                if !current.is_null() && CFEqual(current.cast(), srgb.cast()) != 0 { continue; }
                if !retagged.iter().any(|r| r.layer == layer as usize) {
                    let before = if current.is_null() { current } else { CGColorSpaceRetain(current) };
                    retagged.push(Retagged { layer: layer as usize, before });
                }
                let _: () = msg_send![layer, setColorspace: srgb];
                written += 1;
            }
            CGColorSpaceRelease(srgb);
        } else {
            for r in retagged.drain(..) {
                if let Some(&layer) = layers.iter().find(|l| **l as usize == r.layer) {
                    let _: () = msg_send![layer, setColorspace: r.before];
                    written += 1;
                }
                if !r.before.is_null() { CGColorSpaceRelease(r.before); }
            }
        }
        written
    }

    #[cfg(test)]
    pub(super) mod test_support {
        use super::*;

        #[link(name = "CoreGraphics", kind = "framework")]
        extern "C" {
            static kCGColorSpaceDisplayP3: *const c_void;
            fn CGColorSpaceCopyName(space: *mut CGColorSpace) -> *const c_void;
        }
        #[link(name = "CoreFoundation", kind = "framework")]
        extern "C" {
            fn CFRelease(obj: *const c_void);
        }

        unsafe fn named(layer: *mut AnyObject, name: *const c_void) -> bool {
            let space: *mut CGColorSpace = msg_send![layer, colorspace];
            if space.is_null() { return false; }
            let have = CGColorSpaceCopyName(space);
            let equal = !have.is_null() && CFEqual(have, name) != 0;
            if !have.is_null() { CFRelease(have); }
            equal
        }

        /// Three Metal layers under one root, as three windows might hold
        /// them: one with no colour space, one that is sRGB already (what
        /// the system gives a configured surface), one tagged Display P3.
        /// On, off, and what each carried at every step.
        pub fn round_trip() {
            unsafe {
                let root: *mut AnyObject = msg_send![class!(CALayer), layer];
                let plain: *mut AnyObject = msg_send![class!(CALayer), layer];
                let bare: *mut AnyObject = msg_send![class!(CAMetalLayer), layer];
                let srgb_already: *mut AnyObject = msg_send![class!(CAMetalLayer), layer];
                let p3: *mut AnyObject = msg_send![class!(CAMetalLayer), layer];
                for l in [plain, bare, srgb_already, p3] {
                    let _: () = msg_send![root, addSublayer: l];
                }
                let srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
                let _: () = msg_send![srgb_already, setColorspace: srgb];
                CGColorSpaceRelease(srgb);
                let wide = CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3);
                let _: () = msg_send![p3, setColorspace: wide];
                CGColorSpaceRelease(wide);

                let mut layers = Vec::new();
                collect_metal_layers(root, 0, &mut layers);
                assert_eq!(layers.len(), 3, "the three Metal layers, not the plain one");

                let mut retagged = Vec::new();
                // Off before anything was on: nothing is touched.
                assert_eq!(apply(&mut retagged, &layers, false), 0);
                let none: *mut CGColorSpace = msg_send![bare, colorspace];
                assert!(none.is_null());

                // On: the two that are not sRGB are written, the third is not.
                assert_eq!(apply(&mut retagged, &layers, true), 2);
                for l in [bare, srgb_already, p3] {
                    assert!(named(l, kCGColorSpaceSRGB));
                }
                // A second pass finds them all sRGB and writes nothing.
                assert_eq!(apply(&mut retagged, &layers, true), 0);
                assert_eq!(retagged.len(), 2);

                // Off: each gets back what it had, and the one that was sRGB
                // on its own account stays sRGB.
                assert_eq!(apply(&mut retagged, &layers, false), 2);
                let none: *mut CGColorSpace = msg_send![bare, colorspace];
                assert!(none.is_null(), "the bare layer carries nothing again");
                assert!(named(p3, kCGColorSpaceDisplayP3), "the P3 layer is P3 again");
                assert!(named(srgb_already, kCGColorSpaceSRGB), "never written, still sRGB");
                assert!(retagged.is_empty());
                assert_eq!(apply(&mut retagged, &layers, false), 0);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    pub fn set_window_colour_matching(_on: bool) -> usize { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 8-bit sRGB codes the system itself produces from a grey ramp
    /// tagged `kCGColorSpaceCoreMedia709` (CoreGraphics, macOS 27.2, drawn
    /// into a 16-bit sRGB context): every 16th code and the toe.
    const SYSTEM_RAMP: [(u8, f64); 28] = [
        (0, 0.000), (1, 0.809), (2, 1.619), (3, 2.428), (4, 3.237), (5, 4.047),
        (6, 4.856), (7, 5.603), (8, 6.412), (9, 7.222), (10, 8.031), (11, 8.840),
        (16, 14.008), (32, 35.362), (48, 54.724), (64, 72.903), (80, 90.335),
        (96, 107.082), (112, 123.331), (128, 139.140), (144, 154.642),
        (160, 169.770), (176, 184.712), (192, 199.342), (208, 213.724),
        (224, 227.981), (240, 241.988), (255, 255.000),
    ];

    #[test]
    fn the_curve_is_the_system_conversion() {
        for (code, system) in SYSTEM_RAMP {
            let ours = rec709_video_to_srgb(code as f64 / 255.0) * 255.0;
            assert!((ours - system).abs() < 0.08,
                "code {code}: {ours:.3} here, {system:.3} from the system");
        }
    }

    #[test]
    fn the_curve_is_continuous_and_keeps_the_ends() {
        assert_eq!(rec709_video_to_srgb(0.0), 0.0);
        assert!((rec709_video_to_srgb(1.0) - 1.0).abs() < 1e-12);
        let below = rec709_video_to_srgb(REC709_VIDEO_TOE_END - 1e-9);
        let above = rec709_video_to_srgb(REC709_VIDEO_TOE_END + 1e-9);
        assert!((above - below).abs() < 1e-6, "toe join: {below} / {above}");
        // Mid-tones come out brighter: what the user sees in QuickTime.
        let mid = rec709_video_to_srgb(113.0 / 255.0) * 255.0;
        assert!((mid - 124.3).abs() < 0.2, "code 113 → {mid:.2}");
    }

    // ─── The painter, through egui's own renderer, off screen ─────────
    //
    // The pass is set up as eframe sets up the window's: the renderer's
    // default options (one sample, no depth, dither on) and the surface
    // format the app gets on macOS.

    const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

    struct Bench {
        gpu: vr180_pipeline::gpu::Device,
        renderer: egui_wgpu::Renderer,
    }

    impl Bench {
        fn new() -> Self {
            let gpu = vr180_pipeline::gpu::Device::new().expect("a GPU device");
            let mut renderer = egui_wgpu::Renderer::new(
                &gpu.device, TARGET, egui_wgpu::RendererOptions::default());
            renderer.callback_resources.insert(PreviewPainter::new(&gpu.device, TARGET));
            Self { gpu, renderer }
        }

        fn painter(&self) -> PreviewPainter {
            self.renderer.callback_resources.get::<PreviewPainter>().unwrap().clone()
        }

        /// An RGBA8 picture as the app's preview textures are viewed.
        fn picture(&self, w: u32, h: u32, rgba: &[u8]) -> wgpu::TextureView {
            let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
            let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test picture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.gpu.queue.write_texture(
                tex.as_image_copy(), rgba,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                size);
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        }

        /// Run one egui frame of `ui` on a `w × h` pixel screen and return
        /// what it draws, as RGBA rows.
        #[allow(deprecated)]
        fn draw(&mut self, w: u32, h: u32, scale: f32, mut ui: impl FnMut(&mut egui::Ui)) -> Vec<[u8; 4]> {
            let ctx = egui::Context::default();
            ctx.set_pixels_per_point(scale);
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO, egui::vec2(w as f32 / scale, h as f32 / scale))),
                ..Default::default()
            };
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |u| ui(u));
            });
            assert_eq!(out.pixels_per_point, scale);
            let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
            let (device, queue) = (&self.gpu.device, &self.gpu.queue);
            for (id, delta) in &out.textures_delta.set {
                self.renderer.update_texture(device, queue, *id, delta);
            }
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test window"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: scale };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut buffers = self.renderer.update_buffers(device, queue, &mut encoder, &jobs, &screen);
            {
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("test window"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.2, g: 0.4, b: 0.6, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                self.renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
            }
            buffers.push(encoder.finish());
            queue.submit(buffers);

            // Read the target back: B, G, R, A in memory.
            let row = (w * 4 + 255) & !255;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (row * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let mut px = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    let o = (y * row + x * 4) as usize;
                    px.push([mapped[o + 2], mapped[o + 1], mapped[o], mapped[o + 3]]);
                }
            }
            px
        }
    }

    /// 256 × 4: the top two rows step through every code (each channel on
    /// its own course), the bottom two run the other way.
    fn ramp_picture() -> Vec<u8> {
        let mut rgba = Vec::with_capacity(256 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..256u32 {
                let px = if y < 2 {
                    [x as u8, (255 - x) as u8, ((x * 3) % 256) as u8, 255]
                } else {
                    [(255 - x) as u8, (255 - x) as u8, (255 - x) as u8, 255]
                };
                rgba.extend_from_slice(&px);
            }
        }
        rgba
    }

    #[test]
    fn a_video_frame_is_drawn_through_the_system_curve() {
        let mut bench = Bench::new();
        let rgba = ramp_picture();
        let view = bench.picture(256, 4, &rgba);
        let picture = bench.painter().bind_picture(&bench.gpu.device, &view);
        let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let px = bench.draw(256, 4, 1.0, |ui| {
            ui.painter().add(PreviewPaint {
                picture: picture.clone(), slot: PaintSlot::Main, uv: full,
                encoding: PictureEncoding::Rec709Video,
            }.shape(ui.max_rect()));
        });
        // One texel per pixel: each output is that texel through the curve,
        // rounded with the dither (under half a code either way).
        let mut worst = 0.0f64;
        let mut sum = 0.0f64;
        for y in 0..4usize {
            for x in 0..256usize {
                let src = &rgba[(y * 256 + x) * 4..][..4];
                let out = px[y * 256 + x];
                assert_eq!(out[3], 255);
                for c in 0..3 {
                    let want = rec709_video_to_srgb(src[c] as f64 / 255.0) * 255.0;
                    let err = out[c] as f64 - want;
                    worst = worst.max(err.abs());
                    sum += err;
                }
            }
        }
        let mean = sum / (256.0 * 4.0 * 3.0);
        println!("video frame through the painter: worst {worst:.3}, mean {mean:+.4} of 255");
        assert!(worst < 0.99, "worst {worst}");
        assert!(mean.abs() < 0.03, "mean {mean}");
        // Mid-tones brighter, black and white where they were.
        assert_eq!(px[0][0], 0);
        assert_eq!(px[255][0], 255);
        assert!((px[113][0] as i32 - 124).abs() <= 1, "code 113 → {}", px[113][0]);
    }

    #[test]
    fn a_photograph_is_drawn_as_it_is() {
        let mut bench = Bench::new();
        let rgba = ramp_picture();
        let view = bench.picture(256, 4, &rgba);
        let picture = bench.painter().bind_picture(&bench.gpu.device, &view);
        let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let px = bench.draw(256, 4, 1.0, |ui| {
            ui.painter().add(PreviewPaint {
                picture: picture.clone(), slot: PaintSlot::Main, uv: full,
                encoding: PictureEncoding::Srgb,
            }.shape(ui.max_rect()));
        });
        for (i, out) in px.iter().enumerate() {
            assert_eq!(out[..], rgba[i * 4..][..4], "pixel {i}");
        }
    }

    /// The painter covers the pixels egui's image widget covers, with the
    /// part of the picture it shows there: a zoomed, off-grid rectangle on
    /// a 2× screen, drawn both ways.
    #[test]
    fn the_painter_lands_where_the_image_widget_does() {
        let mut bench = Bench::new();
        // 96 × 64 of noise, so a shifted or stretched sample shows.
        let (tw, th) = (96u32, 64u32);
        let mut state = 0x2545_F491u32;
        let mut rgba = Vec::with_capacity((tw * th * 4) as usize);
        for _ in 0..tw * th {
            for _ in 0..3 {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                rgba.push((state >> 24) as u8);
            }
            rgba.push(255);
        }
        let view = bench.picture(tw, th, &rgba);
        let picture = bench.painter().bind_picture(&bench.gpu.device, &view);
        let egui_id = bench.renderer.register_native_texture(
            &bench.gpu.device, &view, wgpu::FilterMode::Linear);
        let rect = egui::Rect::from_min_size(egui::pos2(10.3, 7.6), egui::vec2(200.4, 100.7));
        let uv = egui::Rect::from_min_max(egui::pos2(0.21, 0.13), egui::pos2(0.74, 0.93));

        let widget = bench.draw(512, 256, 2.0, |ui| {
            let sized = egui::load::SizedTexture::new(egui_id, rect.size());
            egui::Image::new(sized).uv(uv).paint_at(ui, rect);
        });
        let painter = bench.draw(512, 256, 2.0, |ui| {
            ui.painter().add(PreviewPaint {
                picture: picture.clone(), slot: PaintSlot::Main, uv,
                encoding: PictureEncoding::Srgb,
            }.shape(rect));
        });
        // The background where neither draws, the picture where both do.
        let background = widget[0];
        let covered = widget.iter().filter(|p| **p != background).count();
        assert!(covered > 70_000, "the widget drew {covered} pixels");
        let mut differing = 0usize;
        let mut worst = 0i32;
        for (a, b) in widget.iter().zip(&painter) {
            let d = (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap();
            if d != 0 { differing += 1; }
            worst = worst.max(d);
        }
        println!("painter vs image widget: {differing} of {} pixels differ, worst {worst}", widget.len());
        assert!(worst <= 1, "worst {worst}");
        assert!(differing * 200 < covered, "{differing} of {covered} pixels differ");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_layer_is_written_only_when_it_is_not_srgb_and_gets_its_own_back() {
        platform::test_support::round_trip();
    }
}
