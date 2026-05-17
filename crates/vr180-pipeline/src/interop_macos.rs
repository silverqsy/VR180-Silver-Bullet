//! macOS IOSurface ↔ Metal ↔ wgpu zero-copy substrate.
//!
//! Phase 0.6 (decode) hops through host memory:
//!   VT decoder → CVPixelBuffer → `av_hwframe_transfer_data` → NV12 host
//!     → swscale → packed RGB8 → `queue.write_texture` → wgpu compute
//!
//! Phase 0.6.5 (this module) skips the host-memory hop on the decode
//! side:
//!   VT decoder → CVPixelBuffer → CVPixelBufferGetIOSurface
//!     → MTLTexture (Y as R8Unorm, UV as Rg8Unorm) → wgpu::Texture
//!     → compute kernel reads Y + UV planes directly
//!
//! That saves the `av_hwframe_transfer_data` memcpy (~50 MB / 8K frame /
//! stream) PLUS the `queue.write_texture` upload of the same size — a
//! couple-hundred-MB-per-frame bandwidth win at 8K dual-stream.
//!
//! The NV12-aware EAC assembly kernel that consumes these textures
//! lands in Phase 0.6.6 (`nv12_to_eac_cross.wgsl`).
//!
//! ## Implementation notes
//!
//! - `RetainedIOSurface` retains past the AVFrame's recycle window —
//!   VideoToolbox-decoded `AVFrame`s own their CVPixelBuffer only for
//!   the lifetime of the frame; once the frame is reused for the
//!   next decoded packet, the CVPixelBuffer (and its IOSurface) are
//!   gone unless we hold our own retain.
//! - The Cocoa "new" prefix on `newTextureWithDescriptor:iosurface:plane:`
//!   means the returned MTLTexture is `+1` retained — we wrap with
//!   `metal::Texture::from_ptr` which DOESN'T add another retain.
//! - On Apple Silicon's unified memory, `MTLStorageMode::Shared` is
//!   the right storage mode for IOSurface-backed textures: writes
//!   from VideoToolbox / Metal compute / wgpu-hal are all visible to
//!   each other without explicit synchronisation barriers.
//! - The `metal = "0.28"` and `wgpu = "0.20"` pins are not arbitrary —
//!   they MUST match what wgpu-hal's Metal layer expects internally,
//!   otherwise the `metal::Texture` type we pass into
//!   `Device::texture_from_raw` fails to round-trip and the hal escape
//!   silently produces a broken `wgpu::Texture`.

#![cfg(target_os = "macos")]

use crate::{Error, Result};
use ffmpeg_next as ffmpeg;
use foreign_types::ForeignType;
use objc::{msg_send, sel, sel_impl};
use std::ffi::c_void;
use std::ptr::NonNull;

// ─── Raw FFI ──────────────────────────────────────────────────────────

/// Opaque IOSurface reference. CFType-retain-managed.
pub type IOSurfaceRef = *mut c_void;
/// Opaque CVPixelBuffer reference. CFType-retain-managed.
pub type CVPixelBufferRef = *mut c_void;
type CFTypeRef = *const c_void;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    fn CFRelease(cf: CFTypeRef);
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    /// Get-Rule: returned IOSurface is NOT retained by this call.
    /// Returns NULL if the pixel buffer isn't IOSurface-backed.
    fn CVPixelBufferGetIOSurface(pixel_buffer: CVPixelBufferRef) -> IOSurfaceRef;

    /// Create a CVPixelBuffer with the given width / height / pixel
    /// format and attribute dictionary. Returns 0 (kCVReturnSuccess)
    /// on success and writes the buffer to `pixel_buffer_out`. The
    /// returned buffer is `+1` retained (Create-Rule).
    fn CVPixelBufferCreate(
        allocator: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format_type: u32,
        pixel_buffer_attributes: CFDictionaryRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> i32;
}

// FourCC for BGRA 32-bit. Matches `kCVPixelFormatType_32BGRA` in CoreVideo.
const K_CV_PIXEL_FORMAT_TYPE_32BGRA: u32 = u32::from_be_bytes(*b"BGRA");

type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetPlaneCount(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetWidthOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerRowOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetWidth(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetHeight(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerRow(s: IOSurfaceRef) -> usize;
}

// ─── RAII wrapper ─────────────────────────────────────────────────────

/// RAII wrapper around an IOSurface that holds a `CFRetain` reference.
/// `Drop` does `CFRelease`. See `extract_iosurface_from_vt_frame` for
/// the lifetime rationale (CVPixelBuffer goes away when the AVFrame
/// recycles; the IOSurface needs its own retain to outlive that).
pub struct RetainedIOSurface {
    inner: NonNull<c_void>,
}

impl RetainedIOSurface {
    /// Retain an IOSurface obtained via Get-Rule API (e.g.
    /// `CVPixelBufferGetIOSurface`).
    ///
    /// # Safety
    /// `iosurface` must be a valid, currently-retained IOSurfaceRef.
    pub unsafe fn retain(iosurface: IOSurfaceRef) -> Self {
        let inner = NonNull::new(iosurface)
            .expect("RetainedIOSurface::retain: NULL IOSurfaceRef");
        let retained = CFRetain(inner.as_ptr() as CFTypeRef);
        Self {
            inner: NonNull::new(retained as *mut c_void)
                .expect("CFRetain returned NULL"),
        }
    }

    pub fn as_raw(&self) -> IOSurfaceRef { self.inner.as_ptr() }
    pub fn plane_count(&self) -> usize { unsafe { IOSurfaceGetPlaneCount(self.as_raw()) } }
    pub fn plane_width(&self, p: usize) -> usize { unsafe { IOSurfaceGetWidthOfPlane(self.as_raw(), p) } }
    pub fn plane_height(&self, p: usize) -> usize { unsafe { IOSurfaceGetHeightOfPlane(self.as_raw(), p) } }
    pub fn plane_bytes_per_row(&self, p: usize) -> usize { unsafe { IOSurfaceGetBytesPerRowOfPlane(self.as_raw(), p) } }
}

impl Drop for RetainedIOSurface {
    fn drop(&mut self) {
        unsafe { CFRelease(self.inner.as_ptr() as CFTypeRef); }
    }
}

// SAFETY: IOSurface is internally thread-safe; cross-process sharing
// is its design purpose. CFRetain/CFRelease are atomic.
unsafe impl Send for RetainedIOSurface {}
unsafe impl Sync for RetainedIOSurface {}

// ─── NV12-shaped descriptor (cached plane geometry) ───────────────────

/// An NV12 IOSurface + its plane geometry pre-cached so callers don't
/// re-cross the FFI boundary every frame.
pub struct IOSurfaceNv12Descriptor {
    pub surface: RetainedIOSurface,
    /// Full-frame Y-plane width (== UV-plane width × 2).
    pub width: u32,
    /// Full-frame Y-plane height (== UV-plane height × 2).
    pub height: u32,
    pub y_bytes_per_row: u32,
    pub uv_bytes_per_row: u32,
}

impl IOSurfaceNv12Descriptor {
    pub fn new(surface: RetainedIOSurface) -> Result<Self> {
        if surface.plane_count() != 2 {
            return Err(Error::Ffmpeg(format!(
                "expected NV12 (2 planes), got {} planes", surface.plane_count()
            )));
        }
        let width = surface.plane_width(0) as u32;
        let height = surface.plane_height(0) as u32;
        let y_bytes_per_row = surface.plane_bytes_per_row(0) as u32;
        let uv_bytes_per_row = surface.plane_bytes_per_row(1) as u32;
        Ok(Self { surface, width, height, y_bytes_per_row, uv_bytes_per_row })
    }
}

// ─── FFmpeg-side extraction ───────────────────────────────────────────

/// Pull the IOSurface backing a VideoToolbox-decoded `AVFrame`.
/// FFmpeg's VT hwaccel stores the `CVPixelBufferRef` at
/// `AVFrame::data[3]`. We get its IOSurface (Get-Rule) and `CFRetain`
/// it so the surface outlives the AVFrame (which gets reused or
/// freed on the next decode iteration).
pub fn extract_iosurface_from_vt_frame(
    frame: &ffmpeg::frame::Video,
) -> Result<RetainedIOSurface> {
    let fmt = frame.format();
    if fmt != ffmpeg::format::Pixel::VIDEOTOOLBOX {
        return Err(Error::Ffmpeg(format!(
            "frame is not VideoToolbox-decoded (format = {fmt:?})"
        )));
    }
    // SAFETY: `data[3]` is a Copy pointer field on AVFrame; we read it
    // strictly within this borrow of `frame`. ffmpeg-next's `as_ptr`
    // is unsafe because the returned pointer's lifetime is tied to the
    // caller's borrow; we never extend it past this block.
    let cv_pix_buf: CVPixelBufferRef = unsafe {
        let raw_frame = frame.as_ptr();
        (*raw_frame).data[3] as CVPixelBufferRef
    };
    if cv_pix_buf.is_null() {
        return Err(Error::Ffmpeg(
            "AVFrame::data[3] NULL (VT decoder produced no CVPixelBuffer)".into()
        ));
    }
    let iosurface = unsafe { CVPixelBufferGetIOSurface(cv_pix_buf) };
    if iosurface.is_null() {
        return Err(Error::Ffmpeg(
            "CVPixelBufferGetIOSurface returned NULL".into()
        ));
    }
    // Get-Rule → we own no retain yet. Retain to extend lifetime.
    Ok(unsafe { RetainedIOSurface::retain(iosurface) })
}

// ─── Metal texture wrapping (selector: newTextureWithDescriptor:iosurface:plane:) ──

/// Build a `metal::Texture` view over one plane of an IOSurface via
/// `MTLDevice.newTextureWithDescriptor:iosurface:plane:`. metal-rs 0.28
/// does not expose this selector, so we go through `objc::msg_send!`.
///
/// Returns `None` if Metal refuses the descriptor (unsupported pixel
/// format / device combo) — `msg_send!` would happily wrap nil as a
/// dangling `metal::Texture`, so we null-check first.
pub fn metal_texture_from_iosurface_plane(
    device: &metal::DeviceRef,
    iosurface: &RetainedIOSurface,
    plane: usize,
    pixel_format: metal::MTLPixelFormat,
    width: u32,
    height: u32,
) -> Option<metal::Texture> {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_texture_type(metal::MTLTextureType::D2);
    descriptor.set_pixel_format(pixel_format);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(1);
    descriptor.set_sample_count(1);
    descriptor.set_array_length(1);
    // Apple Silicon unified memory — IOSurface-backed textures use Shared.
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_usage(
        metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::ShaderWrite,
    );

    let raw_iosurface = iosurface.as_raw();
    let raw_tex: *mut objc::runtime::Object = unsafe {
        msg_send![
            device,
            newTextureWithDescriptor: &*descriptor
            iosurface: raw_iosurface
            plane: plane as u64
        ]
    };
    if raw_tex.is_null() { return None; }
    // The "new" Cocoa prefix means +1 retain. `from_ptr` wraps without
    // an extra retain — metal::Texture's Drop releases it.
    Some(unsafe { metal::Texture::from_ptr(raw_tex as _) })
}

/// One IOSurface plane wrapped as a `wgpu::Texture`. Holds a retain on
/// the surface so dropping this struct frees both cleanly.
pub struct IOSurfacePlaneTexture {
    pub surface: RetainedIOSurface,
    pub texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

/// Wrap a single IOSurface plane as a `wgpu::Texture`:
/// `IOSurface plane → MTLTexture → wgpu-hal Metal Texture → wgpu::Texture`.
///
/// # Safety
/// - `device` MUST be the wgpu device whose backend is Metal. We surface
///   the wrong-backend case as `Error::Wgpu("not Metal backend")`.
/// - `(plane, format)` must agree with the surface's actual plane layout.
///   Apple doesn't validate at texture-creation time; a mismatch silently
///   yields garbage pixel reads.
pub fn wgpu_texture_from_iosurface_plane(
    device: &wgpu::Device,
    surface: RetainedIOSurface,
    plane: usize,
    metal_format: metal::MTLPixelFormat,
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    label: &str,
) -> Result<IOSurfacePlaneTexture> {
    // 1. Clone the underlying metal::Device out of wgpu's hal escape.
    //    The clone is a cheap NSObject retain.
    let metal_device = unsafe {
        device.as_hal::<wgpu::hal::api::Metal, _, _>(|hal| {
            hal.map(|d| d.raw_device().lock().clone())
        })
    }
    .flatten()
    .ok_or_else(|| Error::Wgpu(
        "wgpu device is not on the Metal backend (as_hal returned None)".into()
    ))?;

    // 2. metal::Texture plane view via newTextureWithDescriptor:iosurface:plane:
    let metal_texture = metal_texture_from_iosurface_plane(
        &metal_device, &surface, plane, metal_format, width, height,
    ).ok_or_else(|| Error::Wgpu(format!(
        "newTextureWithDescriptor:iosurface:plane: returned nil for plane {plane}, \
         format={metal_format:?}, {width}×{height}"
    )))?;

    // 3. Wrap as wgpu-hal Texture.
    let hal_texture = unsafe {
        <wgpu::hal::api::Metal as wgpu::hal::Api>::Device::texture_from_raw(
            metal_texture,
            wgpu_format,
            metal::MTLTextureType::D2,
            1, // array_layers
            1, // mip_levels
            wgpu::hal::CopyExtent { width, height, depth: 1 },
        )
    };

    // 4. Hand to wgpu. Descriptor must mirror the hal texture exactly.
    let usage = wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::STORAGE_BINDING;
    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage,
        view_formats: &[],
    };
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &wgpu_desc)
    };

    Ok(IOSurfacePlaneTexture {
        surface, texture, width, height, format: wgpu_format,
    })
}

// ─── Encode-side: CVPixelBuffer creation ──────────────────────────────
//
// Phase 0.7.5.6 adds the *creation* side of the IOSurface/CVPixelBuffer
// bridge so the GPU can write the encoder's input frame directly. The
// existing path took an IOSurface from VideoToolbox's decoder and wrapped
// it for wgpu *consumption*; the new path creates an IOSurface, wraps it
// for wgpu *production*, and feeds the resulting CVPixelBuffer to
// VideoToolbox's encoder.
//
// Layout: single-plane 32BGRA (CoreVideo's "BGRA" fourcc =
// `kCVPixelFormatType_32BGRA`). VT encoder accepts this directly. We
// write to the IOSurface with channels swapped in the shader (the wgpu
// texture is viewed as Rgba8Unorm but the bytes land in B,G,R,A order),
// so VT reads correct colors. This avoids requiring the wgpu
// `BGRA8UNORM_STORAGE` feature, which not every Vulkan/DX12 adapter
// supports — keeps the code path uniform across backends.

/// RAII wrapper around a CVPixelBuffer. Owns +1 retain; `Drop` releases.
pub struct RetainedCVPixelBuffer {
    inner: NonNull<c_void>,
}

impl RetainedCVPixelBuffer {
    /// Take ownership of a CVPixelBuffer obtained via Create-Rule
    /// (e.g. `CVPixelBufferCreate`). The buffer is +1 retained on entry;
    /// `Drop` releases that retain.
    ///
    /// # Safety
    /// `pb` must be a valid CVPixelBufferRef with refcount ≥ 1, AND the
    /// caller must transfer their reference to us (Create-Rule). Calling
    /// this on a Get-Rule pointer would over-release.
    pub unsafe fn from_create_rule(pb: CVPixelBufferRef) -> Self {
        let inner = NonNull::new(pb).expect("RetainedCVPixelBuffer: NULL ref");
        Self { inner }
    }

    pub fn as_raw(&self) -> CVPixelBufferRef { self.inner.as_ptr() }
}

impl Drop for RetainedCVPixelBuffer {
    fn drop(&mut self) {
        unsafe { CFRelease(self.inner.as_ptr() as CFTypeRef); }
    }
}

// SAFETY: CVPixelBuffer is internally thread-safe; CFRetain/CFRelease
// are atomic.
unsafe impl Send for RetainedCVPixelBuffer {}
unsafe impl Sync for RetainedCVPixelBuffer {}

/// One IOSurface-backed BGRA CVPixelBuffer + a wgpu texture that aliases
/// its bytes. Writing to `wgpu_tex` writes through to the IOSurface;
/// after a `queue.submit() + device.poll(Wait)` the bytes are visible to
/// the VideoToolbox encoder via `pixel_buffer`.
///
/// Designed for one-shot per-frame allocation; for sustained throughput
/// add a CVPixelBufferPool layer on top (deferred — the per-frame
/// alloc is ~100 µs on Apple Silicon which is well under the budget at
/// 30 fps).
pub struct EncodePixelBuffer {
    pub pixel_buffer: RetainedCVPixelBuffer,
    /// Held to keep the IOSurface alive past the CVPixelBuffer's own
    /// retain — defensive in case anything ever releases the
    /// CVPixelBuffer before the encoder is done with the IOSurface.
    pub iosurface: RetainedIOSurface,
    pub wgpu_tex: wgpu::Texture,
    pub width: u32,
    pub height: u32,
}

/// Create a new IOSurface-backed BGRA CVPixelBuffer of the given size
/// and wrap its bytes as a wgpu::Texture (viewed as Rgba8Unorm — see
/// the module note on the channel swap).
///
/// Returns `Err` if:
/// - `CVPixelBufferCreate` fails (non-zero CVReturn).
/// - The pixel buffer isn't IOSurface-backed (shouldn't happen with the
///   attributes we pass, but we check defensively).
/// - The wgpu device isn't on the Metal backend.
pub fn create_bgra_encode_buffer(
    device: &wgpu::Device,
    width: u32, height: u32,
) -> Result<EncodePixelBuffer> {
    // Pass NULL attrs but with an empty CFDictionary holding only the
    // kCVPixelBufferIOSurfacePropertiesKey: an empty IOSurface props
    // dict triggers IOSurface backing with default attributes. Building
    // a real CFDictionary from Rust without the `core-foundation` crate
    // wrangling helpers is verbose, so we build a tiny one inline.
    let attrs = unsafe { build_iosurface_attrs() };

    let mut pb: CVPixelBufferRef = std::ptr::null_mut();
    let ret = unsafe {
        CVPixelBufferCreate(
            std::ptr::null(),         // default allocator
            width as usize,
            height as usize,
            K_CV_PIXEL_FORMAT_TYPE_32BGRA,
            attrs as CFDictionaryRef,
            &mut pb,
        )
    };
    // Release the attrs dict — CVPixelBufferCreate retained what it needed.
    unsafe { CFRelease(attrs as CFTypeRef); }
    if ret != 0 || pb.is_null() {
        return Err(Error::Wgpu(format!(
            "CVPixelBufferCreate returned {ret} (pb={:?})", pb
        )));
    }
    let pixel_buffer = unsafe { RetainedCVPixelBuffer::from_create_rule(pb) };

    // Extract IOSurface (Get-Rule → CFRetain to keep it alive).
    let raw_iosurface = unsafe { CVPixelBufferGetIOSurface(pixel_buffer.as_raw()) };
    if raw_iosurface.is_null() {
        return Err(Error::Wgpu(
            "fresh CVPixelBuffer has no IOSurface backing (attrs ignored?)".into()
        ));
    }
    let iosurface = unsafe { RetainedIOSurface::retain(raw_iosurface) };

    // Sanity-check geometry — BGRA is single-plane, so use the
    // non-plane accessors (IOSurfaceGetWidth, not GetWidthOfPlane).
    let surf_w = unsafe { IOSurfaceGetWidth(iosurface.as_raw()) } as u32;
    let surf_h = unsafe { IOSurfaceGetHeight(iosurface.as_raw()) } as u32;
    if surf_w != width || surf_h != height {
        return Err(Error::Wgpu(format!(
            "IOSurface geometry mismatch: expected {width}×{height}, got {surf_w}×{surf_h}"
        )));
    }

    // Wrap as wgpu::Texture, viewed as Rgba8Unorm.
    // BGRA IOSurface bytes are interpreted as RGBA from the shader's
    // perspective: a Metal sample of (b,g,r,a) bytes via the Rgba8Unorm
    // view returns (b,g,r,a) as (r,g,b,a) — the channel swap is
    // baked into how we WRITE in the shader, not how the encoder reads.
    let metal_format = metal::MTLPixelFormat::RGBA8Unorm;
    let wgpu_format = wgpu::TextureFormat::Rgba8Unorm;

    // For a single-plane BGRA buffer, the "plane index" is 0 even though
    // IOSurfaceGetPlaneCount returns 0 (non-planar surfaces). Metal's
    // newTextureWithDescriptor:iosurface:plane: accepts plane=0 for
    // non-planar surfaces — that's the documented contract.
    let metal_device = unsafe {
        device.as_hal::<wgpu::hal::api::Metal, _, _>(|hal| {
            hal.map(|d| d.raw_device().lock().clone())
        })
    }
    .flatten()
    .ok_or_else(|| Error::Wgpu(
        "wgpu device is not on the Metal backend (as_hal returned None)".into()
    ))?;

    let metal_tex = metal_texture_from_iosurface_plane(
        &metal_device, &iosurface, 0, metal_format, width, height,
    ).ok_or_else(|| Error::Wgpu(format!(
        "newTextureWithDescriptor:iosurface:plane:0 returned nil for BGRA {width}×{height}"
    )))?;

    let hal_texture = unsafe {
        <wgpu::hal::api::Metal as wgpu::hal::Api>::Device::texture_from_raw(
            metal_tex,
            wgpu_format,
            metal::MTLTextureType::D2,
            1, 1,
            wgpu::hal::CopyExtent { width, height, depth: 1 },
        )
    };

    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some("encode_pb_bgra"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
             | wgpu::TextureUsages::STORAGE_BINDING
             | wgpu::TextureUsages::COPY_SRC
             | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let wgpu_tex = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &wgpu_desc)
    };

    Ok(EncodePixelBuffer {
        pixel_buffer, iosurface, wgpu_tex, width, height,
    })
}

/// Build a tiny CFDictionary `{ kCVPixelBufferIOSurfacePropertiesKey: {} }`
/// to pass to `CVPixelBufferCreate`. The presence of the key (with any
/// value, even an empty dict) triggers IOSurface backing. Using
/// `core-foundation`'s typed builders for one key is overkill; we go
/// raw FFI through `CFDictionaryCreate` with a single (key, value) pair.
///
/// Returns a `+1` retained CFDictionaryRef; caller must `CFRelease`.
unsafe fn build_iosurface_attrs() -> *const c_void {
    // Constants from CoreVideo.framework headers. Linked weakly because
    // they're CFStringRef globals, not regular C symbols — we declare
    // them in the `extern` block and link the framework.
    extern "C" {
        static kCVPixelBufferIOSurfacePropertiesKey: CFTypeRef;
        fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            num_values: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *const c_void;

        // kCFTypeDictionaryKeyCallBacks / kCFTypeDictionaryValueCallBacks
        // — predefined callback structs for retaining CFType keys/values.
        static kCFTypeDictionaryKeyCallBacks: c_void;
        static kCFTypeDictionaryValueCallBacks: c_void;
    }

    // Empty inner dict for IOSurface properties (use defaults).
    let inner_keys = std::ptr::null::<*const c_void>();
    let inner_vals = std::ptr::null::<*const c_void>();
    let inner = CFDictionaryCreate(
        std::ptr::null(),
        inner_keys, inner_vals, 0,
        &kCFTypeDictionaryKeyCallBacks as *const _ as *const c_void,
        &kCFTypeDictionaryValueCallBacks as *const _ as *const c_void,
    );

    // Outer dict: { kCVPixelBufferIOSurfacePropertiesKey: <inner> }.
    let key = kCVPixelBufferIOSurfacePropertiesKey;
    let val: *const c_void = inner;
    let outer = CFDictionaryCreate(
        std::ptr::null(),
        &key as *const _ as *const *const c_void,
        &val as *const _ as *const *const c_void,
        1,
        &kCFTypeDictionaryKeyCallBacks as *const _ as *const c_void,
        &kCFTypeDictionaryValueCallBacks as *const _ as *const c_void,
    );

    // The outer dict retains the inner via the callbacks; release our
    // local +1 on the inner.
    CFRelease(inner as CFTypeRef);
    outer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the CFRetain / CFRelease FFI is linked and balanced.
    /// A `CFNumber` is the simplest CFType we can reach without a
    /// Metal device.
    #[test]
    fn cfretain_cfrelease_link() {
        use core_foundation::base::TCFType;
        use core_foundation::number::CFNumber;
        let n = CFNumber::from(42i32);
        let raw = n.as_concrete_TypeRef() as CFTypeRef;
        // Retain → release; if our signatures were wrong this would
        // either fail to link or crash on the imbalance.
        unsafe {
            let _ = CFRetain(raw);
            CFRelease(raw);
        }
    }
}
