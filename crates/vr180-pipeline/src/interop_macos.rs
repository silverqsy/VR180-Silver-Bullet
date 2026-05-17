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
}

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetPlaneCount(s: IOSurfaceRef) -> usize;
    fn IOSurfaceGetWidthOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerRowOfPlane(s: IOSurfaceRef, plane: usize) -> usize;
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
