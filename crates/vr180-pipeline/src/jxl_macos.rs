//! JPEG XL decoding through ImageIO (macOS): one codestream of 16-bit RGB
//! samples to its codes, with no colour conversion.
//!
//! ImageIO decodes JPEG XL natively. The image it hands back keeps the
//! codestream's own colour encoding; its bitmap is read as it is (the
//! provider's bytes), so the values are the stored codes.

use std::ffi::c_void;

use core_foundation::base::{CFRelease, TCFType};
use core_foundation::data::{CFData, CFDataRef};

#[link(name = "ImageIO", kind = "framework")]
extern "C" {
    fn CGImageSourceCreateWithData(data: CFDataRef, options: *const c_void) -> *mut c_void;
    fn CGImageSourceCreateImageAtIndex(source: *mut c_void, index: usize, options: *const c_void) -> *mut c_void;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGImageGetWidth(image: *mut c_void) -> usize;
    fn CGImageGetHeight(image: *mut c_void) -> usize;
    fn CGImageGetBitsPerComponent(image: *mut c_void) -> usize;
    fn CGImageGetBitsPerPixel(image: *mut c_void) -> usize;
    fn CGImageGetBytesPerRow(image: *mut c_void) -> usize;
    fn CGImageGetBitmapInfo(image: *mut c_void) -> u32;
    fn CGImageGetDataProvider(image: *mut c_void) -> *mut c_void;
    fn CGDataProviderCopyData(provider: *mut c_void) -> CFDataRef;
    fn CGImageRelease(image: *mut c_void);
}

const ALPHA_MASK: u32 = 0x1F;
const FLOAT_COMPONENTS: u32 = 1 << 8;
const BYTE_ORDER_MASK: u32 = 0x7000;
const BYTE_ORDER_16_LITTLE: u32 = 1 << 12;
/// Alpha layouts whose extra component comes first.
const ALPHA_FIRST: [u32; 3] = [2, 4, 6];

/// Releases a Core Foundation object when dropped.
struct Owned(*mut c_void);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer came from a Create / Copy call and is released once.
            unsafe { CFRelease(self.0 as *const c_void) };
        }
    }
}

/// Decode one JPEG XL codestream: `(width, height, samples)`, the samples
/// interleaved R, G, B, row-major, as stored.
pub fn decode_rgb16(bytes: &[u8]) -> Result<(usize, usize, Vec<u16>), String> {
    let data = CFData::from_buffer(bytes);
    // SAFETY: plain C calls on objects this function owns; every created
    // object is released by `Owned` or `CGImageRelease`; the provider's data
    // is copied into a Vec before it is released.
    unsafe {
        let source = Owned(CGImageSourceCreateWithData(data.as_concrete_TypeRef(), std::ptr::null()));
        if source.0.is_null() {
            return Err("JPEG XL: not an image ImageIO reads".into());
        }
        let image = CGImageSourceCreateImageAtIndex(source.0, 0, std::ptr::null());
        if image.is_null() {
            return Err("JPEG XL: ImageIO could not decode the codestream".into());
        }
        struct Image(*mut c_void);
        impl Drop for Image {
            fn drop(&mut self) {
                // SAFETY: created above, released once.
                unsafe { CGImageRelease(self.0) };
            }
        }
        let image = Image(image);
        let (w, h) = (CGImageGetWidth(image.0), CGImageGetHeight(image.0));
        let (bits, pixel_bits, row_bytes, info) = (
            CGImageGetBitsPerComponent(image.0), CGImageGetBitsPerPixel(image.0), CGImageGetBytesPerRow(image.0), CGImageGetBitmapInfo(image.0));
        if bits != 16 || !(pixel_bits == 48 || pixel_bits == 64) || info & FLOAT_COMPONENTS != 0 {
            return Err(format!("JPEG XL: {bits}-bit components in {pixel_bits}-bit pixels (bitmap info {info:#x}) is not 16-bit RGB"));
        }
        let pixels = Owned(CGDataProviderCopyData(CGImageGetDataProvider(image.0)) as *mut c_void);
        if pixels.0.is_null() {
            return Err("JPEG XL: ImageIO returned no pixels".into());
        }
        let pixels = CFData::wrap_under_get_rule(pixels.0 as CFDataRef);
        let raw = pixels.bytes();
        let step = pixel_bits / 8;
        if row_bytes < w * step || raw.len() < (h - 1) * row_bytes + w * step {
            return Err("JPEG XL: ImageIO's bitmap is smaller than its image".into());
        }
        let first = if step == 8 && ALPHA_FIRST.contains(&(info & ALPHA_MASK)) { 2 } else { 0 };
        let little = info & BYTE_ORDER_MASK == BYTE_ORDER_16_LITTLE;
        let mut out = Vec::with_capacity(w * h * 3);
        for row in raw.chunks(row_bytes).take(h) {
            for px in row[..w * step].chunks_exact(step) {
                for c in 0..3 {
                    let b = [px[first + 2 * c], px[first + 2 * c + 1]];
                    out.push(if little { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) });
                }
            }
        }
        Ok((w, h, out))
    }
}
