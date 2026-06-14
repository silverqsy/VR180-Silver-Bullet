//! In-process VideoToolbox temporal noise reduction (`VTTemporalNoiseFilter`)
//! driven directly through the Objective-C runtime via `objc2` — no Swift
//! helper, no subprocess, no extra bundled binary.
//!
//! This mirrors the algorithm the Python app reaches through its `vt_denoise`
//! Swift binary, but runs it directly: the four VideoToolbox classes
//! (`VTFrameProcessor`, `VTTemporalNoiseFilterConfiguration`,
//! `VTTemporalNoiseFilterParameters`, `VTFrameProcessorFrame`) are all
//! `@interface … : NSObject` ObjC classes (confirmed from the SDK headers),
//! so they are reachable via `msg_send!`. We use the **synchronous**
//! `processWithParameters:error:` (which Swift can't see — it's
//! `NS_SWIFT_UNAVAILABLE`), so there are no completion-handler blocks or
//! dispatch semaphores: each frame processes inline.
//!
//! ## Pixel-format dance
//!
//! `VTTemporalNoiseFilter` only accepts source/destination buffers in one of
//! Apple's internal lossless-compressed pixel formats (the
//! `supportedSourcePixelFormats` list — all YCbCr, no plain RGBA). So we keep
//! a reusable 16-bit `64RGBALE` I/O buffer and bounce RGBA ⇄ lossless-YCbCr
//! through a `VTPixelTransferSession` (GPU colour-matrix convert), exactly as
//! the Swift helper does. We prefer a 10-bit full-range lossless format so a
//! 10-bit (P010 / DJI OSV) source keeps its precision through the filter.
//!
//! ## Temporal window
//!
//! The filter needs `previousFrameCount` past + `nextFrameCount` future
//! reference frames around each source frame. [`VtDenoiser`] keeps a sliding
//! ring of source frames and emits one denoised frame once enough look-ahead
//! is buffered; [`VtDenoiser::finish`] drains the tail. Frame order and count
//! are preserved 1:1 (N frames in → N frames out, delayed by `nextFrameCount`),
//! so downstream stabilization-by-index and audio sync stay aligned. Clips
//! with a single frame (no possible reference) are the only degenerate case
//! and are handled by the caller falling back to passthrough.

#![cfg(target_os = "macos")]

use crate::{Error, Result};
use core::ffi::c_void;
use std::collections::VecDeque;
use std::ptr::null_mut;

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::msg_send;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, Bool};

use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef as CfStringRef};

// ─── C type aliases ────────────────────────────────────────────────────
type OSType = u32;
type OSStatus = i32;
type CVReturn = i32;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFStringRefRaw = *const c_void;
type CFTypeRefRaw = *const c_void;
type CVPixelBufferPoolRef = *mut c_void;
type VTPixelTransferSessionRef = *mut c_void;

/// Opaque `CVBuffer` pointee. The ObjC runtime types `CVPixelBufferRef`
/// method arguments as `^{__CVBuffer=}`; objc2's debug message verifier
/// rejects a bare `^v` (`*mut c_void`), so we give the pointee the matching
/// `RefEncode` and alias `CVPixelBufferRef` to `*mut CVBuffer`. ABI-identical
/// to a void pointer, so the plain-C CoreVideo calls are unaffected.
#[repr(C)]
struct CVBuffer {
    _private: [u8; 0],
}
unsafe impl RefEncode for CVBuffer {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__CVBuffer", &[]));
}
/// Kept as `*mut c_void` so these plain-C CoreVideo extern signatures match
/// the ones in `interop_macos` (avoids a `clashing_extern_declarations`
/// warning). Only the `initWithBuffer:` message send needs the
/// `^{__CVBuffer=}` encoding — there we cast to `*mut CVBuffer` at the call.
type CVPixelBufferRef = *mut c_void;

const KCV_ATTACHMENT_SHOULD_PROPAGATE: u32 = 1;
const KCV_PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 1;

/// `CMTime`, the one struct we pass through the ObjC ABI by value. Layout is
/// `{value:int64, timescale:int32, flags:uint32, epoch:int64}` → encoding
/// `qiIq` (24 bytes, no padding). The struct *name* only matters with objc2's
/// `verify` feature (off here), so the anonymous `"?"` tag is fine.
#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}
unsafe impl Encode for CMTime {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[i64::ENCODING, i32::ENCODING, u32::ENCODING, i64::ENCODING],
    );
}
unsafe impl RefEncode for CMTime {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
const CMTIME_FLAG_VALID: u32 = 1; // kCMTimeFlags_Valid

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferIOSurfacePropertiesKey: CFStringRefRaw;
    static kCVPixelBufferWidthKey: CFStringRefRaw;
    static kCVPixelBufferHeightKey: CFStringRefRaw;
    static kCVPixelBufferPixelFormatTypeKey: CFStringRefRaw;
    static kCVPixelBufferPoolMinimumBufferCountKey: CFStringRefRaw;
    static kCVImageBufferYCbCrMatrixKey: CFStringRefRaw;
    static kCVImageBufferYCbCrMatrix_ITU_R_709_2: CFStringRefRaw;
    static kCVPixelFormatBitsPerComponent: CFStringRefRaw;

    fn CVPixelBufferCreate(
        alloc: CFAllocatorRef,
        width: usize,
        height: usize,
        fmt: OSType,
        attrs: CFDictionaryRef,
        out: *mut CVPixelBufferRef,
    ) -> CVReturn;
    fn CVPixelBufferPoolCreate(
        alloc: CFAllocatorRef,
        pool_attrs: CFDictionaryRef,
        buf_attrs: CFDictionaryRef,
        out: *mut CVPixelBufferPoolRef,
    ) -> CVReturn;
    fn CVPixelBufferPoolCreatePixelBuffer(
        alloc: CFAllocatorRef,
        pool: CVPixelBufferPoolRef,
        out: *mut CVPixelBufferRef,
    ) -> CVReturn;
    fn CVPixelBufferLockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> CVReturn;
    fn CVPixelBufferUnlockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> CVReturn;
    fn CVPixelBufferGetBaseAddress(pb: CVPixelBufferRef) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pb: CVPixelBufferRef) -> usize;
    fn CVPixelFormatDescriptionCreateWithPixelFormatType(
        alloc: CFAllocatorRef,
        fmt: OSType,
    ) -> CFDictionaryRef;
    fn CVBufferSetAttachment(
        buf: CVPixelBufferRef,
        key: CFStringRefRaw,
        value: CFTypeRefRaw,
        mode: u32,
    );
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CFTypeRefRaw);
    fn CFDictionaryGetValue(dict: CFDictionaryRef, key: *const c_void) -> *const c_void;
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTPixelTransferSessionCreate(
        alloc: CFAllocatorRef,
        out: *mut VTPixelTransferSessionRef,
    ) -> OSStatus;
    fn VTPixelTransferSessionTransferImage(
        session: VTPixelTransferSessionRef,
        src: CVPixelBufferRef,
        dst: CVPixelBufferRef,
    ) -> OSStatus;
}

// ─── small CF helpers (lean on the `core-foundation` crate, already a dep) ─

/// Wrap a borrowed (+0) extern `CFStringRef` constant as a `CFString` without
/// taking ownership.
unsafe fn cfstr(raw: CFStringRefRaw) -> CFString {
    CFString::wrap_under_get_rule(raw as CfStringRef)
}

/// `{ key: value }` one-entry dictionary as a raw `CFDictionaryRef` we keep
/// alive via the returned `CFDictionary` guard.
fn dict1(key: CFString, value: CFType) -> CFDictionary<CFString, CFType> {
    CFDictionary::from_CFType_pairs(&[(key, value)])
}

/// 64RGBALE: 16-bit RGBA, host little-endian — same channel order as our
/// `FisheyePair` RGBA buffers (no BGR swap), zero byte-swizzle on Apple
/// Silicon. `kCVPixelFormatType_64RGBALE` fourcc.
const FMT_64RGBALE: OSType = u32::from_be_bytes(*b"l64r");

// ─── capability probe ──────────────────────────────────────────────────

/// Whether `VTTemporalNoiseFilter` is supported on this machine (macOS 26+ on
/// capable Apple Silicon). Cheap; the UI calls it once to gate the slider.
pub fn is_supported() -> bool {
    let Some(cls) = AnyClass::get(c"VTTemporalNoiseFilterConfiguration") else {
        return false;
    };
    let b: Bool = unsafe { msg_send![cls, isSupported] };
    b.as_bool()
}

// ─── format selection (prefer 10-bit full-range, like the Swift helper) ──

unsafe fn bits_per_component(fmt: OSType) -> i32 {
    let desc = CVPixelFormatDescriptionCreateWithPixelFormatType(std::ptr::null(), fmt);
    if desc.is_null() {
        return 0;
    }
    let key = cfstr(kCVPixelFormatBitsPerComponent);
    let v = CFDictionaryGetValue(desc, key.as_concrete_TypeRef() as *const c_void);
    let bpc = if v.is_null() {
        0
    } else {
        CFNumber::wrap_under_get_rule(v as _).to_i32().unwrap_or(0)
    };
    CFRelease(desc);
    bpc
}

/// Full-range lossless formats have `'f'`/`'F'` as the second fourcc byte
/// (e.g. `xf20`); video-range variants would force the transfer session to
/// squeeze full-range RGB into 64..940 and back, shifting tones.
fn is_full_range(fmt: OSType) -> bool {
    let b1 = ((fmt >> 16) & 0xff) as u8;
    b1 == b'f' || b1 == b'F'
}

/// Pick the source pixel format for the filter from its supported list.
/// Prefer 10-bit at the requested range, then any 10-bit, else the first.
/// `prefer_full_range`: the RGBA path feeds full-range RGB → full-range
/// lossless; the P010 path feeds video-range YCbCr → video-range lossless so
/// the transfer is a pure repack (no 64..940 ↔ 0..1023 scaling round-trip).
unsafe fn pick_source_format(cfg_cls: &AnyClass, prefer_full_range: bool) -> Result<OSType> {
    let arr: Retained<AnyObject> = msg_send![cfg_cls, supportedSourcePixelFormats];
    let n: usize = msg_send![&*arr, count];
    if n == 0 {
        return Err(Error::Ffmpeg("VT: empty supportedSourcePixelFormats".into()));
    }
    let mut fmts = Vec::with_capacity(n);
    for i in 0..n {
        let num: *mut AnyObject = msg_send![&*arr, objectAtIndex: i];
        let v: u32 = msg_send![num, unsignedIntValue];
        fmts.push(v);
    }
    // 1st pass: 10-bit at the requested range.
    for &f in &fmts {
        if bits_per_component(f) == 10 && is_full_range(f) == prefer_full_range {
            return Ok(f);
        }
    }
    // 2nd pass: any 10-bit.
    for &f in &fmts {
        if bits_per_component(f) == 10 {
            return Ok(f);
        }
    }
    Ok(fmts[0])
}

// ─── the denoiser ──────────────────────────────────────────────────────

/// A single-stream temporal denoiser. One per eye (we run two — left/right —
/// independently, mirroring the Python app's two `s0`/`s4` helper processes).
///
/// Feed frames with [`push`](Self::push) and collect the emitted (delayed)
/// denoised frames; call [`finish`](Self::finish) at EOF to drain the tail.
/// I/O is RGBA — 8-bit (`RGBA8`, 4 B/px) or 16-bit (`RGBA64LE`, 8 B/px) — set
/// by `bit_depth`; internally everything runs at 10-bit.
pub struct VtDenoiser {
    frame_cls: &'static AnyClass,
    params_cls: &'static AnyClass,
    nsarray_cls: &'static AnyClass,
    _config: Retained<AnyObject>,
    processor: Retained<AnyObject>,
    transfer_to: VTPixelTransferSessionRef,
    transfer_from: VTPixelTransferSessionRef,
    io_pb: CVPixelBufferRef,
    pool: CVPixelBufferPoolRef,
    prev_count: usize,
    next_count: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    strength: f32,
    ring: VecDeque<Retained<AnyObject>>,
    ring_pts: VecDeque<i64>,
    source_idx: usize,
    frame_counter: i64,
    first_done: bool,
    // Coarse perf accounting (logged at Drop when VR180_DENOISE_TIMING is set).
    t_cpu: std::cell::Cell<f64>,
    t_xfer: std::cell::Cell<f64>,
    t_proc: std::cell::Cell<f64>,
}

impl VtDenoiser {
    /// Build a denoiser for `width`×`height` RGBA frames at the given
    /// `bit_depth` (8 or 16) and `strength` (0.0–1.0). Errors if the filter
    /// isn't supported or the dimensions/format are rejected.
    pub fn new(width: u32, height: u32, bit_depth: u8, strength: f32) -> Result<Self> {
        if !is_supported() {
            return Err(Error::Ffmpeg("VTTemporalNoiseFilter unsupported".into()));
        }
        let w = width as usize;
        let h = height as usize;
        unsafe {
            let cfg_cls = AnyClass::get(c"VTTemporalNoiseFilterConfiguration")
                .ok_or_else(|| Error::Ffmpeg("VT: no config class".into()))?;
            let proc_cls = AnyClass::get(c"VTFrameProcessor")
                .ok_or_else(|| Error::Ffmpeg("VT: no processor class".into()))?;
            let frame_cls = AnyClass::get(c"VTFrameProcessorFrame")
                .ok_or_else(|| Error::Ffmpeg("VT: no frame class".into()))?;
            let params_cls = AnyClass::get(c"VTTemporalNoiseFilterParameters")
                .ok_or_else(|| Error::Ffmpeg("VT: no params class".into()))?;
            let nsarray_cls = AnyClass::get(c"NSArray")
                .ok_or_else(|| Error::Ffmpeg("VT: no NSArray class".into()))?;

            // RGBA in → prefer full-range lossless.
            let target_fmt = pick_source_format(cfg_cls, true)?;
            let bpc = bits_per_component(target_fmt);
            tracing::info!(
                "vt_denoise: {w}x{h} bit_depth={bit_depth} strength={strength:.2} \
                 lossless_fmt=0x{target_fmt:08x} ({bpc}-bit)"
            );

            // Configuration (nullable init → Option).
            let alloc: Allocated<AnyObject> = msg_send![cfg_cls, alloc];
            let config: Option<Retained<AnyObject>> = msg_send![
                alloc,
                initWithFrameWidth: w as isize,
                frameHeight: h as isize,
                sourcePixelFormat: target_fmt
            ];
            let config = config
                .ok_or_else(|| Error::Ffmpeg(format!("VT: config init failed {w}x{h}")))?;

            let prev_count: isize = msg_send![&*config, previousFrameCount];
            let next_count: isize = msg_send![&*config, nextFrameCount];
            let prev_count = prev_count.max(0) as usize;
            let next_count = next_count.max(0) as usize;
            tracing::info!("vt_denoise: prev={prev_count} next={next_count}");

            // Processor + session.
            let alloc: Allocated<AnyObject> = msg_send![proc_cls, alloc];
            let processor: Retained<AnyObject> = msg_send![alloc, init];
            let ok: Bool = msg_send![
                &*processor,
                startSessionWithConfiguration: &*config,
                error: null_mut::<*mut AnyObject>()
            ];
            if !ok.as_bool() {
                return Err(Error::Ffmpeg("VT: startSession failed".into()));
            }

            // Pixel-transfer sessions (RGBA ⇄ lossless YCbCr).
            let mut transfer_to: VTPixelTransferSessionRef = null_mut();
            let mut transfer_from: VTPixelTransferSessionRef = null_mut();
            VTPixelTransferSessionCreate(std::ptr::null(), &mut transfer_to);
            VTPixelTransferSessionCreate(std::ptr::null(), &mut transfer_from);
            if transfer_to.is_null() || transfer_from.is_null() {
                return Err(Error::Ffmpeg("VT: pixel transfer session create failed".into()));
            }

            // Reusable 64RGBALE I/O buffer (single, reused every frame).
            let empty = CFDictionary::<CFString, CFType>::from_CFType_pairs(&[]);
            let io_attrs = dict1(
                cfstr(kCVPixelBufferIOSurfacePropertiesKey),
                empty.as_CFType(),
            );
            let mut io_pb: CVPixelBufferRef = null_mut();
            let r = CVPixelBufferCreate(
                std::ptr::null(),
                w,
                h,
                FMT_64RGBALE,
                io_attrs.as_concrete_TypeRef() as CFDictionaryRef,
                &mut io_pb,
            );
            if r != 0 || io_pb.is_null() {
                return Err(Error::Ffmpeg(format!("VT: 64RGBALE buffer create failed ({r})")));
            }
            // Tag only the YCbCr matrix (709) so the transfer does a pure
            // matrix convert without a lossy EOTF/OETF gamma round-trip.
            CVBufferSetAttachment(
                io_pb,
                kCVImageBufferYCbCrMatrixKey,
                kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                KCV_ATTACHMENT_SHOULD_PROPAGATE,
            );

            // Recycling pool for the lossless source/dest buffers.
            let window = prev_count + 1 + next_count;
            let min_count = CFNumber::from((window + 4) as i32);
            let pool_attrs = dict1(
                cfstr(kCVPixelBufferPoolMinimumBufferCountKey),
                min_count.as_CFType(),
            );
            let buf_attrs: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[
                (cfstr(kCVPixelBufferWidthKey), CFNumber::from(w as i32).as_CFType()),
                (cfstr(kCVPixelBufferHeightKey), CFNumber::from(h as i32).as_CFType()),
                (
                    cfstr(kCVPixelBufferPixelFormatTypeKey),
                    CFNumber::from(target_fmt as i32).as_CFType(),
                ),
                (
                    cfstr(kCVPixelBufferIOSurfacePropertiesKey),
                    CFDictionary::<CFString, CFType>::from_CFType_pairs(&[]).as_CFType(),
                ),
            ]);
            let mut pool: CVPixelBufferPoolRef = null_mut();
            let r = CVPixelBufferPoolCreate(
                std::ptr::null(),
                pool_attrs.as_concrete_TypeRef() as CFDictionaryRef,
                buf_attrs.as_concrete_TypeRef() as CFDictionaryRef,
                &mut pool,
            );
            if r != 0 || pool.is_null() {
                return Err(Error::Ffmpeg(format!("VT: pool create failed ({r})")));
            }

            Ok(Self {
                frame_cls,
                params_cls,
                nsarray_cls,
                _config: config,
                processor,
                transfer_to,
                transfer_from,
                io_pb,
                pool,
                prev_count,
                next_count,
                width: w,
                height: h,
                bit_depth,
                strength,
                ring: VecDeque::new(),
                ring_pts: VecDeque::new(),
                source_idx: 0,
                frame_counter: 0,
                first_done: false,
                t_cpu: std::cell::Cell::new(0.0),
                t_xfer: std::cell::Cell::new(0.0),
                t_proc: std::cell::Cell::new(0.0),
            })
        }
    }

    /// Number of look-ahead frames the filter delays output by (so the caller
    /// can map output index → input index, and knows it's a 1-frame clip when
    /// `finish()` produced nothing).
    pub fn latency(&self) -> usize {
        self.next_count
    }

    fn out_frame_bytes(&self) -> usize {
        let bpp = if self.bit_depth >= 16 { 8 } else { 4 };
        self.width * self.height * bpp
    }

    /// Allocate a recycled lossless buffer from the pool (caller owns the +1).
    unsafe fn pool_buffer(&self) -> Result<CVPixelBufferRef> {
        let mut pb: CVPixelBufferRef = null_mut();
        let r = CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), self.pool, &mut pb);
        if r != 0 || pb.is_null() {
            return Err(Error::Ffmpeg(format!("VT: pool buffer alloc failed ({r})")));
        }
        CVBufferSetAttachment(
            pb,
            kCVImageBufferYCbCrMatrixKey,
            kCVImageBufferYCbCrMatrix_ITU_R_709_2,
            KCV_ATTACHMENT_SHOULD_PROPAGATE,
        );
        Ok(pb)
    }

    /// Wrap a pixel buffer in a `VTFrameProcessorFrame`; the frame retains the
    /// buffer, so we drop our own +1 and let the frame own it.
    unsafe fn wrap_frame(&self, pb: CVPixelBufferRef, pts_val: i64) -> Result<Retained<AnyObject>> {
        let pts = CMTime {
            value: pts_val,
            timescale: 30,
            flags: CMTIME_FLAG_VALID,
            epoch: 0,
        };
        let alloc: Allocated<AnyObject> = msg_send![self.frame_cls, alloc];
        let frame: Option<Retained<AnyObject>> =
            msg_send![alloc, initWithBuffer: pb as *mut CVBuffer, presentationTimeStamp: pts];
        let frame = frame.ok_or_else(|| Error::Ffmpeg("VT: frame wrap failed".into()))?;
        CFRelease(pb as CFTypeRefRaw);
        Ok(frame)
    }

    fn make_nsarray(&self, frames: &[&AnyObject]) -> Retained<AnyObject> {
        unsafe {
            if frames.is_empty() {
                msg_send![self.nsarray_cls, array]
            } else {
                let ptrs: Vec<*const AnyObject> =
                    frames.iter().map(|f| *f as *const AnyObject).collect();
                msg_send![self.nsarray_cls, arrayWithObjects: ptrs.as_ptr(), count: ptrs.len()]
            }
        }
    }

    /// RGBA frame → reusable 64RGBALE I/O buffer → transfer to a fresh
    /// lossless source buffer. Returns the lossless buffer (+1).
    unsafe fn build_source(&self, rgba: &[u8]) -> Result<CVPixelBufferRef> {
        let w = self.width;
        let h = self.height;
        let tc = std::time::Instant::now();
        CVPixelBufferLockBaseAddress(self.io_pb, 0);
        let base = CVPixelBufferGetBaseAddress(self.io_pb) as *mut u8;
        let stride = CVPixelBufferGetBytesPerRow(self.io_pb);
        if self.bit_depth >= 16 {
            // Source RGBA64LE is byte-identical to the 64RGBALE I/O buffer
            // (R,G,B,A u16 host-LE) → straight per-row memcpy instead of a
            // scalar per-pixel reorder. (Alpha rides along; the lossless
            // YCbCr transfer drops it and projection ignores it.)
            let row_bytes = w * 8;
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    rgba.as_ptr().add(row * row_bytes),
                    base.add(row * stride),
                    row_bytes,
                );
            }
        } else {
            for row in 0..h {
                let dst = base.add(row * stride) as *mut u16;
                for col in 0..w {
                    let si = (row * w + col) * 4;
                    *dst.add(col * 4) = rgba[si] as u16 * 257;
                    *dst.add(col * 4 + 1) = rgba[si + 1] as u16 * 257;
                    *dst.add(col * 4 + 2) = rgba[si + 2] as u16 * 257;
                    *dst.add(col * 4 + 3) = 65535;
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(self.io_pb, 0);
        self.t_cpu.set(self.t_cpu.get() + tc.elapsed().as_secs_f64() * 1e3);

        let pb = self.pool_buffer()?;
        let tx = std::time::Instant::now();
        let st = VTPixelTransferSessionTransferImage(self.transfer_to, self.io_pb, pb);
        self.t_xfer.set(self.t_xfer.get() + tx.elapsed().as_secs_f64() * 1e3);
        if st != 0 {
            CFRelease(pb as CFTypeRefRaw);
            return Err(Error::Ffmpeg(format!("VT: transfer-to-lossless failed ({st})")));
        }
        Ok(pb)
    }

    /// Denoised lossless buffer → 64RGBALE I/O buffer → RGBA out.
    unsafe fn read_dest(&self, dest_pb: CVPixelBufferRef, out: &mut [u8]) -> Result<()> {
        let tx = std::time::Instant::now();
        let st = VTPixelTransferSessionTransferImage(self.transfer_from, dest_pb, self.io_pb);
        self.t_xfer.set(self.t_xfer.get() + tx.elapsed().as_secs_f64() * 1e3);
        if st != 0 {
            return Err(Error::Ffmpeg(format!("VT: transfer-from-lossless failed ({st})")));
        }
        let w = self.width;
        let h = self.height;
        let tc = std::time::Instant::now();
        CVPixelBufferLockBaseAddress(self.io_pb, KCV_PIXEL_BUFFER_LOCK_READ_ONLY);
        let base = CVPixelBufferGetBaseAddress(self.io_pb) as *const u8;
        let stride = CVPixelBufferGetBytesPerRow(self.io_pb);
        if self.bit_depth >= 16 {
            // 64RGBALE → RGBA64LE: identical layout, per-row memcpy.
            let row_bytes = w * 8;
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    base.add(row * stride),
                    out.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
        } else {
            for row in 0..h {
                let src = base.add(row * stride) as *const u16;
                for col in 0..w {
                    let di = (row * w + col) * 4;
                    out[di] = (*src.add(col * 4) >> 8) as u8;
                    out[di + 1] = (*src.add(col * 4 + 1) >> 8) as u8;
                    out[di + 2] = (*src.add(col * 4 + 2) >> 8) as u8;
                    out[di + 3] = 255;
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(self.io_pb, KCV_PIXEL_BUFFER_LOCK_READ_ONLY);
        self.t_cpu.set(self.t_cpu.get() + tc.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// Feed one RGBA source frame; returns zero or more denoised frames that
    /// are now ready (output is delayed by `next_count` frames).
    pub fn push(&mut self, rgba: &[u8]) -> Result<Vec<Vec<u8>>> {
        let pts_val = self.frame_counter;
        let pb = unsafe { self.build_source(rgba)? };
        let frame = unsafe { self.wrap_frame(pb, pts_val)? };
        self.ring.push_back(frame);
        self.ring_pts.push_back(pts_val);
        self.frame_counter += 1;

        let mut out = Vec::new();
        while let Some(f) = self.emit_one(false)? {
            out.push(f);
        }
        Ok(out)
    }

    /// Drain the look-ahead tail at EOF.
    pub fn finish(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        while let Some(f) = self.emit_one(true)? {
            out.push(f);
        }
        Ok(out)
    }

    /// One step of the sliding-window filter. Returns `Ok(None)` when not
    /// enough look-ahead is buffered (and `!eof`), or the stream is drained.
    fn emit_one(&mut self, eof: bool) -> Result<Option<Vec<u8>>> {
        let src_pos = self.prev_count.min(self.source_idx);
        if src_pos >= self.ring.len() {
            return Ok(None);
        }
        let next_avail = self.ring.len() - (src_pos + 1);
        if !eof && next_avail < self.next_count {
            return Ok(None);
        }
        let next_take = next_avail.min(self.next_count);
        if src_pos == 0 && next_take == 0 {
            // No reference frame possible (single-frame clip) — caller falls
            // back to passthrough.
            return Ok(None);
        }

        let out = unsafe {
            let source_frame: &AnyObject = &self.ring[src_pos];
            let prev_refs: Vec<&AnyObject> =
                (0..src_pos).map(|i| &*self.ring[i] as &AnyObject).collect();
            let next_refs: Vec<&AnyObject> = ((src_pos + 1)..(src_pos + 1 + next_take))
                .map(|i| &*self.ring[i] as &AnyObject)
                .collect();
            let prev_arr = self.make_nsarray(&prev_refs);
            let next_arr = self.make_nsarray(&next_refs);

            let dest_pb = self.pool_buffer()?;
            let dest_frame = self.wrap_frame(dest_pb, self.ring_pts[src_pos])?;

            let disc = !self.first_done;
            let alloc: Allocated<AnyObject> = msg_send![self.params_cls, alloc];
            let params: Option<Retained<AnyObject>> = msg_send![
                alloc,
                initWithSourceFrame: source_frame,
                nextFrames: &*next_arr,
                previousFrames: &*prev_arr,
                destinationFrame: &*dest_frame,
                filterStrength: self.strength,
                // `hasDiscontinuity:` is CoreFoundation `Boolean` (unsigned
                // char, encoding 'C'), NOT ObjC `BOOL` — pass a u8.
                hasDiscontinuity: disc as u8
            ];
            let params =
                params.ok_or_else(|| Error::Ffmpeg("VT: params init failed".into()))?;

            let tp = std::time::Instant::now();
            let ok: Bool = msg_send![
                &*self.processor,
                processWithParameters: &*params,
                error: null_mut::<*mut AnyObject>()
            ];
            self.t_proc.set(self.t_proc.get() + tp.elapsed().as_secs_f64() * 1e3);
            if !ok.as_bool() {
                return Err(Error::Ffmpeg("VT: process failed".into()));
            }
            self.first_done = true;

            let mut buf = vec![0u8; self.out_frame_bytes()];
            self.read_dest(dest_pb, &mut buf)?;
            // dest_frame drops here → its pixel buffer returns to the pool.
            buf
        };

        // Advance the window (mirror of the Swift FILE-mode slide).
        self.source_idx += 1;
        if self.source_idx > self.prev_count {
            self.ring.pop_front();
            self.ring_pts.pop_front();
            self.source_idx -= 1;
        }
        Ok(Some(out))
    }
}

impl std::fmt::Debug for VtDenoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VtDenoiser")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bit_depth", &self.bit_depth)
            .field("strength", &self.strength)
            .field("prev_count", &self.prev_count)
            .field("next_count", &self.next_count)
            .field("buffered", &self.ring.len())
            .finish()
    }
}

impl Drop for VtDenoiser {
    fn drop(&mut self) {
        if std::env::var_os("VR180_DENOISE_TIMING").is_some() {
            let n = self.frame_counter.max(1) as f64;
            tracing::info!(
                "vt_denoise[{}x{}] per-frame avg: cpu_copy={:.1}ms xfer={:.1}ms process={:.1}ms (n={})",
                self.width,
                self.height,
                self.t_cpu.get() / n,
                self.t_xfer.get() / n,
                self.t_proc.get() / n,
                self.frame_counter,
            );
        }
        unsafe {
            let _: () = msg_send![&*self.processor, endSession];
            if !self.io_pb.is_null() {
                CFRelease(self.io_pb as CFTypeRefRaw);
            }
            if !self.pool.is_null() {
                CFRelease(self.pool as CFTypeRefRaw);
            }
            if !self.transfer_to.is_null() {
                CFRelease(self.transfer_to as CFTypeRefRaw);
            }
            if !self.transfer_from.is_null() {
                CFRelease(self.transfer_from as CFTypeRefRaw);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// P010 IOSurface denoiser — the zero-copy export path
// ═══════════════════════════════════════════════════════════════════════

use crate::interop_macos::{
    create_p010_encode_buffer_fmt, cvpixelbuffer_pixel_format, p010_format_for_source,
    EncodePixelBufferP010,
};

/// Like [`VtDenoiser`] but I/O is a P010 `CVPixelBuffer` (IOSurface-backed)
/// rather than an RGBA `Vec`, so the macOS zero-copy OSV export can denoise
/// **entirely on the GPU** — no CPU readback. Per frame: the VT-decoded P010
/// buffer is transferred to a lossless source buffer, run through the
/// temporal filter (sliding window), then the denoised lossless result is
/// transferred into a fresh P010 [`EncodePixelBufferP010`] whose IOSurface
/// planes the projection shader reads directly.
///
/// Everything stays YCbCr 709 video-range (the OSV decode format), so the
/// transfers are pure repacks with no RGB matrix or range round-trip. One
/// instance per eye. Output count/order preserved 1:1 (drained at `finish`).
pub struct P010Denoiser {
    frame_cls: &'static AnyClass,
    params_cls: &'static AnyClass,
    nsarray_cls: &'static AnyClass,
    _config: Retained<AnyObject>,
    processor: Retained<AnyObject>,
    transfer_to: VTPixelTransferSessionRef,
    transfer_from: VTPixelTransferSessionRef,
    pool: CVPixelBufferPoolRef,
    prev_count: usize,
    next_count: usize,
    width: u32,
    height: u32,
    strength: f32,
    ring: VecDeque<Retained<AnyObject>>,
    ring_pts: VecDeque<i64>,
    source_idx: usize,
    frame_counter: i64,
    first_done: bool,
    /// Output P010 fourcc, matched to the source's range (video vs full) so
    /// the lossless round-trip stays level-symmetric. Set on the first push.
    out_fmt: u32,
}

impl P010Denoiser {
    /// Build a P010 denoiser for `width`×`height` (the native fisheye eye
    /// dims, both even) at `strength` (0.0–1.0).
    pub fn new(width: u32, height: u32, strength: f32) -> Result<Self> {
        if !is_supported() {
            return Err(Error::Ffmpeg("VTTemporalNoiseFilter unsupported".into()));
        }
        let w = width as usize;
        let h = height as usize;
        unsafe {
            let cfg_cls = AnyClass::get(c"VTTemporalNoiseFilterConfiguration")
                .ok_or_else(|| Error::Ffmpeg("VT: no config class".into()))?;
            let proc_cls = AnyClass::get(c"VTFrameProcessor")
                .ok_or_else(|| Error::Ffmpeg("VT: no processor class".into()))?;
            let frame_cls = AnyClass::get(c"VTFrameProcessorFrame")
                .ok_or_else(|| Error::Ffmpeg("VT: no frame class".into()))?;
            let params_cls = AnyClass::get(c"VTTemporalNoiseFilterParameters")
                .ok_or_else(|| Error::Ffmpeg("VT: no params class".into()))?;
            let nsarray_cls = AnyClass::get(c"NSArray")
                .ok_or_else(|| Error::Ffmpeg("VT: no NSArray class".into()))?;

            // P010 source is video-range YCbCr → video-range lossless.
            let target_fmt = pick_source_format(cfg_cls, false)?;
            tracing::info!(
                "p010_denoise: {w}x{h} strength={strength:.2} lossless_fmt=0x{target_fmt:08x}"
            );

            let alloc: Allocated<AnyObject> = msg_send![cfg_cls, alloc];
            let config: Option<Retained<AnyObject>> = msg_send![
                alloc,
                initWithFrameWidth: w as isize,
                frameHeight: h as isize,
                sourcePixelFormat: target_fmt
            ];
            let config = config
                .ok_or_else(|| Error::Ffmpeg(format!("VT: P010 config init failed {w}x{h}")))?;
            let prev_count: isize = msg_send![&*config, previousFrameCount];
            let next_count: isize = msg_send![&*config, nextFrameCount];
            let prev_count = prev_count.max(0) as usize;
            let next_count = next_count.max(0) as usize;

            let alloc: Allocated<AnyObject> = msg_send![proc_cls, alloc];
            let processor: Retained<AnyObject> = msg_send![alloc, init];
            let ok: Bool = msg_send![
                &*processor,
                startSessionWithConfiguration: &*config,
                error: null_mut::<*mut AnyObject>()
            ];
            if !ok.as_bool() {
                return Err(Error::Ffmpeg("VT: P010 startSession failed".into()));
            }

            let mut transfer_to: VTPixelTransferSessionRef = null_mut();
            let mut transfer_from: VTPixelTransferSessionRef = null_mut();
            VTPixelTransferSessionCreate(std::ptr::null(), &mut transfer_to);
            VTPixelTransferSessionCreate(std::ptr::null(), &mut transfer_from);
            if transfer_to.is_null() || transfer_from.is_null() {
                return Err(Error::Ffmpeg("VT: P010 transfer session create failed".into()));
            }

            let window = prev_count + 1 + next_count;
            let min_count = CFNumber::from((window + 4) as i32);
            let pool_attrs = dict1(
                cfstr(kCVPixelBufferPoolMinimumBufferCountKey),
                min_count.as_CFType(),
            );
            let buf_attrs: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[
                (cfstr(kCVPixelBufferWidthKey), CFNumber::from(w as i32).as_CFType()),
                (cfstr(kCVPixelBufferHeightKey), CFNumber::from(h as i32).as_CFType()),
                (
                    cfstr(kCVPixelBufferPixelFormatTypeKey),
                    CFNumber::from(target_fmt as i32).as_CFType(),
                ),
                (
                    cfstr(kCVPixelBufferIOSurfacePropertiesKey),
                    CFDictionary::<CFString, CFType>::from_CFType_pairs(&[]).as_CFType(),
                ),
            ]);
            let mut pool: CVPixelBufferPoolRef = null_mut();
            let r = CVPixelBufferPoolCreate(
                std::ptr::null(),
                pool_attrs.as_concrete_TypeRef() as CFDictionaryRef,
                buf_attrs.as_concrete_TypeRef() as CFDictionaryRef,
                &mut pool,
            );
            if r != 0 || pool.is_null() {
                return Err(Error::Ffmpeg(format!("VT: P010 pool create failed ({r})")));
            }

            Ok(Self {
                frame_cls,
                params_cls,
                nsarray_cls,
                _config: config,
                processor,
                transfer_to,
                transfer_from,
                pool,
                prev_count,
                next_count,
                width,
                height,
                strength,
                ring: VecDeque::new(),
                ring_pts: VecDeque::new(),
                source_idx: 0,
                frame_counter: 0,
                first_done: false,
                out_fmt: 0,
            })
        }
    }

    pub fn latency(&self) -> usize {
        self.next_count
    }

    /// Allocate a recycled lossless buffer (caller owns the +1); tag 709 so
    /// the YCbCr↔YCbCr transfers stay matrix-consistent with the source.
    unsafe fn pool_buffer(&self) -> Result<CVPixelBufferRef> {
        let mut pb: CVPixelBufferRef = null_mut();
        let r = CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), self.pool, &mut pb);
        if r != 0 || pb.is_null() {
            return Err(Error::Ffmpeg(format!("VT: P010 pool buffer alloc failed ({r})")));
        }
        CVBufferSetAttachment(
            pb,
            kCVImageBufferYCbCrMatrixKey,
            kCVImageBufferYCbCrMatrix_ITU_R_709_2,
            KCV_ATTACHMENT_SHOULD_PROPAGATE,
        );
        Ok(pb)
    }

    /// Wrap a lossless pixel buffer as a `VTFrameProcessorFrame`; the frame
    /// retains the buffer, so we drop our own +1.
    unsafe fn wrap_frame(&self, pb: CVPixelBufferRef, pts_val: i64) -> Result<Retained<AnyObject>> {
        let pts = CMTime {
            value: pts_val,
            timescale: 30,
            flags: CMTIME_FLAG_VALID,
            epoch: 0,
        };
        let alloc: Allocated<AnyObject> = msg_send![self.frame_cls, alloc];
        let frame: Option<Retained<AnyObject>> =
            msg_send![alloc, initWithBuffer: pb as *mut CVBuffer, presentationTimeStamp: pts];
        let frame = frame.ok_or_else(|| Error::Ffmpeg("VT: P010 frame wrap failed".into()))?;
        CFRelease(pb as CFTypeRefRaw);
        Ok(frame)
    }

    fn make_nsarray(&self, frames: &[&AnyObject]) -> Retained<AnyObject> {
        unsafe {
            if frames.is_empty() {
                msg_send![self.nsarray_cls, array]
            } else {
                let ptrs: Vec<*const AnyObject> =
                    frames.iter().map(|f| *f as *const AnyObject).collect();
                msg_send![self.nsarray_cls, arrayWithObjects: ptrs.as_ptr(), count: ptrs.len()]
            }
        }
    }

    /// Transfer a decoded P010 buffer into a fresh lossless source buffer.
    unsafe fn to_lossless(&self, src_p010: CVPixelBufferRef) -> Result<CVPixelBufferRef> {
        let pb = self.pool_buffer()?;
        let st = VTPixelTransferSessionTransferImage(self.transfer_to, src_p010, pb);
        if st != 0 {
            CFRelease(pb as CFTypeRefRaw);
            return Err(Error::Ffmpeg(format!("VT: P010→lossless transfer failed ({st})")));
        }
        Ok(pb)
    }

    /// Feed one decoded P010 `CVPixelBuffer`; returns zero or more denoised
    /// P010 frames ready (delayed by `next_count`). The source buffer is
    /// consumed immediately (converted into the window) so the caller may
    /// drop it after this returns.
    pub fn push(
        &mut self,
        device: &wgpu::Device,
        src_p010: CVPixelBufferRef,
    ) -> Result<Vec<EncodePixelBufferP010>> {
        // Match the output P010 range to the source so the lossless
        // round-trip is level-symmetric (GoPro .360 is full-range `pc`, DJI
        // OSV is video-range `tv`). Constant across the clip; set each push.
        self.out_fmt = p010_format_for_source(cvpixelbuffer_pixel_format(src_p010));
        let pts_val = self.frame_counter;
        let lossless_src = unsafe { self.to_lossless(src_p010)? };
        let frame = unsafe { self.wrap_frame(lossless_src, pts_val)? };
        self.ring.push_back(frame);
        self.ring_pts.push_back(pts_val);
        self.frame_counter += 1;

        let mut out = Vec::new();
        while let Some(o) = self.emit_one(device, false)? {
            out.push(o);
        }
        Ok(out)
    }

    /// Drain the look-ahead tail at EOF.
    pub fn finish(&mut self, device: &wgpu::Device) -> Result<Vec<EncodePixelBufferP010>> {
        let mut out = Vec::new();
        while let Some(o) = self.emit_one(device, true)? {
            out.push(o);
        }
        Ok(out)
    }

    fn emit_one(
        &mut self,
        device: &wgpu::Device,
        eof: bool,
    ) -> Result<Option<EncodePixelBufferP010>> {
        let src_pos = self.prev_count.min(self.source_idx);
        if src_pos >= self.ring.len() {
            return Ok(None);
        }
        let next_avail = self.ring.len() - (src_pos + 1);
        if !eof && next_avail < self.next_count {
            return Ok(None);
        }
        let next_take = next_avail.min(self.next_count);
        if src_pos == 0 && next_take == 0 {
            return Ok(None);
        }

        let out = unsafe {
            let source_frame: &AnyObject = &self.ring[src_pos];
            let prev_refs: Vec<&AnyObject> =
                (0..src_pos).map(|i| &*self.ring[i] as &AnyObject).collect();
            let next_refs: Vec<&AnyObject> = ((src_pos + 1)..(src_pos + 1 + next_take))
                .map(|i| &*self.ring[i] as &AnyObject)
                .collect();
            let prev_arr = self.make_nsarray(&prev_refs);
            let next_arr = self.make_nsarray(&next_refs);

            let dest_lossless = self.pool_buffer()?;
            let dest_frame = self.wrap_frame(dest_lossless, self.ring_pts[src_pos])?;

            let disc = !self.first_done;
            let alloc: Allocated<AnyObject> = msg_send![self.params_cls, alloc];
            let params: Option<Retained<AnyObject>> = msg_send![
                alloc,
                initWithSourceFrame: source_frame,
                nextFrames: &*next_arr,
                previousFrames: &*prev_arr,
                destinationFrame: &*dest_frame,
                filterStrength: self.strength,
                hasDiscontinuity: disc as u8
            ];
            let params =
                params.ok_or_else(|| Error::Ffmpeg("VT: P010 params init failed".into()))?;
            let ok: Bool = msg_send![
                &*self.processor,
                processWithParameters: &*params,
                error: null_mut::<*mut AnyObject>()
            ];
            if !ok.as_bool() {
                return Err(Error::Ffmpeg("VT: P010 process failed".into()));
            }
            self.first_done = true;

            // Denoised lossless → fresh IOSurface-backed P010 buffer the
            // projection reads, in the source's range (`out_fmt`).
            // `dest_lossless` stays alive via `dest_frame` until the transfer
            // completes (TransferImage is synchronous).
            let epb = create_p010_encode_buffer_fmt(device, self.width, self.height, self.out_fmt)?;
            let st = VTPixelTransferSessionTransferImage(
                self.transfer_from,
                dest_lossless,
                epb.pixel_buffer.as_raw(),
            );
            if st != 0 {
                return Err(Error::Ffmpeg(format!("VT: lossless→P010 transfer failed ({st})")));
            }
            // dest_frame drops here → dest_lossless returns to the pool.
            epb
        };

        self.source_idx += 1;
        if self.source_idx > self.prev_count {
            self.ring.pop_front();
            self.ring_pts.pop_front();
            self.source_idx -= 1;
        }
        Ok(Some(out))
    }
}

impl std::fmt::Debug for P010Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("P010Denoiser")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("strength", &self.strength)
            .field("prev_count", &self.prev_count)
            .field("next_count", &self.next_count)
            .field("buffered", &self.ring.len())
            .finish()
    }
}

impl Drop for P010Denoiser {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![&*self.processor, endSession];
            if !self.pool.is_null() {
                CFRelease(self.pool as CFTypeRefRaw);
            }
            if !self.transfer_to.is_null() {
                CFRelease(self.transfer_to as CFTypeRefRaw);
            }
            if !self.transfer_from.is_null() {
                CFRelease(self.transfer_from as CFTypeRefRaw);
            }
        }
    }
}
