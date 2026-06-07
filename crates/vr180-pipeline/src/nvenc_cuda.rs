//! GPU-resident NVENC encode (Windows): keep the composited P010 frame in
//! VRAM and hand it to `hevc_nvenc` via CUDA, eliminating the CPU readback
//! that otherwise caps 8K export.
//!
//! Adapted from slr-studio-neo's `interop.rs` (proven at the 8K NVENC ceiling)
//! with one change forced by wgpu 29: that project shares an exportable Vulkan
//! *buffer* (`buffer_from_raw`), but wgpu-hal 29 dropped Vulkan `buffer_from_raw`
//! — so we share an exportable **linear image** via the surviving
//! `texture_from_raw` instead (validated in `examples/vk_cuda_smoke.rs`).
//!
//! Two components:
//! - [`CudaNvencEncoder`] — ffmpeg CUDA hwdevice + P010 hwframes pool +
//!   `hevc_nvenc`; per frame does an intra-VRAM `cuMemcpy2DAsync` (DtoD) from
//!   our shared planes into NVENC's CUDA frame on ffmpeg's CUstream, then
//!   `cuStreamSynchronize` + `send_frame`. Writes a bare video file; the
//!   caller muxes audio via the normal `finalize_with_audio`.
//! - [`SharedP010Frame`] — exportable linear R16 (Y) + Rg16 (UV) images,
//!   wrapped as wgpu textures (compose/copy destination) AND imported into
//!   CUDA (the DtoD source).

#![cfg(target_os = "windows")]

use crate::interop_windows::VulkanImportCtx;
use crate::{Error, Result};
use ash::vk;
use cudarc::driver::sys as cu_sys;
use ffmpeg_next as ffmpeg;
use std::ffi::CString;
use std::ptr;
use wgpu::hal::api::Vulkan;

// ── ffmpeg CUstream extraction (port from slr interop.rs) ──────────────────

/// Extract ffmpeg's `CUstream` from its `AV_HWDEVICE_TYPE_CUDA` device buffer.
/// `AVCUDADeviceContext` is `{ CUcontext cuda_ctx; CUstream stream; .. }`, so
/// the stream is the second pointer-sized field of `hwctx`.
unsafe fn cuda_stream_from_ffmpeg(av_buffer_ref: *mut ffmpeg::ffi::AVBufferRef) -> cu_sys::CUstream {
    if av_buffer_ref.is_null() {
        return ptr::null_mut();
    }
    let dev_ctx = (*av_buffer_ref).data as *mut ffmpeg::ffi::AVHWDeviceContext;
    if dev_ctx.is_null() {
        return ptr::null_mut();
    }
    let hwctx = (*dev_ctx).hwctx;
    if hwctx.is_null() {
        return ptr::null_mut();
    }
    let cu_ptrs = hwctx as *const cu_sys::CUcontext;
    let stream_ptr = cu_ptrs.add(1) as *const cu_sys::CUstream;
    *stream_ptr
}

/// Intra-VRAM 2D copy (device→device), async on `stream`.
unsafe fn dtod_2d_async(
    stream: cu_sys::CUstream,
    src: cu_sys::CUdeviceptr,
    src_pitch: usize,
    dst: cu_sys::CUdeviceptr,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<()> {
    let copy = cu_sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cu_sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: ptr::null(),
        srcDevice: src,
        srcArray: ptr::null_mut(),
        srcPitch: src_pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cu_sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: ptr::null_mut(),
        dstDevice: dst,
        dstArray: ptr::null_mut(),
        dstPitch: dst_pitch,
        WidthInBytes: width_bytes,
        Height: height,
    };
    let r = cu_sys::lib().cuMemcpy2DAsync_v2(&copy, stream);
    if r != cu_sys::CUresult::CUDA_SUCCESS {
        return Err(Error::Ffmpeg(format!("cuMemcpy2DAsync_v2 failed: {r:?}")));
    }
    Ok(())
}

// ── CudaNvencEncoder ───────────────────────────────────────────────────────

/// `hevc_nvenc` driven from CUDA hardware frames. The composited P010 stays in
/// VRAM end-to-end; this struct owns the ffmpeg CUDA hwdevice, the input frame
/// pool, the encoder, and the output muxer (video-only — caller muxes audio).
pub struct CudaNvencEncoder {
    octx: *mut ffmpeg::ffi::AVFormatContext,
    enc: *mut ffmpeg::ffi::AVCodecContext,
    hw_device_ref: *mut ffmpeg::ffi::AVBufferRef,
    hw_frames_ref: *mut ffmpeg::ffi::AVBufferRef,
    stream: cu_sys::CUstream,
    stream_index: i32,
    time_base: ffmpeg::ffi::AVRational,
    frame_count: i64,
    pub width: u32,
    pub height: u32,
}

// The raw pointers are only ever touched on the single export thread that owns
// the encoder; we never share it across threads.
unsafe impl Send for CudaNvencEncoder {}

impl CudaNvencEncoder {
    /// `bitrate_kbps` is the VBR target. `bit_depth` must be 10 (Main10 / P010).
    pub fn new(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps: f32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        crate::decode::init();
        unsafe {
            let (num, den) = approx_rational(fps);
            let time_base = ffmpeg::ffi::AVRational { num: den, den: num }; // 1/fps style

            // CUDA hw device (retains device-0 primary ctx — shared with cudarc).
            let mut hw_device_ref: *mut ffmpeg::ffi::AVBufferRef = ptr::null_mut();
            let dev_id = CString::new("0").unwrap();
            let r = ffmpeg::ffi::av_hwdevice_ctx_create(
                &mut hw_device_ref,
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
                dev_id.as_ptr(),
                ptr::null_mut(),
                0,
            );
            if r < 0 {
                return Err(Error::Ffmpeg(format!("av_hwdevice_ctx_create(CUDA): {r}")));
            }
            let stream = cuda_stream_from_ffmpeg(hw_device_ref);

            // P010 hw frames pool (depth 32 to absorb 8K per-frame variance).
            let hw_frames_ref = ffmpeg::ffi::av_hwframe_ctx_alloc(hw_device_ref);
            if hw_frames_ref.is_null() {
                return Err(Error::Ffmpeg("av_hwframe_ctx_alloc failed".into()));
            }
            {
                let fc = (*hw_frames_ref).data as *mut ffmpeg::ffi::AVHWFramesContext;
                (*fc).format = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
                (*fc).sw_format = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_P010LE;
                (*fc).width = width as i32;
                (*fc).height = height as i32;
                (*fc).initial_pool_size = 32;
            }
            let r = ffmpeg::ffi::av_hwframe_ctx_init(hw_frames_ref);
            if r < 0 {
                return Err(Error::Ffmpeg(format!("av_hwframe_ctx_init: {r}")));
            }

            // Encoder.
            let cname = CString::new("hevc_nvenc").unwrap();
            let codec = ffmpeg::ffi::avcodec_find_encoder_by_name(cname.as_ptr());
            if codec.is_null() {
                return Err(Error::Ffmpeg("hevc_nvenc not found".into()));
            }
            let enc = ffmpeg::ffi::avcodec_alloc_context3(codec);
            if enc.is_null() {
                return Err(Error::Ffmpeg("avcodec_alloc_context3 failed".into()));
            }
            (*enc).width = width as i32;
            (*enc).height = height as i32;
            (*enc).time_base = time_base;
            (*enc).framerate = ffmpeg::ffi::AVRational { num, den };
            (*enc).pix_fmt = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
            (*enc).bit_rate = (bitrate_kbps as i64) * 1000;
            (*enc).rc_max_rate = ((bitrate_kbps as i64) * 1000 * 3) / 2;
            (*enc).gop_size = 60;
            (*enc).codec_tag = u32::from_le_bytes(*b"hvc1");
            (*enc).hw_frames_ctx = ffmpeg::ffi::av_buffer_ref(hw_frames_ref);

            // Colorimetry tags → HEVC VUI + mp4 `colr` atom. We compose
            // VIDEO-RANGE (limited) Rec.709 YCbCr — the broadcast/distribution
            // standard (YouTube VR180, headsets) and consistent with the
            // readback + libx265 paths and the source. A compliant decoder
            // expands limited→full and recovers the same grade the (now
            // gamma-correct) preview shows. Primaries/transfer/matrix Rec.709.
            (*enc).color_range     = ffmpeg::ffi::AVColorRange::AVCOL_RANGE_MPEG;
            (*enc).color_primaries = ffmpeg::ffi::AVColorPrimaries::AVCOL_PRI_BT709;
            (*enc).color_trc       = ffmpeg::ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
            (*enc).colorspace      = ffmpeg::ffi::AVColorSpace::AVCOL_SPC_BT709;

            let set = |k: &str, v: &str| {
                let k = CString::new(k).unwrap();
                let v = CString::new(v).unwrap();
                ffmpeg::ffi::av_opt_set((*enc).priv_data, k.as_ptr(), v.as_ptr(), 0);
            };
            // p4 = NVENC's visually-transparent quality/speed sweet spot at 8K
            // (slr-studio-neo's measured default; p5+ are slower for marginal
            // quality, p1-p3 trade real quality for speed).
            set("preset", "p4");
            set("tune", "hq");
            set("rc", "vbr");
            set("profile", "main10");

            // Global header flag for mp4.
            let out_c = CString::new(path.to_string_lossy().to_string()).unwrap();
            let mut octx: *mut ffmpeg::ffi::AVFormatContext = ptr::null_mut();
            ffmpeg::ffi::avformat_alloc_output_context2(
                &mut octx, ptr::null_mut(), ptr::null(), out_c.as_ptr(),
            );
            if octx.is_null() {
                return Err(Error::Ffmpeg("avformat_alloc_output_context2 failed".into()));
            }
            if (*(*octx).oformat).flags & ffmpeg::ffi::AVFMT_GLOBALHEADER as i32 != 0 {
                (*enc).flags |= ffmpeg::ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }

            let r = ffmpeg::ffi::avcodec_open2(enc, codec, ptr::null_mut());
            if r < 0 {
                let mut buf = [0i8; 256];
                ffmpeg::ffi::av_strerror(r, buf.as_mut_ptr(), buf.len());
                let msg = std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
                return Err(Error::Ffmpeg(format!("avcodec_open2(hevc_nvenc CUDA): {r} ({msg})")));
            }

            let st = ffmpeg::ffi::avformat_new_stream(octx, ptr::null());
            ffmpeg::ffi::avcodec_parameters_from_context((*st).codecpar, enc);
            (*st).time_base = time_base;
            let stream_index = (*st).index;

            if ffmpeg::ffi::avio_open(&mut (*octx).pb, out_c.as_ptr(), ffmpeg::ffi::AVIO_FLAG_WRITE) < 0 {
                return Err(Error::Ffmpeg("avio_open failed".into()));
            }
            if ffmpeg::ffi::avformat_write_header(octx, ptr::null_mut()) < 0 {
                return Err(Error::Ffmpeg("avformat_write_header failed".into()));
            }

            Ok(Self {
                octx, enc, hw_device_ref, hw_frames_ref, stream, stream_index,
                time_base, frame_count: 0, width, height,
            })
        }
    }

    /// ffmpeg's CUstream — the caller uses it (or the encoder uses it
    /// internally) so the DtoD and NVENC reads are ordered.
    pub fn cuda_stream(&self) -> cu_sys::CUstream {
        self.stream
    }

    /// Encode one frame from CUDA-resident P010 planes. The caller must have
    /// ensured the GPU producer (wgpu compose/copy) finished writing the
    /// source planes before calling. All copies are intra-VRAM (no PCIe).
    ///
    /// # Safety
    /// `src_y`/`src_uv` are valid CUdeviceptrs in the device-0 primary context
    /// (current on this thread) for the given pitches/dims.
    pub unsafe fn encode_cuda_planes(
        &mut self,
        src_y: cu_sys::CUdeviceptr,
        src_y_pitch: usize,
        src_uv: cu_sys::CUdeviceptr,
        src_uv_pitch: usize,
    ) -> Result<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        let row_bytes = w * 2; // 16-bit luma / interleaved chroma row

        let mut frame = ffmpeg::ffi::av_frame_alloc();
        let r = ffmpeg::ffi::av_hwframe_get_buffer(self.hw_frames_ref, frame, 0);
        if r < 0 {
            ffmpeg::ffi::av_frame_free(&mut frame);
            return Err(Error::Ffmpeg(format!("av_hwframe_get_buffer: {r}")));
        }
        let dst_y = (*frame).data[0] as cu_sys::CUdeviceptr;
        let dst_uv = (*frame).data[1] as cu_sys::CUdeviceptr;
        let dst_y_pitch = (*frame).linesize[0] as usize;
        let dst_uv_pitch = (*frame).linesize[1] as usize;

        dtod_2d_async(self.stream, src_y, src_y_pitch, dst_y, dst_y_pitch, row_bytes, h)?;
        dtod_2d_async(self.stream, src_uv, src_uv_pitch, dst_uv, dst_uv_pitch, row_bytes, h / 2)?;
        // Ensure the DtoD finished before NVENC reads the frame.
        let r = cu_sys::lib().cuStreamSynchronize(self.stream);
        if r != cu_sys::CUresult::CUDA_SUCCESS {
            ffmpeg::ffi::av_frame_free(&mut frame);
            return Err(Error::Ffmpeg(format!("cuStreamSynchronize: {r:?}")));
        }

        (*frame).pts = self.frame_count;
        self.frame_count += 1;
        let r = ffmpeg::ffi::avcodec_send_frame(self.enc, frame);
        ffmpeg::ffi::av_frame_free(&mut frame);
        if r < 0 {
            return Err(Error::Ffmpeg(format!("send_frame: {r}")));
        }
        self.drain(false)
    }

    unsafe fn drain(&mut self, flush: bool) -> Result<()> {
        if flush {
            ffmpeg::ffi::avcodec_send_frame(self.enc, ptr::null());
        }
        let mut pkt = ffmpeg::ffi::av_packet_alloc();
        loop {
            let r = ffmpeg::ffi::avcodec_receive_packet(self.enc, pkt);
            if r < 0 {
                break; // EAGAIN / EOF / error → done for now
            }
            (*pkt).stream_index = self.stream_index;
            let st_tb = (**(*self.octx).streams.add(self.stream_index as usize)).time_base;
            ffmpeg::ffi::av_packet_rescale_ts(pkt, self.time_base, st_tb);
            ffmpeg::ffi::av_interleaved_write_frame(self.octx, pkt);
            ffmpeg::ffi::av_packet_unref(pkt);
        }
        ffmpeg::ffi::av_packet_free(&mut pkt);
        Ok(())
    }

    /// Flush the encoder and write the trailer. Leaves a complete video file.
    pub fn finish(&mut self) -> Result<()> {
        unsafe {
            self.drain(true)?;
            ffmpeg::ffi::av_write_trailer(self.octx);
            ffmpeg::ffi::avio_closep(&mut (*self.octx).pb);
        }
        Ok(())
    }
}

impl Drop for CudaNvencEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.enc.is_null() {
                ffmpeg::ffi::avcodec_free_context(&mut self.enc);
            }
            if !self.hw_frames_ref.is_null() {
                ffmpeg::ffi::av_buffer_unref(&mut self.hw_frames_ref);
            }
            if !self.hw_device_ref.is_null() {
                ffmpeg::ffi::av_buffer_unref(&mut self.hw_device_ref);
            }
            if !self.octx.is_null() {
                ffmpeg::ffi::avformat_free_context(self.octx);
                self.octx = ptr::null_mut();
            }
        }
    }
}

fn approx_rational(fps: f32) -> (i32, i32) {
    // Match the precision the rest of the pipeline uses for common rates.
    let common = [
        (24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1),
        (50, 1), (60000, 1001), (60, 1),
    ];
    let mut best = (30, 1);
    let mut best_err = f32::MAX;
    for &(n, d) in &common {
        let e = (fps - n as f32 / d as f32).abs();
        if e < best_err {
            best_err = e;
            best = (n, d);
        }
    }
    if best_err > 0.01 {
        ((fps * 1000.0).round() as i32, 1000)
    } else {
        best
    }
}

// ── SharedP010Frame ────────────────────────────────────────────────────────

struct SharedPlane {
    #[allow(dead_code)]
    tex: wgpu::Texture, // wgpu view of the exportable linear image (copy dst)
    devptr: cu_sys::CUdeviceptr,
    extmem: cu_sys::CUexternalMemory,
    row_pitch: usize,
    offset: u64,
}

/// One persistent P010 frame shared between wgpu (compose/copy destination)
/// and CUDA (the DtoD source for NVENC). Y is a linear R16 image at full SBS
/// res; UV is a linear Rg16 image at half res. Created once, reused per frame.
pub struct SharedP010Frame {
    y: SharedPlane,
    uv: SharedPlane,
    pub width: u32,
    pub height: u32,
}

unsafe impl Send for SharedP010Frame {}

impl SharedP010Frame {
    /// `width`/`height` are the SBS (full) frame dims.
    pub fn new(
        ctx: &VulkanImportCtx,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let y = make_shared_plane(ctx, device, width, height, wgpu::TextureFormat::R16Unorm, vk::Format::R16_UNORM)?;
        let uv = make_shared_plane(ctx, device, width / 2, height / 2, wgpu::TextureFormat::Rg16Unorm, vk::Format::R16G16_UNORM)?;
        Ok(Self { y, uv, width, height })
    }

    /// wgpu textures for `copy_texture_to_texture` destinations.
    pub fn y_texture(&self) -> &wgpu::Texture { &self.y.tex }
    pub fn uv_texture(&self) -> &wgpu::Texture { &self.uv.tex }

    /// CUDA source pointers + row pitches for the DtoD into NVENC's frame.
    pub fn y_cuda(&self) -> (cu_sys::CUdeviceptr, usize) {
        (self.y.devptr + self.y.offset, self.y.row_pitch)
    }
    pub fn uv_cuda(&self) -> (cu_sys::CUdeviceptr, usize) {
        (self.uv.devptr + self.uv.offset, self.uv.row_pitch)
    }
}

impl Drop for SharedP010Frame {
    fn drop(&mut self) {
        unsafe {
            // Free CUDA mapped buffers + external memory (before the wgpu
            // textures free the underlying Vulkan image/memory on their drop).
            let _ = cu_sys::lib().cuMemFree_v2(self.y.devptr);
            let _ = cu_sys::lib().cuDestroyExternalMemory(self.y.extmem);
            let _ = cu_sys::lib().cuMemFree_v2(self.uv.devptr);
            let _ = cu_sys::lib().cuDestroyExternalMemory(self.uv.extmem);
        }
    }
}

fn make_shared_plane(
    ctx: &VulkanImportCtx,
    device: &wgpu::Device,
    w: u32,
    h: u32,
    wgpu_fmt: wgpu::TextureFormat,
    vk_fmt: vk::Format,
) -> Result<SharedPlane> {
    let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
    let (image, memory, mem_size, row_pitch, offset) = unsafe {
        let mut ext_img = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let img_ci = vk::ImageCreateInfo::default()
            .push_next(&mut ext_img)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_fmt)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = ctx.device.create_image(&img_ci, None)
            .map_err(|e| Error::Wgpu(format!("create linear image: {e}")))?;
        let mem_req = ctx.device.get_image_memory_requirements(image);
        let mem_props = ctx.instance.get_physical_device_memory_properties(ctx.physical_device);
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                (mem_req.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| Error::Wgpu("no DEVICE_LOCAL memory type".into()))?;
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index)
            .push_next(&mut export_info)
            .push_next(&mut dedicated);
        let memory = ctx.device.allocate_memory(&alloc, None)
            .map_err(|e| Error::Wgpu(format!("allocate exportable memory: {e}")))?;
        ctx.device.bind_image_memory(image, memory, 0)
            .map_err(|e| Error::Wgpu(format!("bind_image_memory: {e}")))?;
        let sub = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, array_layer: 0,
        };
        let sl = ctx.device.get_image_subresource_layout(image, sub);
        (image, memory, mem_req.size, sl.row_pitch as usize, sl.offset)
    };

    // Wrap as a wgpu texture (Dedicated → wgpu frees image+memory on drop).
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let tex = unsafe {
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("shared-p010-linear"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_fmt,
            usage: wgpu::TextureUses::COPY_DST | wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let hal_tex = {
            let hal_device = device.as_hal::<Vulkan>().expect("wgpu not Vulkan");
            hal_device.texture_from_raw(image, &hal_desc, None, wgpu::hal::vulkan::TextureMemory::Dedicated(memory))
        };
        device.create_texture_from_hal::<Vulkan>(
            hal_tex,
            &wgpu::TextureDescriptor {
                label: Some("shared-p010-linear"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_fmt,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    };

    // Export the memory + import into CUDA.
    let (devptr, extmem) = unsafe {
        let info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(handle_type);
        let win32 = ctx.ext_mem_win32.get_memory_win32_handle(&info)
            .map_err(|e| Error::Wgpu(format!("get_memory_win32_handle: {e}")))?;
        let extmem = cudarc::driver::result::external_memory::import_external_memory_opaque_win32(
            win32 as std::os::windows::io::RawHandle, mem_size,
        ).map_err(|e| Error::Ffmpeg(format!("cuImportExternalMemory: {e:?}")))?;
        let devptr = cudarc::driver::result::external_memory::get_mapped_buffer(extmem, 0, mem_size)
            .map_err(|e| Error::Ffmpeg(format!("cuExternalMemoryGetMappedBuffer: {e:?}")))?;
        // CUDA holds its own ref now; close our exported handle.
        let _ = windows::Win32::Foundation::CloseHandle(windows::Win32::Foundation::HANDLE(win32 as _));
        (devptr, extmem)
    };

    Ok(SharedPlane { tex, devptr, extmem, row_pitch, offset })
}
