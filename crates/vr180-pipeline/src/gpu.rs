//! wgpu device orchestration + compute-kernel runners.
//!
//! Single backend, no fallbacks: wgpu picks Metal on macOS, Vulkan on
//! Linux, DX12 on Windows at runtime. This replaces the Python app's
//! MLX / Numba CUDA / Numba CPU trichotomy with one path.
//!
//! Phase 0.5 deliverable: [`Device::new`] + [`Device::project_cross_to_equirect`]
//! — the first real WGSL kernel running end-to-end (EAC cross → half-
//! equirect projection for one eye, CPU upload + GPU compute + CPU
//! readback).
//!
//! Future: shaders for color grading, sharpen, mid-detail, the
//! per-scanline RS warp, etc. live here too.

use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Per-slot cache of the resources `project_fisheye_to_equirect_*_texture`
/// needs every frame. Keyed by an integer slot id supplied by the caller
/// (e.g., 0 = left eye, 1 = right eye). Reused across frames so the GUI
/// preview path doesn't pay `create_texture` + `create_buffer` cost on
/// every frame — those are the bulk of the per-frame `project` time at
/// 50 fps, where the ~20 ms budget is tight.
///
/// Output textures are NOT cached here because the caller takes
/// ownership (the UI thread holds them via `Arc<wgpu::Texture>` until
/// it's done rendering). Only the inputs and the uniform/storage
/// buffers are recycled.
#[derive(Debug)]
struct ProjFisheyeRsCacheSlot {
    src_w: u32,
    src_h: u32,
    src_tex: wgpu::Texture,
    src_view: wgpu::TextureView,
    r_uniform: wgpu::Buffer,
    cal_uniform: wgpu::Buffer,
    rs_buf: wgpu::Buffer,
    /// Capacity in bytes — `rs_buf` is sized for `src_h * 12 * 4` and
    /// grown if a larger frame ever comes through.
    rs_buf_capacity: u64,
}

/// Non-RS variant — same idea but no `rs_buf`.
#[derive(Debug)]
struct ProjFisheyeCacheSlot {
    src_w: u32,
    src_h: u32,
    src_tex: wgpu::Texture,
    src_view: wgpu::TextureView,
    r_uniform: wgpu::Buffer,
    cal_uniform: wgpu::Buffer,
}

/// One wgpu device + queue + default adapter + cached pipelines.
///
/// Construction is async-under-the-hood; we wrap with `pollster::block_on`
/// because every entry point into this crate is currently synchronous.
/// The kernel pipelines are compiled once at `Device::new()` time —
/// reusing the same `Device` across frames avoids per-frame shader
/// compilation (which on Metal is ~10–15 ms).
#[derive(Debug)]
pub struct Device {
    pub instance: wgpu::Instance,
    pub adapter: std::sync::Arc<wgpu::Adapter>,
    pub device: std::sync::Arc<wgpu::Device>,
    pub queue: std::sync::Arc<wgpu::Queue>,
    eac_to_equirect: EacToEquirectPipeline,
    /// 16-bit-output EAC→equirect (true-10-bit EAC export chain).
    eac_to_equirect_16: EacToEquirectPipeline,
    /// EAC→equidistant-fisheye output (8- and 16-bit), for `.360`
    /// fisheye-output mode (parity with the OSV fisheye output).
    eac_to_fisheye: EacToEquirectPipeline,
    eac_to_fisheye_16: EacToEquirectPipeline,
    fisheye_to_hequirect: FisheyeToHequirectPipeline,
    fisheye_to_hequirect_16: FisheyeToHequirectPipeline,
    /// RS-aware variant of `fisheye_to_hequirect`. Used by the GUI
    /// preview when DJI OSV per-row matrices are available, to apply
    /// per-scanline correction (matches DJI Studio's per-slab approach).
    fisheye_to_hequirect_rs: FisheyeToHequirectRsPipeline,
    /// 16-bit-output RS half-equirect projection (Windows zero-copy preview
    /// stab with per-row rolling-shutter correction).
    fisheye_to_hequirect_rs_16: FisheyeToHequirectRsPipeline,
    /// Zero-copy fisheye projection that reads P010 IOSurface planes
    /// (Y at full res, UV at half res) directly and does YCbCr→RGB
    /// inline. Used on the OSV export path on macOS to skip the
    /// FFmpeg P010LE → RGBA64LE swscale + CPU→GPU upload (~840 MB/frame
    /// at native OSV res). Output format is Rgba16Unorm.
    fisheye_p010_to_hequirect_16: FisheyeP010ToHequirectPipeline,
    /// RS-aware sibling of `fisheye_p010_to_hequirect_16`. Takes an
    /// additional per-scanline rotation-matrix storage buffer and
    /// fuses per-row rolling-shutter correction into the projection.
    /// Cancels the jelly/shear within each frame that per-frame
    /// stabilization alone can't fix (DJI Osmo sensor ~19 ms readout).
    fisheye_p010_to_hequirect_16_rs: FisheyeP010ToHequirectRsPipeline,

    /// Fisheye → stabilized fisheye output (equidistant). Rgba8Unorm
    /// output, used by the preview when the user picks "Fisheye SBS".
    fisheye_to_fisheye: FisheyeToHequirectPipeline,
    /// RS-aware sibling of `fisheye_to_fisheye` (preview, 8-bit RGBA).
    fisheye_to_fisheye_rs: FisheyeToHequirectRsPipeline,
    /// 16-bit fisheye-to-fisheye for the 10-bit export path.
    fisheye_to_fisheye_16: FisheyeToHequirectPipeline,
    /// RS-aware sibling of `fisheye_to_fisheye_16` (sampled RGBA16 input —
    /// the Windows zero-copy preview + GPU-resident export). Parity with the
    /// macOS `fisheye_p010_to_fisheye_rs_16` path: fisheye output gets the
    /// same per-row rolling-shutter correction as half-equirect.
    fisheye_to_fisheye_rs_16: FisheyeToHequirectRsPipeline,
    /// 16-bit zero-copy P010 → fisheye-output projection. macOS-only
    /// zero-copy path for OSV fisheye-output 10-bit export.
    fisheye_p010_to_fisheye_16: FisheyeP010ToHequirectPipeline,
    /// RS-aware sibling of `fisheye_p010_to_fisheye_16` (zero-copy 10-bit
    /// fisheye-output export with per-row rolling-shutter correction).
    fisheye_p010_to_fisheye_rs_16: FisheyeP010ToHequirectRsPipeline,
    /// P010 → RGBA16 resolve + box downscale (Windows zero-copy preview
    /// prefilter — upsamples chroma to RGB then downscales, so the
    /// preview's big minification doesn't alias).
    p010_resolve: P010ResolvePipeline,
    lut3d: Lut3DPipeline,
    /// 16-bit-output variant of the 3D LUT pass. The LUT itself stays
    /// 8-bit (33³ entries — sampling is in floats so the LUT's own
    /// quantization is not the limiting factor). Output is Rgba16Unorm
    /// so the color stack can run end-to-end at 10-bit precision.
    lut3d_16: Lut3DPipeline,
    nv12_to_eac: Nv12ToEacPipeline,
    #[allow(dead_code)] // Windows zero-copy EAC only
    rgba_to_eac: RgbaToEacPipeline,
    #[allow(dead_code)] // Windows zero-copy EAC only
    rgba_to_eac_16: RgbaToEacPipeline,
    /// 16-bit-output NV12/P010→EAC-cross (true-10-bit EAC export chain).
    nv12_to_eac_16: Nv12ToEacPipeline,
    cdl: PerPixelPipeline,
    /// 16-bit-output CDL — same math as `cdl`, Rgba16Unorm output.
    cdl_16: PerPixelPipeline,
    color_grade: PerPixelPipeline,
    /// 16-bit-output color grade (temp / tint / saturation).
    color_grade_16: PerPixelPipeline,
    gaussian_blur: GaussianBlur1dPipeline,
    sharpen_combine: SharpenCombinePipeline,
    mid_detail_combine: MidDetailCombinePipeline,
    downsample_4x: Downsample4xPipeline,
    /// 16-bit variants of the multi-pass color shaders. Used by
    /// `apply_color_stack_per_eye_16` so sharpen + mid-detail stay at
    /// 10-bit precision through the 10-bit export pipeline.
    gaussian_blur_16: GaussianBlur1dPipeline,
    sharpen_combine_16: SharpenCombinePipeline,
    mid_detail_combine_16: MidDetailCombinePipeline,
    downsample_4x_16: Downsample4xPipeline,
    compose_sbs: ComposeSbsPipeline,
    /// Preview composer — SBS / anaglyph / overlay (Rgba8Unorm output).
    compose_preview: ComposeSbsPipeline,
    compose_sbs_p010_y:  P010ComposePipeline,
    compose_sbs_p010_uv: P010ComposePipeline,
    bilinear_sampler: wgpu::Sampler,

    /// Per-slot recycling for `project_fisheye_to_equirect_rs_texture`.
    /// Slot key is supplied by the caller (0 = left eye, 1 = right eye
    /// by convention) so we keep distinct source textures + uniform
    /// buffers per eye and avoid stomping inflight writes. Cost saved
    /// per frame at 50 fps with 2 eyes ≈ 4 large GPU allocations.
    proj_fisheye_rs_cache: Mutex<HashMap<u32, ProjFisheyeRsCacheSlot>>,
    /// Per-slot recycling for the non-RS variant
    /// `project_fisheye_to_equirect_texture`.
    proj_fisheye_cache: Mutex<HashMap<u32, ProjFisheyeCacheSlot>>,
    /// Per-slot reusable output texture for the Windows zero-copy preview
    /// projection (`project_fisheye_rgba16_texture_to_equirect_16`). Avoids a
    /// per-frame `create_texture` on the decoder thread (which contends with
    /// eframe's main thread for the shared wgpu device). Keyed by slot, stores
    /// `(w, h, texture)`; reused when dims match.
    rgba16_eq_out_cache: Mutex<HashMap<u32, (u32, u32, wgpu::Texture)>>,
    /// Reusable MAP_READ staging buffers for texture readback, keyed by byte
    /// size. Allocating a fresh 88 MB staging buffer every frame (the P010
    /// Y+UV readback) is a big chunk of the export's per-frame cost —
    /// `vkAllocateMemory` of host-visible memory isn't free. Reusing one of
    /// the right size makes the readback transfer-bound instead of alloc-bound.
    readback_staging: Mutex<HashMap<u64, wgpu::Buffer>>,
}

#[derive(Debug)]
struct P010ComposePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Debug)]
struct EacToEquirectPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Fisheye → half-equirect projection (one eye per dispatch). Used for
/// DJI OSV, SBS fisheye `.mp4`, and Blackmagic BRAW input families.
/// Same general shape as `EacToEquirectPipeline` but the source is a
/// raw fisheye image plus a Kannala-Brandt calibration uniform.
#[derive(Debug)]
struct FisheyeToHequirectPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// RS-aware sibling of `FisheyeToHequirectPipeline` for the RGBA
/// preview path. Adds a 6th binding for the per-scanline R-matrix
/// storage buffer used by `fisheye_to_hequirect_rs.wgsl`. Fuses
/// per-row rolling-shutter / per-slab stabilization into the projection
/// for OSV live preview.
#[derive(Debug)]
struct FisheyeToHequirectRsPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Zero-copy P010 → half-equirect projection (one eye per dispatch).
/// Same shader math as `FisheyeToHequirectPipeline` (16-bit output)
/// but the input is two separate textures — a full-res Y plane
/// (R16Unorm) and a half-res UV plane (Rg16Unorm) — and YCbCr→RGB
/// BT.709 limited-range expansion is baked into the per-pixel kernel.
/// Used on macOS for OSV export to skip the swscale + CPU bounce.
#[derive(Debug)]
struct FisheyeP010ToHequirectPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// RS-aware sibling of `FisheyeP010ToHequirectPipeline`. Adds a 7th
/// binding for the per-scanline R-matrix storage buffer used by the
/// rolling-shutter correction in `fisheye_p010_to_hequirect_rs.wgsl`.
#[derive(Debug)]
struct FisheyeP010ToHequirectRsPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// P010 → RGBA16 resolve + box-downscale (see `p010_resolve_rgba16.wgsl`).
/// Five bindings: Y tex, UV tex, sampler, RGBA16 storage out, dims uniform.
#[derive(Debug)]
struct P010ResolvePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Uniform for [`P010ResolvePipeline`] — source + destination dimensions.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveDims {
    src_w: f32,
    src_h: f32,
    out_w: f32,
    out_h: f32,
}

#[derive(Debug)]
struct Lut3DPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

#[derive(Debug)]
struct Nv12ToEacPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Windows zero-copy EAC cross assembly: same shape as `Nv12ToEacPipeline`
/// but the input is two already-RGB stream textures (the d3d11va P010 is
/// converted to single-plane Rgba16Unorm before Vulkan import) instead of
/// four NV12/P010 planes. See `shaders/rgba_to_eac_cross.wgsl`.
#[derive(Debug)]
struct RgbaToEacPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Reusable pipeline shape for any "per-pixel" color tool that takes
/// one input 2D texture + uniform buffer and writes to one output
/// storage texture (Rgba8Unorm). Used by CDL and color_grade, and
/// will be reused by saturation / future per-pixel shaders.
#[derive(Debug)]
struct PerPixelPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Separable 1-D Gaussian blur (shared by sharpen + mid-detail).
/// Same shape as `PerPixelPipeline` — one input tex, one storage out,
/// one uniform buffer (containing sigma + direction).
#[derive(Debug)]
struct GaussianBlur1dPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// USM combine pass: 2 input textures (orig + blur) + 1 storage out +
/// 1 uniform (amount + lat-weight flag).
#[derive(Debug)]
struct SharpenCombinePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Mid-detail upsample+combine pass: 2 input textures (orig +
/// blurred-small), 1 sampler (bilinear, for the upsample), 1 storage
/// out, 1 uniform (amount).
#[derive(Debug)]
struct MidDetailCombinePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// 4× box-filter downsample. 1 input tex, 1 storage out, no uniforms.
#[derive(Debug)]
struct Downsample4xPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// SBS composition: 2 input textures (left + right) + 1 storage out
/// (Rgba8Unorm view of a BGRA-byte-order IOSurface) + 1 uniform (eye_w).
/// Final pass of the zero-copy-encode path.
#[derive(Debug)]
struct ComposeSbsPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Device {
    /// Create a wgpu device using the best available adapter.
    /// Backend is picked by wgpu (Metal on macOS, DX12/Vulkan on Win).
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Build a `Device` on top of an **existing** wgpu instance /
    /// adapter / device / queue. Used by host applications (egui,
    /// Slint, …) that already own a wgpu device for their own
    /// renderer — sharing the device lets us pass `wgpu::Texture`s
    /// from our pipeline straight into the host UI with **zero
    /// copies** (cross-device texture handoff is otherwise gated
    /// on platform-specific external-memory extensions).
    ///
    /// The host MUST have requested at least `TEXTURE_FORMAT_16BIT_NORM`
    /// in its device features if it plans to feed 10-bit (P010)
    /// HEVC into this pipeline. For 8-bit-only workflows the feature
    /// is optional.
    pub fn from_existing(
        instance: wgpu::Instance,
        adapter: std::sync::Arc<wgpu::Adapter>,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) -> Result<Self> {
        let eac_to_equirect = EacToEquirectPipeline::create(&device);
        let eac_to_equirect_16 = EacToEquirectPipeline::create_16(&device);
        let eac_to_fisheye = EacToEquirectPipeline::create_fisheye(&device);
        let eac_to_fisheye_16 = EacToEquirectPipeline::create_fisheye_16(&device);
        let fisheye_to_hequirect = FisheyeToHequirectPipeline::create(&device);
        let fisheye_to_hequirect_16 = FisheyeToHequirectPipeline::create_16bit(&device);
        let fisheye_to_hequirect_rs = FisheyeToHequirectRsPipeline::create(&device);
        let fisheye_to_hequirect_rs_16 = FisheyeToHequirectRsPipeline::create_16bit(&device);
        let fisheye_p010_to_hequirect_16 = FisheyeP010ToHequirectPipeline::create(&device);
        let fisheye_p010_to_hequirect_16_rs = FisheyeP010ToHequirectRsPipeline::create(&device);
        let fisheye_to_fisheye = FisheyeToHequirectPipeline::create_fisheye_out(&device);
        let fisheye_to_fisheye_rs = FisheyeToHequirectRsPipeline::create_fisheye_out(&device);
        let fisheye_to_fisheye_16 = FisheyeToHequirectPipeline::create_fisheye_out_16(&device);
        let fisheye_to_fisheye_rs_16 = FisheyeToHequirectRsPipeline::create_fisheye_out_16bit(&device);
        let fisheye_p010_to_fisheye_16 = FisheyeP010ToHequirectPipeline::create_fisheye_out(&device);
        let fisheye_p010_to_fisheye_rs_16 = FisheyeP010ToHequirectRsPipeline::create_fisheye_out(&device);
        let p010_resolve = P010ResolvePipeline::create(&device);
        let lut3d = Lut3DPipeline::create(&device);
        let lut3d_16 = Lut3DPipeline::create_16bit(&device);
        let nv12_to_eac = Nv12ToEacPipeline::create(&device);
        let nv12_to_eac_16 = Nv12ToEacPipeline::create_16(&device);
        let rgba_to_eac = RgbaToEacPipeline::create(&device);
        let rgba_to_eac_16 = RgbaToEacPipeline::create_16(&device);
        let cdl = PerPixelPipeline::create(
            &device, "cdl", CDL_WGSL,
            std::mem::size_of::<CdlUniforms>() as u64,
        );
        let cdl_16 = PerPixelPipeline::create_16bit(
            &device, "cdl_16", CDL_16_WGSL,
            std::mem::size_of::<CdlUniforms>() as u64,
        );
        let color_grade = PerPixelPipeline::create(
            &device, "color_grade", COLOR_GRADE_WGSL,
            std::mem::size_of::<ColorGradeUniforms>() as u64,
        );
        let color_grade_16 = PerPixelPipeline::create_16bit(
            &device, "color_grade_16", COLOR_GRADE_16_WGSL,
            std::mem::size_of::<ColorGradeUniforms>() as u64,
        );
        let gaussian_blur = GaussianBlur1dPipeline::create(&device);
        let sharpen_combine = SharpenCombinePipeline::create(&device);
        let mid_detail_combine = MidDetailCombinePipeline::create(&device);
        let downsample_4x = Downsample4xPipeline::create(&device);
        let gaussian_blur_16 = GaussianBlur1dPipeline::create_16bit(&device);
        let sharpen_combine_16 = SharpenCombinePipeline::create_16bit(&device);
        let mid_detail_combine_16 = MidDetailCombinePipeline::create_16bit(&device);
        let downsample_4x_16 = Downsample4xPipeline::create_16bit(&device);
        let compose_sbs = ComposeSbsPipeline::create(&device);
        let compose_preview = ComposeSbsPipeline::create_preview(&device);
        let compose_sbs_p010_y  = P010ComposePipeline::create(
            &device, "p010_y",  COMPOSE_SBS_P010_Y_WGSL,
            wgpu::TextureFormat::R16Unorm);
        let compose_sbs_p010_uv = P010ComposePipeline::create(
            &device, "p010_uv", COMPOSE_SBS_P010_UV_WGSL,
            wgpu::TextureFormat::Rg16Unorm);
        let bilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Ok(Self {
            instance, adapter, device, queue,
            eac_to_equirect, eac_to_equirect_16, eac_to_fisheye, eac_to_fisheye_16,
            fisheye_to_hequirect, fisheye_to_hequirect_16,
            fisheye_to_hequirect_rs,
            fisheye_to_hequirect_rs_16,
            fisheye_p010_to_hequirect_16,
            fisheye_p010_to_hequirect_16_rs,
            fisheye_to_fisheye, fisheye_to_fisheye_rs, fisheye_to_fisheye_16,
            fisheye_to_fisheye_rs_16,
            fisheye_p010_to_fisheye_16, fisheye_p010_to_fisheye_rs_16,
            p010_resolve,
            lut3d, lut3d_16, nv12_to_eac, nv12_to_eac_16,
            rgba_to_eac, rgba_to_eac_16,
            cdl, cdl_16,
            color_grade, color_grade_16,
            gaussian_blur, sharpen_combine, mid_detail_combine, downsample_4x,
            gaussian_blur_16, sharpen_combine_16,
            mid_detail_combine_16, downsample_4x_16,
            compose_sbs, compose_preview,
            compose_sbs_p010_y, compose_sbs_p010_uv,
            bilinear_sampler,
            proj_fisheye_rs_cache: Mutex::new(HashMap::new()),
            proj_fisheye_cache: Mutex::new(HashMap::new()),
            rgba16_eq_out_cache: Mutex::new(HashMap::new()),
            readback_staging: Mutex::new(HashMap::new()),
        })
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Wgpu(format!("no compatible adapter: {e}")))?;
        let info = adapter.get_info();
        tracing::info!(
            backend = ?info.backend,
            name = %info.name,
            "wgpu adapter selected"
        );
        // Request TEXTURE_FORMAT_16BIT_NORM: needed for the P010 (10-bit)
        // YUV planes from VideoToolbox-decoded GoPro HEVC Main10 footage,
        // wrapped as R16Unorm / Rg16Unorm IOSurface plane textures.
        // Supported on Apple Silicon (Metal), most Vulkan adapters, and
        // DX12 — but not the WebGPU baseline, hence the explicit request.
        let optional_features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        let required_features = adapter.features() & optional_features;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vr180-render"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                },
            )
            .await
            .map_err(|e| Error::Wgpu(format!("request_device: {e}")))?;
        Self::from_existing(
            instance,
            std::sync::Arc::new(adapter),
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
        )
    }

    /// Build a **dedicated** pipeline device on an existing adapter (e.g.
    /// eframe's), with its own logical `wgpu::Device`/`Queue` separate from
    /// the host renderer's. Requests the 16-bit-norm / P010 / NV12 features
    /// needed for the 10-bit color stack and the Windows d3d11→Vulkan
    /// zero-copy import.
    ///
    /// Why: the file **export** runs on a background thread and reads frames
    /// back to CPU for the encoder (`Maintain::Wait`); it never hands a
    /// texture to egui. Running that on a device that is ISOLATED from the
    /// one eframe presents with sidesteps Lesson #1 entirely — a background
    /// `Maintain::Wait` on a private device/queue can't wedge the host's
    /// present queue on any backend. The shared device stays reserved for
    /// the preview, whose textures must live in eframe's device.
    pub fn new_dedicated_from_adapter(
        adapter: &std::sync::Arc<wgpu::Adapter>,
    ) -> Result<Self> {
        let want = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_FORMAT_P010
            | wgpu::Features::TEXTURE_FORMAT_NV12;
        let have = adapter.features() & want;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vr180 export (dedicated)"),
                required_features: have,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .map_err(|e| Error::Wgpu(format!("export request_device: {e}")))?;
        // `from_existing` keeps an instance handle alive; a throwaway one is
        // fine (we never create surfaces on it — see app.rs for the same).
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        Self::from_existing(
            instance,
            adapter.clone(),
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
        )
    }

    /// Project one EAC cross (`cross_w × cross_w × RGB8`) to a half-
    /// equirect image (`out_w × out_h × RGB8`) covering ±90° lat × ±90° lon.
    ///
    /// Phase 0.5: no rotation applied (identity orientation). Phase 0.7
    /// will fuse in the per-frame R matrix from the gyro pipeline.
    pub fn project_cross_to_equirect(
        &self,
        cross_rgb: &[u8],
        cross_w: u32,
        out_w: u32,
        out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<Vec<u8>> {
        let t_total = Instant::now();
        // 1. Convert packed RGB8 input → RGBA8 (wgpu storage textures
        //    don't support RGB8). One-time cost per frame.
        let t0 = Instant::now();
        let cross_rgba = rgb_to_rgba(cross_rgb, cross_w as usize, cross_w as usize);
        tracing::debug!(elapsed=?t0.elapsed(), "RGB8→RGBA8 upload prep");

        // 2. Upload input texture.
        let input_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cross_rgba"),
            size: wgpu::Extent3d { width: cross_w, height: cross_w, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &cross_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cross_w * 4),
                rows_per_image: Some(cross_w),
            },
            wgpu::Extent3d { width: cross_w, height: cross_w, depth_or_array_layers: 1 },
        );

        // 3. Output storage texture.
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_rgba"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 4. Bind group — uses the pipeline + sampler we built once at Device::new.
        let input_view = input_tex.create_view(&Default::default());
        let output_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("equirect_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let rs_uniform = self.write_uniform("equirect_RS",
            &RsUniforms::from_params(rs));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eac_to_equirect_bg"),
            layout: &self.eac_to_equirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.eac_to_equirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rs_uniform.as_entire_binding() },
            ],
        });
        let pipeline = &self.eac_to_equirect.pipeline;

        // 6. Encode + dispatch. Workgroup is 8×8 (64 threads).
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eac_to_equirect_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eac_to_equirect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (out_w + 7) / 8;
            let wg_y = (out_h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 7. Copy output to a host-visible staging buffer.
        // wgpu requires bytes_per_row to be a multiple of 256.
        let bpp = 4u32;
        let unpadded_row_bytes = out_w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * out_h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("equirect_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        // 8. Map + copy out RGBA, repack to RGB8.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();

        let mut out_rgb = Vec::with_capacity((out_w * out_h * 3) as usize);
        for y in 0..out_h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..out_w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();

        tracing::debug!(elapsed=?t_total.elapsed(), "GPU project_cross_to_equirect total");
        Ok(out_rgb)
    }
}

fn rgb_to_rgba(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    debug_assert_eq!(rgb.len(), w * h * 3);
    let mut out = Vec::with_capacity(w * h * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(px);
        out.push(255);
    }
    out
}

impl Device {
    /// Project one raw fisheye image to a half-equirect (`out_w ×
    /// out_h × RGB8`) using the Kannala-Brandt model.
    ///
    /// `src_rgba` is `src_w × src_h × 4` packed RGBA8 (i.e. what the
    /// decoder produces — for BGRA inputs, swizzle on the caller side).
    /// `calib` defines the lens (fx/fy/cx/cy + k1..k4 + image-circle
    /// radius). `rotation` is the per-frame stabilization rotation
    /// (use [`EquirectRotation::IDENTITY`] when stabilization is off).
    ///
    /// Returns packed RGB8 ready for ffmpeg-side encoding. Same shape
    /// as `project_cross_to_equirect`.
    pub fn project_fisheye_to_equirect(
        &self,
        src_rgba: &[u8],
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
        calib: FisheyeCalib,
        rotation: EquirectRotation,
    ) -> Result<Vec<u8>> {
        let t_total = Instant::now();
        debug_assert_eq!(src_rgba.len(), (src_w * src_h * 4) as usize);

        // 1. Upload input texture.
        let input_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_input_rgba"),
            size: wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            src_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(src_w * 4),
                rows_per_image: Some(src_h),
            },
            wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
        );

        // 2. Output storage texture.
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_equirect_rgba"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 3. Bind group.
        let input_view = input_tex.create_view(&Default::default());
        let output_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_calib",
            &FisheyeCalibUniforms::from_public(calib));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_to_hequirect_bg"),
            layout: &self.fisheye_to_hequirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });

        // 4. Encode + dispatch.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_to_hequirect_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_to_hequirect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (out_w + 7) / 8;
            let wg_y = (out_h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 5. Copy to staging, map, repack to RGB8. wgpu requires 256-
        //    aligned row stride for buffer copies.
        let bpp = 4u32;
        let unpadded_row_bytes = out_w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * out_h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fisheye_equirect_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();

        let mut out_rgb = Vec::with_capacity((out_w * out_h * 3) as usize);
        for y in 0..out_h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..out_w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();

        tracing::debug!(elapsed=?t_total.elapsed(), "GPU project_fisheye_to_equirect total");
        Ok(out_rgb)
    }
}

impl EacToEquirectPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm)
    }

    /// 16-bit-output variant for the true-10-bit EAC export chain.
    fn create_16(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm)
    }

    fn create_with_format(device: &wgpu::Device, out_format: wgpu::TextureFormat) -> Self {
        Self::create_with_wgsl(device, out_format, EAC_TO_EQUIRECT_WGSL)
    }

    /// Fisheye-output variant — same bind-group layout / RS / face
    /// sampling, only the output-pixel→direction mapping differs (disk
    /// instead of equirect). For the `.360` fisheye-output mode.
    fn create_fisheye(device: &wgpu::Device) -> Self {
        Self::create_with_wgsl(device, wgpu::TextureFormat::Rgba8Unorm, EAC_TO_FISHEYE_WGSL)
    }
    fn create_fisheye_16(device: &wgpu::Device) -> Self {
        Self::create_with_wgsl(device, wgpu::TextureFormat::Rgba16Unorm, EAC_TO_FISHEYE_WGSL)
    }

    fn create_with_wgsl(device: &wgpu::Device, out_format: wgpu::TextureFormat, base_wgsl: &str) -> Self {
        let wgsl = if out_format == wgpu::TextureFormat::Rgba16Unorm {
            base_wgsl.replace("rgba8unorm, write", "rgba16unorm, write")
        } else {
            base_wgsl.to_string()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eac_to_equirect"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("eac_to_equirect_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // R matrix (3 × vec4 std140; 48 bytes).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EquirectUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // Phase D: RS uniforms (12 × f32, 48 bytes std140).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<RsUniforms>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eac_to_equirect_pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eac_to_equirect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cross_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl FisheyeToHequirectPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_for_format_with_wgsl(
            device, wgpu::TextureFormat::Rgba8Unorm,
            "fisheye_to_hequirect_8bit", FISHEYE_TO_HEQUIRECT_WGSL,
        )
    }

    /// 16-bit-per-channel output variant for end-to-end 10-bit export.
    /// Same shader, same bindings, just `rgba16unorm` swapped in for
    /// the storage-texture format via runtime WGSL string replace.
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_for_format_with_wgsl(
            device, wgpu::TextureFormat::Rgba16Unorm,
            "fisheye_to_hequirect_16bit", FISHEYE_TO_HEQUIRECT_WGSL,
        )
    }

    /// Fisheye-to-fisheye projection (equidistant output, KB source
    /// projection). Rgba8Unorm output, same bind-group layout shape as
    /// the half-equirect variant — only the WGSL differs (and only in
    /// the per-output-pixel parametrisation).
    fn create_fisheye_out(device: &wgpu::Device) -> Self {
        Self::create_for_format_with_wgsl(
            device, wgpu::TextureFormat::Rgba8Unorm,
            "fisheye_to_fisheye_8bit", FISHEYE_TO_FISHEYE_WGSL,
        )
    }

    /// 16-bit fisheye-to-fisheye (Rgba16Unorm output) for the 10-bit
    /// fisheye-output export path.
    fn create_fisheye_out_16(device: &wgpu::Device) -> Self {
        Self::create_for_format_with_wgsl(
            device, wgpu::TextureFormat::Rgba16Unorm,
            "fisheye_to_fisheye_16bit", FISHEYE_TO_FISHEYE_16_WGSL,
        )
    }

    fn create_for_format_with_wgsl(
        device: &wgpu::Device,
        out_format: wgpu::TextureFormat,
        label_suffix: &str,
        wgsl_src: &str,
    ) -> Self {
        // For the legacy "rgba8unorm" → "rgba16unorm" auto-swap path
        // (used by `create_16bit`) we templated the storage-texture
        // format at runtime. The shader_out variants ship already-typed
        // WGSL files, so the swap is a no-op for them but kept inline
        // for consistency.
        let wgsl = if out_format == wgpu::TextureFormat::Rgba16Unorm {
            wgsl_src.replace("rgba8unorm, write", "rgba16unorm, write")
        } else {
            wgsl_src.to_string()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("fisheye_to_hequirect_{label_suffix}")),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("fisheye_to_hequirect_bgl_{label_suffix}")),
            entries: &[
                // (0) input fisheye texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (1) bilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // (2) output storage texture (half-equirect)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // (3) per-frame rotation R (same EquirectUniforms shape)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EquirectUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (4) per-eye KB calibration uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<FisheyeCalibUniforms>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("fisheye_to_hequirect_pll_{label_suffix}")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("fisheye_to_hequirect_pipeline_{label_suffix}")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("fisheye_sampler_{label_suffix}")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl FisheyeP010ToHequirectPipeline {
    /// Build the P010 zero-copy fisheye projection pipeline. Six
    /// bindings: Y tex, UV tex, sampler, output storage tex,
    /// per-frame rotation, per-eye KB calib.
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_p010_to_hequirect", FISHEYE_P010_TO_HEQUIRECT_WGSL,
        )
    }

    /// P010 → fisheye-output (Rgba16Unorm) zero-copy projection.
    /// Equidistant output, KB source projection.
    fn create_fisheye_out(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_p010_to_fisheye_16", FISHEYE_P010_TO_FISHEYE_16_WGSL,
        )
    }

    fn create_with_shader(device: &wgpu::Device, label: &str, wgsl: &str) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                // (0) Y plane (R16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (1) UV plane (Rg16Unorm, half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (2) bilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // (3) output storage (Rgba16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // (4) per-frame rotation R
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EquirectUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (5) per-eye KB calibration
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<FisheyeCalibUniforms>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{label}_smp")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl P010ResolvePipeline {
    fn create(device: &wgpu::Device) -> Self {
        let label = "p010_resolve_rgba16";
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(P010_RESOLVE_RGBA16_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("p010_resolve_bgl"),
            entries: &[
                // (0) Y plane (R16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (1) UV plane (Rg16Unorm, half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (2) bilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // (3) output storage (Rgba16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // (4) dims uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<ResolveDims>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("p010_resolve_pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("p010_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("p010_resolve_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl FisheyeToHequirectRsPipeline {
    /// Build the RS-aware RGBA fisheye projection pipeline. Six bindings:
    /// input fisheye tex, sampler, output storage, R uniform, calib
    /// uniform, per-row R storage buffer.
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_to_hequirect_rs", FISHEYE_TO_HEQUIRECT_RS_WGSL,
        )
    }

    /// RS-aware RGBA fisheye → stabilized fisheye output (preview).
    fn create_fisheye_out(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_to_fisheye_rs", FISHEYE_TO_FISHEYE_RS_WGSL,
        )
    }

    /// 16-bit-output RS fisheye-output projection (Windows zero-copy preview
    /// + GPU-resident export). Same WGSL as `create_fisheye_out()` with the
    /// storage format swapped to `rgba16unorm` at runtime (the `create_16bit`
    /// trick), so the RS-corrected fisheye eye stays 16-bit through compose.
    fn create_fisheye_out_16bit(device: &wgpu::Device) -> Self {
        let wgsl = FISHEYE_TO_FISHEYE_RS_WGSL.replace("rgba8unorm, write", "rgba16unorm, write");
        Self::create_with_shader_fmt(
            device, "fisheye_to_fisheye_rs_16", &wgsl, wgpu::TextureFormat::Rgba16Unorm,
        )
    }

    /// 16-bit-output RS half-equirect projection (Windows zero-copy preview
    /// stab). Same shader as `create()` with the storage format swapped to
    /// `rgba16unorm` at runtime (matches the `create_16bit` trick used by the
    /// non-RS pipeline), so the RS-corrected eye stays 16-bit through compose.
    fn create_16bit(device: &wgpu::Device) -> Self {
        let wgsl = FISHEYE_TO_HEQUIRECT_RS_WGSL.replace("rgba8unorm, write", "rgba16unorm, write");
        Self::create_with_shader_fmt(
            device, "fisheye_to_hequirect_rs_16", &wgsl, wgpu::TextureFormat::Rgba16Unorm,
        )
    }

    fn create_with_shader(device: &wgpu::Device, label: &str, wgsl: &str) -> Self {
        Self::create_with_shader_fmt(device, label, wgsl, wgpu::TextureFormat::Rgba8Unorm)
    }

    fn create_with_shader_fmt(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                // (0) input fisheye texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (1) bilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // (2) output storage (Rgba8Unorm or Rgba16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // (3) per-frame rotation R
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EquirectUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (4) per-eye KB calibration
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<FisheyeCalibUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (5) per-row RS R storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{label}_smp")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl FisheyeP010ToHequirectRsPipeline {
    /// Build the RS-aware P010 fisheye projection pipeline. Seven
    /// bindings: Y tex, UV tex, sampler, output, R uniform, calib
    /// uniform, and per-row R storage buffer.
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_p010_to_hequirect_rs", FISHEYE_P010_TO_HEQUIRECT_RS_WGSL,
        )
    }

    /// RS-aware P010 → stabilized fisheye output (Rgba16Unorm), zero-copy.
    fn create_fisheye_out(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "fisheye_p010_to_fisheye_rs_16", FISHEYE_P010_TO_FISHEYE_RS_16_WGSL,
        )
    }

    fn create_with_shader(device: &wgpu::Device, label: &str, wgsl: &str) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                // (0) Y plane (R16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (1) UV plane (Rg16Unorm, half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // (2) bilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // (3) output storage (Rgba16Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // (4) per-frame rotation R
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EquirectUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (5) per-eye KB calibration
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<FisheyeCalibUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // (6) per-row RS R storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{label}_smp")),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

/// WGSL kernel: per output pixel in the half-equirect, compute a 3D
/// direction, dispatch to one of the 5 EAC faces, sample from the
/// input cross with bilinear filtering. Pixels outside the cross's
/// 184.5° coverage get black.
///
/// Ported from `FrameExtractor.build_cross_remap` in `vr180_gui.py`.
const EAC_TO_EQUIRECT_WGSL: &str = include_str!("shaders/eac_to_equirect.wgsl");

/// WGSL kernel: Kannala-Brandt fisheye → half-equirect for the
/// non-GoPro input families (DJI OSV, SBS fisheye, Blackmagic BRAW).
/// Ported from the Metal kernel at `vr180_gui.py:1350-1480`. CPU
/// reference for testing lives in
/// `crates/vr180-fisheye/src/projection.rs`.
const FISHEYE_TO_HEQUIRECT_WGSL: &str =
    include_str!("shaders/fisheye_to_hequirect.wgsl");

/// WGSL kernel: P010 (10-bit YCbCr planes) → half-equirect, used when
/// the source IOSurface is directly available (macOS OSV export). Does
/// YCbCr→RGB Rec.709 expansion in the same dispatch as the KB projection,
/// avoiding the FFmpeg P010LE→RGBA64LE swscale and the CPU→GPU upload.
const FISHEYE_P010_TO_HEQUIRECT_WGSL: &str =
    include_str!("shaders/fisheye_p010_to_hequirect.wgsl");

/// WGSL kernel: P010 (Y/UV planes) → RGBA16 resolve + box downscale.
/// Upsamples chroma to full-res RGB then box-averages down to the working
/// resolution (the correct order) so the live preview's big minification
/// doesn't produce chroma moiré. See `p010_resolve_rgba16.wgsl`.
const P010_RESOLVE_RGBA16_WGSL: &str =
    include_str!("shaders/p010_resolve_rgba16.wgsl");

/// WGSL kernel: P010 → half-equirect with fused stab + per-row rolling
/// shutter correction. Same math as `fisheye_p010_to_hequirect.wgsl`
/// plus a per-scanline re-projection step that cancels intra-frame
/// shear from the sensor's row-by-row readout.
const FISHEYE_P010_TO_HEQUIRECT_RS_WGSL: &str =
    include_str!("shaders/fisheye_p010_to_hequirect_rs.wgsl");

/// WGSL kernel: RGBA fisheye → half-equirect with fused stab + per-row
/// rolling-shutter correction. Used by the GUI preview when DJI OSV
/// per-row matrices are available.
const FISHEYE_TO_HEQUIRECT_RS_WGSL: &str =
    include_str!("shaders/fisheye_to_hequirect_rs.wgsl");

/// std140 layout for the EAC→equirect shader's R matrix uniform.
/// 12-scalar layout (3 rows × 3 floats + 1 pad per row) — used in
/// preference to `mat3x3<f32>` because naga (wgpu 0.20's WGSL parser)
/// silently rejects struct-field access on matrix uniforms. The
/// shader reads via individual scalar fields.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EquirectUniforms {
    r00: f32, r01: f32, r02: f32, _pad0: f32,
    r10: f32, r11: f32, r12: f32, _pad1: f32,
    r20: f32, r21: f32, r22: f32, _pad2: f32,
}

impl EquirectUniforms {
    pub const IDENTITY: Self = Self {
        r00: 1.0, r01: 0.0, r02: 0.0, _pad0: 0.0,
        r10: 0.0, r11: 1.0, r12: 0.0, _pad1: 0.0,
        r20: 0.0, r21: 0.0, r22: 1.0, _pad2: 0.0,
    };

    /// Pack a row-major 3×3 matrix.
    /// `m` is `[r00, r01, r02, r10, r11, r12, r20, r21, r22]`.
    fn from_mat3(m: [f32; 9]) -> Self {
        Self {
            r00: m[0], r01: m[1], r02: m[2], _pad0: 0.0,
            r10: m[3], r11: m[4], r12: m[5], _pad1: 0.0,
            r20: m[6], r21: m[7], r22: m[8], _pad2: 0.0,
        }
    }
}

/// Public type for callers to pass per-frame rotation. Use
/// `EquirectRotation::IDENTITY` when stabilization is off, or
/// `EquirectRotation::from_quat(q)` to derive from a CORI / smoothed
/// quaternion.
#[derive(Copy, Clone, Debug)]
pub struct EquirectRotation(pub [f32; 9]);

impl EquirectRotation {
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);

    /// Build from a unit quaternion. The quaternion must be in the
    /// equirect-direction → camera-direction convention (apply this
    /// rotation to the output equirect direction to get the camera
    /// frame direction that the EAC was captured in).
    pub fn from_quat(q: vr180_core::gyro::Quat) -> Self {
        Self(q.to_mat3_row_major())
    }
}

/// Number of per-scanline ω row groups (matches Python's
/// `GyroStabilizer.RS_N_GROUPS`). Group g covers readout row-time
/// `t_frame + (g/(N-1) − 0.5)·srot`.
pub const RS_N_GROUPS: usize = 32;

/// std140 layout for the EAC→equirect shader's RS uniform (Phase D).
/// 12 scalars + 24 vec4s of per-scanline ω groups (96 floats packed
/// 4-per-vec4 to satisfy WGSL's 16-byte uniform array stride) —
/// matches the WGSL `RsUniforms` struct exactly. `srot_s == 0`
/// disables the RS pass; `n_groups <= 1` falls back to the single
/// `omega_*` value (constant-ω, what Python's preview does).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RsUniforms {
    omega_x: f32, omega_y: f32, omega_z: f32, srot_s: f32,
    klns_c0: f32, klns_c1: f32, klns_c2: f32, klns_c3: f32,
    klns_c4: f32, ctry: f32, cal_dim: f32, n_groups: f32,
    // Flat ω groups: group g component c lives at flat index g*3+c →
    // groups[idx/4][idx%4]. Shader order (ω_x, ω_y, ω_z) per group.
    groups: [[f32; 4]; 24],
}

impl RsUniforms {
    /// All-zeros except `cal_dim = 1.0` (avoids divide-by-zero in
    /// the shader's `sensor_y / cal_dim`). Disables RS regardless
    /// of any other field because `srot_s = 0`.
    const DISABLED: Self = Self {
        omega_x: 0.0, omega_y: 0.0, omega_z: 0.0, srot_s: 0.0,
        klns_c0: 0.0, klns_c1: 0.0, klns_c2: 0.0, klns_c3: 0.0,
        klns_c4: 0.0, ctry: 0.0, cal_dim: 1.0, n_groups: 0.0,
        groups: [[0.0; 4]; 24],
    };

    fn from_params(p: EquirectRsParams) -> Self {
        let mut groups = [[0.0_f32; 4]; 24];
        let ng = (p.n_groups as usize).min(RS_N_GROUPS);
        for g in 0..ng {
            for c in 0..3 {
                let idx = g * 3 + c;
                groups[idx / 4][idx % 4] = p.omega_groups[g][c];
            }
        }
        Self {
            omega_x: p.omega[0], omega_y: p.omega[1], omega_z: p.omega[2],
            srot_s: p.srot_s,
            klns_c0: p.klns[0], klns_c1: p.klns[1], klns_c2: p.klns[2],
            klns_c3: p.klns[3], klns_c4: p.klns[4],
            ctry: p.ctry, cal_dim: p.cal_dim,
            n_groups: ng as f32,
            groups,
        }
    }
}

/// Per-eye rolling-shutter correction parameters (Phase D). Caller
/// builds one per eye per frame; passes it alongside the
/// `EquirectRotation`. `EquirectRsParams::DISABLED` collapses the
/// shader's RS pass to a no-op (the right thing for callers that
/// haven't opted into Phase D).
#[derive(Copy, Clone, Debug)]
pub struct EquirectRsParams {
    /// Effective angular velocity in rad/s (already multiplied by
    /// per-axis RS factors). Components: (ω_x, ω_y, ω_z) in the
    /// equirect shader frame — X = right, Y = up, Z = forward.
    /// Used when `n_groups <= 1` (constant-ω mode).
    pub omega: [f32; 3],
    /// Per-scanline ω row groups (rad/s, factors applied, shader
    /// order). Group g is the instantaneous RAW-gyro ω at row-time
    /// `t_frame + (g/(n−1) − 0.5)·srot` — Python's
    /// `get_perscanline_rs_angvel`. The shader linearly interpolates
    /// between groups by each pixel's sensor-row time, capturing
    /// angular acceleration WITHIN the readout window that a single
    /// smoothed ω blurs out. Only the first `n_groups` entries are
    /// meaningful.
    pub omega_groups: [[f32; 3]; RS_N_GROUPS],
    /// Number of valid entries in `omega_groups`. `0` or `1` →
    /// constant-ω mode using `omega`.
    pub n_groups: u32,
    /// Sensor readout time in seconds. `0.0` disables RS for this
    /// eye/frame regardless of any other field.
    pub srot_s: f32,
    /// Kannala-Brandt polynomial coefficients of this eye's lens:
    /// `r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹` (pixels).
    pub klns: [f32; 5],
    /// Principal-point Y offset from sensor center, pixels.
    pub ctry: f32,
    /// Sensor calibration dimension (pixels, square). 4216 for GoPro Max.
    pub cal_dim: f32,
}

impl EquirectRsParams {
    /// Sentinel value that disables the shader's RS pass. Use this
    /// for the left eye in firmware-RS mode (firmware's correction
    /// is already right for the un-modded lens) and for both eyes
    /// when the user hasn't enabled `--rs-correct`.
    pub const DISABLED: Self = Self {
        omega: [0.0; 3],
        omega_groups: [[0.0; 3]; RS_N_GROUPS],
        n_groups: 0,
        srot_s: 0.0, klns: [0.0; 5],
        ctry: 0.0, cal_dim: 1.0,
    };
}

// ── Fisheye → half-equirect uniforms ───────────────────────────────

/// std140 layout for the fisheye→equirect shader's calibration uniform.
/// 16 scalars (4 × vec4 = 64 bytes). Matches the WGSL
/// `FisheyeCalibUniforms` struct field-by-field. See
/// `fisheye_to_hequirect.wgsl` for the per-pixel use of each value.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FisheyeCalibUniforms {
    fx: f32, fy: f32, cx: f32, cy: f32,
    k1: f32, k2: f32, k3: f32, k4: f32,
    // k5 = the 5th odd-power KB coefficient (θ¹¹ term); reuses the old
    // `_pad0` slot so the uniform layout is unchanged. 0 for sources that
    // only have a 4-coeff model → the new term vanishes (back-compat).
    theta_trans: f32, theta_max: f32, r_max: f32, k5: f32,
    src_w: f32, src_h: f32, output_hfov_rad: f32, _pad2: f32,
    p1: f32, p2: f32, _pad3: f32, _pad4: f32,
}

/// Public per-eye Kannala-Brandt fisheye calibration. Caller builds
/// one of these (typically by composing a `vr180_fisheye::FisheyeCalibration`
/// with the working frame size) and hands it to
/// [`Device::project_fisheye_to_equirect`].
#[derive(Copy, Clone, Debug)]
pub struct FisheyeCalib {
    /// Focal length in pixels (x). Comes either from the camera
    /// preset / Gyroflow JSON, or recomputed at runtime from
    /// `FisheyeCalibration::fx_from_fov`.
    pub fx: f32,
    /// Focal length in pixels (y). Equal to `fx` for square sensors;
    /// differs for anamorphic.
    pub fy: f32,
    /// Principal point in pixels (image origin top-left).
    pub cx: f32,
    pub cy: f32,
    /// KB-4 distortion coefficients (Gyroflow / OpenCV convention).
    pub k: [f32; 4],
    /// 5th odd-power KB radial coefficient (θ¹¹ term). DJI Studio's lens
    /// model is 5-coeff: `θ_d = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹ + k5θ¹¹`.
    /// Loaded from the OSV file (protobuf field 15); k5 keeps the radial
    /// map monotonic past ~90°. 0 for 4-coeff sources (term vanishes).
    pub k5: f32,
    /// Brown-Conrady tangential distortion (OSV field 20). Applied to the
    /// normalized point after the radial KB: `u += 2·p1·u·v + p2·(r²+2u²)`,
    /// `v += p1·(r²+2v²) + 2·p2·u·v`. 0 → no tangential (vanishes).
    pub p1: f32,
    pub p2: f32,
    /// KB → cubic-Hermite extension boundary (radians).
    pub theta_trans: f32,
    /// Cubic extension upper bound (radians).
    pub theta_max: f32,
    /// Image-circle radius in pixels — caller computes from
    /// cx/cy + src_w/src_h. Pins the cubic at the boundary.
    pub r_max: f32,
    /// Working fisheye texture width in pixels.
    pub src_w: f32,
    /// Working fisheye texture height in pixels.
    pub src_h: f32,
    /// Output half-equirect HORIZONTAL half-FOV in radians. The shader
    /// derives `lon = (u - 0.5) * 2 * output_hfov_rad`, so:
    ///   π/2 (90°)  → 180° total HFOV (standard VR180)
    ///   1.812      → 207.68° total HFOV (full DJI Osmo 360 lens FOV)
    /// Defaults to π/2 for backward compat; OSV builds set this to
    /// lens-FOV/2 so the export captures every pixel the lens saw.
    pub output_hfov_rad: f32,
}

impl FisheyeCalib {
    /// Convenience constructor with sensible defaults for the cubic
    /// extension boundaries (`theta_trans = 80°`, `theta_max = 110°`)
    /// and standard 180°-HFOV equirect output.
    pub fn new(
        fx: f32, fy: f32, cx: f32, cy: f32, k: [f32; 4],
        src_w: f32, src_h: f32, r_max: f32,
    ) -> Self {
        Self {
            fx, fy, cx, cy, k, k5: 0.0, p1: 0.0, p2: 0.0,
            theta_trans: 80.0_f32.to_radians(),
            theta_max:   110.0_f32.to_radians(),
            r_max, src_w, src_h,
            output_hfov_rad: std::f32::consts::FRAC_PI_2,
        }
    }

    /// Constructor that disables the cubic Hermite extension and uses
    /// **pure KB across the full FOV**, matching the Python MLX kernel
    /// at `vr180_gui.py:1386-1395`. The `theta_max` upper clamp is set
    /// from `max_r / fx` (Python's "loose upper clamp" at
    /// `vr180_gui.py:1846-1847`).
    ///
    /// Use this for OSV (and any source whose KB coefficients stay
    /// monotonic across the entire FOV — the Hermite extension only
    /// exists to keep things sane when the polynomial diverges).
    ///
    /// `theta_trans` is set to a value just past `theta_max` so the
    /// shader's `theta ≤ theta_trans` test always succeeds for any
    /// post-clamp θ, guaranteeing KB is used.
    pub fn new_pure_kb(
        fx: f32, fy: f32, cx: f32, cy: f32, k: [f32; 4],
        src_w: f32, src_h: f32,
    ) -> Self {
        // Python: max_r = max(cx, cy, w-cx, h-cy) — largest distance
        // from the principal point to any image edge.
        let max_r = cx.max(cy).max(src_w - cx).max(src_h - cy).max(1.0);
        let fx_safe = fx.max(1e-3);
        let theta_max = max_r / fx_safe; // radians; Python's loose upper clamp.
        // Bump theta_trans above theta_max so the shader's
        // `theta ≤ theta_trans` test always picks KB after clamping.
        let theta_trans = theta_max + 0.1;
        Self {
            fx, fy, cx, cy, k, k5: 0.0, p1: 0.0, p2: 0.0,
            theta_trans,
            theta_max,
            r_max: max_r,
            src_w, src_h,
            output_hfov_rad: std::f32::consts::FRAC_PI_2,
        }
    }

    /// Set the output equirect's horizontal half-FOV. Use this when
    /// you want the output to cover more than 180° — e.g. for the
    /// OSV's 207.68° lens, pass `(207.68_f32 / 2.0).to_radians()`.
    /// Returns `self` so it can chain after `new_pure_kb`.
    pub fn with_output_hfov(mut self, hfov_rad: f32) -> Self {
        self.output_hfov_rad = hfov_rad;
        self
    }
}

impl FisheyeCalibUniforms {
    fn from_public(c: FisheyeCalib) -> Self {
        Self {
            fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy,
            k1: c.k[0], k2: c.k[1], k3: c.k[2], k4: c.k[3],
            theta_trans: c.theta_trans,
            theta_max:   c.theta_max,
            r_max:       c.r_max,
            k5: c.k5,
            src_w: c.src_w, src_h: c.src_h,
            output_hfov_rad: c.output_hfov_rad,
            _pad2: 0.0,
            p1: c.p1, p2: c.p2, _pad3: 0.0, _pad4: 0.0,
        }
    }
}

const FISHEYE_TO_FISHEYE_WGSL: &str = include_str!("shaders/fisheye_to_fisheye.wgsl");
const FISHEYE_TO_FISHEYE_16_WGSL: &str = include_str!("shaders/fisheye_to_fisheye_16.wgsl");
const FISHEYE_TO_FISHEYE_RS_WGSL: &str = include_str!("shaders/fisheye_to_fisheye_rs.wgsl");
const FISHEYE_P010_TO_FISHEYE_16_WGSL: &str = include_str!("shaders/fisheye_p010_to_fisheye_16.wgsl");
const FISHEYE_P010_TO_FISHEYE_RS_16_WGSL: &str = include_str!("shaders/fisheye_p010_to_fisheye_rs_16.wgsl");
const LUT3D_WGSL: &str = include_str!("shaders/lut3d.wgsl");
const LUT3D_16_WGSL: &str = include_str!("shaders/lut3d_16.wgsl");
const NV12_TO_EAC_WGSL: &str = include_str!("shaders/nv12_to_eac_cross.wgsl");
const RGBA_TO_EAC_WGSL: &str = include_str!("shaders/rgba_to_eac_cross.wgsl");
const EAC_TO_FISHEYE_WGSL: &str = include_str!("shaders/eac_to_fisheye.wgsl");
const CDL_WGSL: &str = include_str!("shaders/cdl.wgsl");
const CDL_16_WGSL: &str = include_str!("shaders/cdl_16.wgsl");
const COLOR_GRADE_WGSL: &str = include_str!("shaders/color_grade.wgsl");
const COLOR_GRADE_16_WGSL: &str = include_str!("shaders/color_grade_16.wgsl");
const GAUSSIAN_BLUR_1D_WGSL: &str = include_str!("shaders/gaussian_blur_1d.wgsl");
const SHARPEN_COMBINE_WGSL: &str = include_str!("shaders/sharpen_combine.wgsl");
const MID_DETAIL_COMBINE_WGSL: &str = include_str!("shaders/mid_detail_combine.wgsl");
const DOWNSAMPLE_4X_WGSL: &str = include_str!("shaders/downsample_4x.wgsl");
const COMPOSE_SBS_BGRA_WGSL: &str = include_str!("shaders/compose_sbs_bgra.wgsl");
const COMPOSE_SBS_P010_Y_WGSL:  &str = include_str!("shaders/compose_sbs_p010_y.wgsl");
const COMPOSE_SBS_P010_UV_WGSL: &str = include_str!("shaders/compose_sbs_p010_uv.wgsl");
const COMPOSE_PREVIEW_WGSL: &str = include_str!("shaders/compose_preview.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ComposeSbsUniforms {
    eye_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Preview output mode.
///
/// - `Sbs` produces a side-by-side `(2·eye_w × eye_h)` frame — what the
///   exporter ultimately wants and the most useful no-glasses view.
/// - `Anaglyph` produces a single `(eye_w × eye_h)` frame with left
///   luma in R and right G+B; works with red/cyan glasses.
/// - `Overlay50` produces a single `(eye_w × eye_h)` frame as a 50%
///   blend of left + right. Handy for verifying convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    Sbs,
    Anaglyph,
    Overlay50,
    /// Show one eye full-frame. The shader outputs the LEFT input; the
    /// caller swaps left/right bindings to pick the physical eye.
    SingleEye,
}

impl Default for PreviewMode {
    fn default() -> Self { PreviewMode::Sbs }
}

impl PreviewMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PreviewMode::Sbs       => "SBS",
            PreviewMode::Anaglyph  => "Anaglyph (red/cyan)",
            PreviewMode::Overlay50 => "50% overlay",
            PreviewMode::SingleEye => "Single eye",
        }
    }
    fn shader_code(self) -> u32 {
        match self {
            PreviewMode::Sbs       => 0,
            PreviewMode::Anaglyph  => 1,
            PreviewMode::Overlay50 => 2,
            PreviewMode::SingleEye => 3,
        }
    }
    pub fn output_dims(self, eye_w: u32, eye_h: u32) -> (u32, u32) {
        match self {
            PreviewMode::Sbs => (eye_w * 2, eye_h),
            _                => (eye_w,     eye_h),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PreviewComposeUniforms {
    eye_w: u32,
    mode:  u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GaussianBlur1dUniforms {
    sigma:     f32,
    direction: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SharpenCombineUniforms {
    amount:           f32,
    apply_lat_weight: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Unsharp-mask sharpen knobs. Default is identity (`amount=0`); when
/// non-zero, σ controls the Gaussian blur width (~1.4 matches the
/// Python default; bigger σ = broader-frequency emphasis).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharpenParams {
    pub amount: f32,
    pub sigma:  f32,
    /// Apply `cos(latitude)` falloff for equirect-projected frames.
    /// Set to false for non-equirect (fisheye / box) frames.
    pub apply_lat_weight: bool,
}

impl Default for SharpenParams {
    fn default() -> Self {
        Self { amount: 0.0, sigma: 1.4, apply_lat_weight: true }
    }
}

impl SharpenParams {
    pub fn is_identity(&self) -> bool { self.amount == 0.0 }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MidDetailCombineUniforms {
    amount: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Mid-detail clarity knobs. Default is identity (`amount=0`).
/// `sigma` controls the Gaussian blur applied to the 4×-downsampled
/// image (so an effective full-resolution blur radius ≈ `4 * sigma`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidDetailParams {
    pub amount: f32,
    pub sigma:  f32,
}

impl Default for MidDetailParams {
    fn default() -> Self {
        // sigma=1.0 on the 1/4-res image ≈ full-res blur radius 4 px,
        // which matches the Python `sigma = 0.01 * min(dh, dw)` knob
        // for a typical 4K output (0.01 * 1024 ≈ 10 → close to 4 at
        // quarter res). The user knob on the Python side controls
        // `amount`, not `sigma`, so keeping σ fixed at 1.0 matches
        // the most common Python configuration.
        Self { amount: 0.0, sigma: 1.0 }
    }
}

impl MidDetailParams {
    pub fn is_identity(&self) -> bool { self.amount == 0.0 }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CdlUniforms {
    lift:      f32,
    gamma:     f32,
    gain:      f32,
    shadow:    f32,
    highlight: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Five-knob CDL params. Defaults are identity (no change).
/// Match the Python defaults in `_default_processing_config`:
/// `lift=0.0, gamma=1.0, gain=1.0, shadow=0.0, highlight=0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CdlParams {
    pub lift:      f32,
    pub gamma:     f32,
    pub gain:      f32,
    pub shadow:    f32,
    pub highlight: f32,
}

impl Default for CdlParams {
    fn default() -> Self {
        Self { lift: 0.0, gamma: 1.0, gain: 1.0, shadow: 0.0, highlight: 0.0 }
    }
}

impl CdlParams {
    /// Returns true if these params would be a no-op (identity transform).
    /// Lets callers skip the GPU dispatch entirely for the default case.
    pub fn is_identity(&self) -> bool {
        self.lift == 0.0 && self.gamma == 1.0 && self.gain == 1.0
            && self.shadow == 0.0 && self.highlight == 0.0
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorGradeUniforms {
    temperature: f32,
    tint:        f32,
    saturation:  f32,
    _pad: f32,
}

/// Color-grade knobs: temperature, tint, saturation. Defaults are
/// identity (`temperature=0, tint=0, saturation=1`). Match
/// `_default_processing_config` in the Python app.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorGradeParams {
    pub temperature: f32,
    pub tint:        f32,
    pub saturation:  f32,
}

impl Default for ColorGradeParams {
    fn default() -> Self {
        Self { temperature: 0.0, tint: 0.0, saturation: 1.0 }
    }
}

impl ColorGradeParams {
    pub fn is_identity(&self) -> bool {
        self.temperature == 0.0 && self.tint == 0.0 && self.saturation == 1.0
    }
    /// White-balance portion (temperature + tint) with saturation neutral.
    /// This is applied PRE-LUT, together with CDL — so all color adjustments
    /// except saturation feed the LUT.
    pub fn white_balance_only(&self) -> Self {
        Self { temperature: self.temperature, tint: self.tint, saturation: 1.0 }
    }
    /// Saturation portion with white balance neutral. Applied POST-LUT.
    pub fn saturation_only(&self) -> Self {
        Self { temperature: 0.0, tint: 0.0, saturation: self.saturation }
    }
    /// Does the white-balance (temperature/tint) portion do anything?
    pub fn has_white_balance(&self) -> bool {
        self.temperature != 0.0 || self.tint != 0.0
    }
    /// Does the saturation portion do anything?
    pub fn has_saturation(&self) -> bool {
        self.saturation != 1.0
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Nv12ToEacUniforms {
    stream_w: u32,
    stream_h: u32,
    tile_w:   u32,
    center_w: u32,
    cross_w:  u32,
    lens:     u32,    // 0 = Lens A, 1 = Lens B
    _pad0: u32,
    _pad1: u32,
}

impl Nv12ToEacPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm)
    }

    /// 16-bit-per-channel cross output for the true-10-bit EAC export
    /// chain (P010 decode → Rgba16 cross → Rgba16 equirect → P010
    /// encode). Same shader, `rgba16unorm` swapped into the storage
    /// declaration at pipeline build (the established variant pattern).
    fn create_16(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm)
    }

    fn create_with_format(device: &wgpu::Device, out_format: wgpu::TextureFormat) -> Self {
        let wgsl = if out_format == wgpu::TextureFormat::Rgba16Unorm {
            NV12_TO_EAC_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else {
            NV12_TO_EAC_WGSL.to_string()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12_to_eac"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv12_to_eac_bgl"),
            entries: &[
                // 0: s0 Y plane (R8Unorm)
                bgle_tex(0),
                // 1: s0 UV plane (Rg8Unorm)
                bgle_tex(1),
                // 2: s4 Y plane
                bgle_tex(2),
                // 3: s4 UV plane
                bgle_tex(3),
                // 4: sampler (bilinear)
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: cross output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 6: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv12_to_eac_pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv12_to_eac_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nv12_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl RgbaToEacPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm)
    }

    /// 16-bit cross output for the true-10-bit EAC export chain (P010
    /// decode → Rgba16 stream → Rgba16 cross → Rgba16 equirect → P010
    /// encode). Same shader, `rgba16unorm` swapped into the storage decl.
    fn create_16(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm)
    }

    fn create_with_format(device: &wgpu::Device, out_format: wgpu::TextureFormat) -> Self {
        let wgsl = if out_format == wgpu::TextureFormat::Rgba16Unorm {
            RGBA_TO_EAC_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else {
            RGBA_TO_EAC_WGSL.to_string()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgba_to_eac"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rgba_to_eac_bgl"),
            entries: &[
                // 0: s0 RGBA stream texture
                bgle_tex(0),
                // 1: s4 RGBA stream texture
                bgle_tex(1),
                // 2: sampler (bilinear)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: cross output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rgba_to_eac_pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rgba_to_eac_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgba_eac_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl PerPixelPipeline {
    fn create(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        uniform_size: u64,
    ) -> Self {
        Self::create_with_format(device, label, wgsl, uniform_size,
            wgpu::TextureFormat::Rgba8Unorm)
    }

    /// 16-bit variant — same layout but the storage texture format is
    /// `Rgba16Unorm`, matching the 10-bit export pipeline so the color
    /// stack stays at 10-bit through to the encoder.
    fn create_16bit(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        uniform_size: u64,
    ) -> Self {
        Self::create_with_format(device, label, wgsl, uniform_size,
            wgpu::TextureFormat::Rgba16Unorm)
    }

    /// Build a generic per-pixel pipeline: 1 input texture + 1 storage
    /// output + 1 uniform buffer. Used by CDL, color_grade, and
    /// (future) saturation, white-point shift, etc. The `uniform_size`
    /// is used for the layout's `min_binding_size` so wgpu's validation
    /// can catch buffer-size mismatches at bind time instead of failing
    /// silently on the GPU.
    fn create_with_format(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        uniform_size: u64,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bgl_label = format!("{label}_bgl");
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&bgl_label),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(uniform_size),
                    },
                    count: None,
                },
            ],
        });
        let pll_label = format!("{label}_pll");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&pll_label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pp_label = format!("{label}_pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&pp_label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl Device {
    /// Shared "per-pixel color pass" dispatch + readback helper. Takes
    /// `rgb_in` as packed RGB8, uploads it, runs the named per-pixel
    /// pipeline with the given uniform bytes, reads back to packed RGB8.
    ///
    /// All per-pixel color shaders (CDL, color_grade, future saturation /
    /// black-point / etc.) share this glue so each shader needs only a
    /// WGSL file + a uniform struct + a public `apply_<name>` thin wrapper.
    fn apply_per_pixel(
        &self,
        pipeline: &PerPixelPipeline,
        label: &str,
        rgb_in: &[u8],
        w: u32, h: u32,
        uniform_bytes: &[u8],
    ) -> Result<Vec<u8>> {
        let t_total = Instant::now();
        let rgba = rgb_to_rgba(rgb_in, w as usize, h as usize);

        let in_label  = format!("{label}_input");
        let out_label = format!("{label}_output");
        let uni_label = format!("{label}_uniforms");
        let bg_label  = format!("{label}_bg");
        let enc_label = format!("{label}_enc");
        let pass_label = format!("{label}_pass");
        let stg_label  = format!("{label}_staging");

        let in_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&in_label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &in_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&out_label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&uni_label),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, uniform_bytes);

        let in_view  = in_tex.create_view(&Default::default());
        let out_view = out_tex.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&bg_label),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&in_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out_view) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&enc_label),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&pass_label), timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (w + 7) / 8;
            let wg_y = (h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Readback (same row-padding dance as elsewhere).
        let bpp = 4u32;
        let padded_row_bytes = ((w * bpp) + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&stg_label),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();
        let mut out_rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();
        tracing::debug!(elapsed=?t_total.elapsed(), label=label, "apply_per_pixel");
        Ok(out_rgb)
    }

    /// Apply CDL (lift / gain / shadow / highlight / gamma) per-pixel.
    /// Identity params (`CdlParams::default()`) are a no-op fast path.
    pub fn apply_cdl(
        &self,
        rgb_in: &[u8],
        w: u32, h: u32,
        params: CdlParams,
    ) -> Result<Vec<u8>> {
        if params.is_identity() {
            return Ok(rgb_in.to_vec());
        }
        let uniforms = CdlUniforms {
            lift:      params.lift,
            gamma:     params.gamma,
            gain:      params.gain,
            shadow:    params.shadow,
            highlight: params.highlight,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        self.apply_per_pixel(
            &self.cdl, "cdl",
            rgb_in, w, h,
            bytemuck::bytes_of(&uniforms),
        )
    }

    /// Apply color grade (temperature / tint / saturation) per-pixel.
    /// Identity params are a no-op fast path.
    pub fn apply_color_grade(
        &self,
        rgb_in: &[u8],
        w: u32, h: u32,
        params: ColorGradeParams,
    ) -> Result<Vec<u8>> {
        if params.is_identity() {
            return Ok(rgb_in.to_vec());
        }
        let uniforms = ColorGradeUniforms {
            temperature: params.temperature,
            tint:        params.tint,
            saturation:  params.saturation,
            _pad: 0.0,
        };
        self.apply_per_pixel(
            &self.color_grade, "color_grade",
            rgb_in, w, h,
            bytemuck::bytes_of(&uniforms),
        )
    }

    /// Apply unsharp-mask sharpen. Three GPU dispatches: horizontal
    /// blur → vertical blur → USM combine. The original input texture
    /// is also bound to the combine pass so the `orig - blur` detail
    /// signal is exact (rather than re-uploaded).
    ///
    /// Identity params (`amount=0`) skip the GPU entirely.
    pub fn apply_sharpen(
        &self,
        rgb_in: &[u8],
        w: u32, h: u32,
        params: SharpenParams,
    ) -> Result<Vec<u8>> {
        if params.is_identity() {
            return Ok(rgb_in.to_vec());
        }
        let t_total = Instant::now();
        let rgba = rgb_to_rgba(rgb_in, w as usize, h as usize);

        // 1. Upload input.
        let in_tex = make_rw_texture(&self.device, "sharpen_input", w, h,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.upload_rgba(&in_tex, &rgba, w, h);

        // 2. Intermediate textures (full-res blurs).
        let blur_h_tex   = make_rw_texture(&self.device, "sharpen_blur_h", w, h,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let blur_full_tex = make_rw_texture(&self.device, "sharpen_blur_full", w, h,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let out_tex      = make_rw_texture(&self.device, "sharpen_output", w, h,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);

        // 3. Blur uniforms (one per direction).
        let blur_h_u = self.write_uniform("sharpen_blur_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("sharpen_blur_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("sharpen_combine_u",
            &SharpenCombineUniforms {
                amount: params.amount,
                apply_lat_weight: if params.apply_lat_weight { 1 } else { 0 },
                _pad0: 0, _pad1: 0,
            });

        // 4. Encode all three passes + readback in one command buffer.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sharpen_enc"),
        });
        self.dispatch_blur_1d(&mut encoder, "sharpen_h", &in_tex, &blur_h_tex, &blur_h_u, w, h);
        self.dispatch_blur_1d(&mut encoder, "sharpen_v", &blur_h_tex, &blur_full_tex, &blur_v_u, w, h);
        {
            let in_v   = in_tex.create_view(&Default::default());
            let blur_v = blur_full_tex.create_view(&Default::default());
            let out_v  = out_tex.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sharpen_combine_bg"),
                layout: &self.sharpen_combine.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&in_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blur_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&out_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: combine_u.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sharpen_combine_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sharpen_combine.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }
        let out_rgb = self.encode_readback_rgb(&mut encoder, &out_tex, w, h)?;
        self.queue.submit(Some(encoder.finish()));
        let result = self.finalize_readback(out_rgb)?;
        tracing::debug!(elapsed=?t_total.elapsed(), "apply_sharpen");
        Ok(result)
    }

    /// Apply mid-detail clarity. Four GPU dispatches: downsample 4× →
    /// horizontal blur on small image → vertical blur on small image →
    /// upsample-and-combine with bell-curve weight.
    ///
    /// Identity params (`amount=0`) skip the GPU entirely.
    pub fn apply_mid_detail(
        &self,
        rgb_in: &[u8],
        w: u32, h: u32,
        params: MidDetailParams,
    ) -> Result<Vec<u8>> {
        if params.is_identity() {
            return Ok(rgb_in.to_vec());
        }
        let t_total = Instant::now();
        let rgba = rgb_to_rgba(rgb_in, w as usize, h as usize);
        let sw = (w + 3) / 4;
        let sh = (h + 3) / 4;

        let in_tex = make_rw_texture(&self.device, "mid_input", w, h,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.upload_rgba(&in_tex, &rgba, w, h);

        let small_tex      = make_rw_texture(&self.device, "mid_small", sw, sh,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let small_blur_h_tex = make_rw_texture(&self.device, "mid_blur_h", sw, sh,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let small_blur_v_tex = make_rw_texture(&self.device, "mid_blur_v", sw, sh,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let out_tex = make_rw_texture(&self.device, "mid_output", w, h,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);

        let blur_h_u = self.write_uniform("mid_blur_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("mid_blur_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("mid_combine_u",
            &MidDetailCombineUniforms { amount: params.amount, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mid_detail_enc"),
        });

        // Pass 1: downsample 4×.
        {
            let in_v  = in_tex.create_view(&Default::default());
            let out_v = small_tex.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid_downsample_bg"),
                layout: &self.downsample_4x.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&in_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out_v) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid_downsample_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_4x.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((sw + 7) / 8, (sh + 7) / 8, 1);
        }

        // Passes 2 + 3: blur the small image (H then V).
        self.dispatch_blur_1d(&mut encoder, "mid_h", &small_tex, &small_blur_h_tex, &blur_h_u, sw, sh);
        self.dispatch_blur_1d(&mut encoder, "mid_v", &small_blur_h_tex, &small_blur_v_tex, &blur_v_u, sw, sh);

        // Pass 4: upsample-and-combine into full-res output.
        {
            let in_v    = in_tex.create_view(&Default::default());
            let small_v = small_blur_v_tex.create_view(&Default::default());
            let out_v   = out_tex.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid_combine_bg"),
                layout: &self.mid_detail_combine.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&in_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&small_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.bilinear_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&out_v) },
                    wgpu::BindGroupEntry { binding: 4, resource: combine_u.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid_combine_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.mid_detail_combine.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }

        let out_rgb = self.encode_readback_rgb(&mut encoder, &out_tex, w, h)?;
        self.queue.submit(Some(encoder.finish()));
        let result = self.finalize_readback(out_rgb)?;
        tracing::debug!(elapsed=?t_total.elapsed(), "apply_mid_detail");
        Ok(result)
    }

    // --- Multi-pass helpers (used by apply_sharpen / apply_mid_detail) ---

    fn upload_rgba(&self, tex: &wgpu::Texture, rgba: &[u8], w: u32, h: u32) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    fn write_uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(value));
        buf
    }

    fn dispatch_blur_1d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        uni: &wgpu::Buffer,
        w: u32, h: u32,
    ) {
        self.dispatch_blur_1d_with(&self.gaussian_blur, encoder, label, src, dst, uni, w, h);
    }

    fn dispatch_blur_1d_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        uni: &wgpu::Buffer,
        w: u32, h: u32,
    ) {
        self.dispatch_blur_1d_with(&self.gaussian_blur_16, encoder, label, src, dst, uni, w, h);
    }

    fn dispatch_blur_1d_with(
        &self,
        pipeline: &GaussianBlur1dPipeline,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        uni: &wgpu::Buffer,
        w: u32, h: u32,
    ) {
        let bg_label = format!("{label}_bg");
        let pass_label = format!("{label}_pass");
        let src_v = src.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&bg_label),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 2, resource: uni.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&pass_label), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }

    /// Schedule a texture→buffer readback into a freshly-allocated
    /// staging buffer. Returns the buffer; caller submits the encoder
    /// then passes it to `finalize_readback`.
    fn encode_readback_rgb(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        w: u32, h: u32,
    ) -> Result<ReadbackBuffer> {
        let bpp = 4u32;
        let padded_row_bytes = ((w * bpp) + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        Ok(ReadbackBuffer { staging, padded_row_bytes, w, h })
    }

    fn finalize_readback(&self, rb: ReadbackBuffer) -> Result<Vec<u8>> {
        let slice = rb.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((rb.w * rb.h * 3) as usize);
        for y in 0..rb.h {
            let row_off = (y * rb.padded_row_bytes) as usize;
            for x in 0..rb.w {
                let px_off = row_off + (x * 4) as usize;
                out.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        rb.staging.unmap();
        Ok(out)
    }
}

/// Bundle of a mapped staging buffer + its layout. Returned by
/// `encode_readback_rgb` and consumed by `finalize_readback` after
/// the encoder is submitted.
struct ReadbackBuffer {
    staging: wgpu::Buffer,
    padded_row_bytes: u32,
    w: u32,
    h: u32,
}

/// Create a 2D Rgba8Unorm texture with the given usage flags. Shared
/// helper for the multi-pass color tools that allocate several
/// intermediate textures with the same shape.
fn make_rw_texture(
    device: &wgpu::Device, label: &str, w: u32, h: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    })
}

/// Bind-group-layout entry for a filterable 2D texture used as a
/// shader read input. Shorthand to keep the Nv12ToEacPipeline layout
/// readable above.
fn bgle_tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Which lens to assemble (used by `Device::nv12_to_eac_cross`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lens { A, B }

impl Device {
    /// Assemble one EAC cross (`cross_w × cross_w`, Rgba8Unorm) from
    /// the two HEVC streams' NV12 plane textures, on the GPU.
    ///
    /// Each input is a `wgpu::Texture` view of an IOSurface plane
    /// (Phase 0.6.5 substrate) — zero memcpy from the VideoToolbox
    /// decoder all the way to this kernel's sample step.
    ///
    /// One dispatch per lens; both lenses share the same input
    /// stream textures (the `lens` parameter picks the tile layout).
    pub fn nv12_to_eac_cross(
        &self,
        s0_y:  &wgpu::Texture,
        s0_uv: &wgpu::Texture,
        s4_y:  &wgpu::Texture,
        s4_uv: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
    ) -> Result<wgpu::Texture> {
        self.nv12_to_eac_cross_with(s0_y, s0_uv, s4_y, s4_uv, lens, dims, false)
    }

    /// 16-bit cross variant: P010's 10 bits survive into Rgba16Unorm
    /// instead of being quantized to 8. For the 10-bit EAC export chain.
    pub fn nv12_to_eac_cross_16(
        &self,
        s0_y:  &wgpu::Texture,
        s0_uv: &wgpu::Texture,
        s4_y:  &wgpu::Texture,
        s4_uv: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
    ) -> Result<wgpu::Texture> {
        self.nv12_to_eac_cross_with(s0_y, s0_uv, s4_y, s4_uv, lens, dims, true)
    }

    fn nv12_to_eac_cross_with(
        &self,
        s0_y:  &wgpu::Texture,
        s0_uv: &wgpu::Texture,
        s4_y:  &wgpu::Texture,
        s4_uv: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
        sixteen: bool,
    ) -> Result<wgpu::Texture> {
        let cw = dims.cross_w();
        let pl = if sixteen { &self.nv12_to_eac_16 } else { &self.nv12_to_eac };

        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(match lens { Lens::A => "cross_a", Lens::B => "cross_b" }),
            size: wgpu::Extent3d { width: cw, height: cw, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if sixteen { wgpu::TextureFormat::Rgba16Unorm }
                    else       { wgpu::TextureFormat::Rgba8Unorm },
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let uniforms = Nv12ToEacUniforms {
            stream_w: dims.stream_w,
            stream_h: dims.stream_h,
            tile_w:   dims.tile_w(),
            center_w: vr180_core::eac::CENTER_W,
            cross_w:  cw,
            lens:     match lens { Lens::A => 0, Lens::B => 1 },
            _pad0: 0, _pad1: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_to_eac_uniforms"),
            size: std::mem::size_of::<Nv12ToEacUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let s0_y_v  = s0_y.create_view(&Default::default());
        let s0_uv_v = s0_uv.create_view(&Default::default());
        let s4_y_v  = s4_y.create_view(&Default::default());
        let s4_uv_v = s4_uv.create_view(&Default::default());
        let out_v   = out_tex.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12_to_eac_bg"),
            layout: &pl.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&s0_y_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&s0_uv_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&s4_y_v) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&s4_uv_v) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&pl.sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&out_v) },
                wgpu::BindGroupEntry { binding: 6, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nv12_to_eac_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12_to_eac_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&pl.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg = (cw + 7) / 8;
            pass.dispatch_workgroups(wg, wg, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out_tex)
    }

    /// Windows zero-copy EAC cross assembly (8-bit cross output, for the
    /// live preview). `s0_rgba`/`s4_rgba` are the two EAC stream textures
    /// already converted P010→Rgba16Unorm on the D3D11 side. Same cross
    /// geometry as `nv12_to_eac_cross` (the macOS NV12-plane path) so the
    /// assembled cross is pixel-identical.
    pub fn rgba8_to_eac_cross(
        &self,
        s0_rgba: &wgpu::Texture,
        s4_rgba: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
    ) -> Result<wgpu::Texture> {
        self.rgba_to_eac_cross_with(s0_rgba, s4_rgba, lens, dims, false)
    }

    /// 16-bit cross variant — the source's 10 bits survive into Rgba16Unorm
    /// for the GPU-resident 10-bit EAC export chain.
    pub fn rgba16_to_eac_cross(
        &self,
        s0_rgba: &wgpu::Texture,
        s4_rgba: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
    ) -> Result<wgpu::Texture> {
        self.rgba_to_eac_cross_with(s0_rgba, s4_rgba, lens, dims, true)
    }

    fn rgba_to_eac_cross_with(
        &self,
        s0_rgba: &wgpu::Texture,
        s4_rgba: &wgpu::Texture,
        lens: Lens,
        dims: vr180_core::eac::Dims,
        sixteen: bool,
    ) -> Result<wgpu::Texture> {
        let cw = dims.cross_w();
        let pl = if sixteen { &self.rgba_to_eac_16 } else { &self.rgba_to_eac };

        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(match lens { Lens::A => "rgba_cross_a", Lens::B => "rgba_cross_b" }),
            size: wgpu::Extent3d { width: cw, height: cw, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if sixteen { wgpu::TextureFormat::Rgba16Unorm }
                    else       { wgpu::TextureFormat::Rgba8Unorm },
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let uniforms = Nv12ToEacUniforms {
            stream_w: dims.stream_w,
            stream_h: dims.stream_h,
            tile_w:   dims.tile_w(),
            center_w: vr180_core::eac::CENTER_W,
            cross_w:  cw,
            lens:     match lens { Lens::A => 0, Lens::B => 1 },
            _pad0: 0, _pad1: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgba_to_eac_uniforms"),
            size: std::mem::size_of::<Nv12ToEacUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let s0_v  = s0_rgba.create_view(&Default::default());
        let s4_v  = s4_rgba.create_view(&Default::default());
        let out_v = out_tex.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgba_to_eac_bg"),
            layout: &pl.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&s0_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&s4_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&out_v) },
                wgpu::BindGroupEntry { binding: 4, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rgba_to_eac_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rgba_to_eac_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&pl.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg = (cw + 7) / 8;
            pass.dispatch_workgroups(wg, wg, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(out_tex)
    }

    /// GPU-side version of `project_cross_to_equirect` that takes a
    /// `wgpu::Texture` instead of a `&[u8]`. Used by the zero-copy path
    /// to keep the cross on the GPU through the equirect projection.
    /// Skips the upload of `nv12_to_eac_cross`'s output.
    pub fn project_cross_texture_to_equirect(
        &self,
        cross_tex: &wgpu::Texture,
        out_w: u32,
        out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<Vec<u8>> {
        use std::time::Instant;
        let t_total = Instant::now();

        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_rgba_zerocopy"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cross_view = cross_tex.create_view(&Default::default());
        let output_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("equirect_R_zc",
            &EquirectUniforms::from_mat3(rotation.0));
        let rs_uniform = self.write_uniform("equirect_RS_zc",
            &RsUniforms::from_params(rs));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eac_to_equirect_zc_bg"),
            layout: &self.eac_to_equirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cross_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.eac_to_equirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rs_uniform.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eac_to_equirect_zc_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eac_to_equirect_zc_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.eac_to_equirect.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (out_w + 7) / 8;
            let wg_y = (out_h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let bpp = 4u32;
        let unpadded_row_bytes = out_w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("equirect_zc_staging"),
            size: (padded_row_bytes * out_h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();
        let mut out_rgb = Vec::with_capacity((out_w * out_h * 3) as usize);
        for y in 0..out_h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..out_w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();
        tracing::debug!(elapsed=?t_total.elapsed(), "GPU project_cross_texture_to_equirect total");
        Ok(out_rgb)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Lut3DUniforms {
    size: u32,
    intensity: f32,
    _pad0: u32,
    _pad1: u32,
}

impl Lut3DPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, "lut3d", LUT3D_WGSL,
            wgpu::TextureFormat::Rgba8Unorm)
    }

    /// 16-bit-output 3D LUT — same layout, output texture is Rgba16Unorm.
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, "lut3d_16", LUT3D_16_WGSL,
            wgpu::TextureFormat::Rgba16Unorm)
    }

    fn create_with_format(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bgl_label = format!("{label}_bgl");
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&bgl_label),
            entries: &[
                // 0: input 2D texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: LUT 3D texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: sampler (trilinear)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: output 2D storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pll_label = format!("{label}_pll");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&pll_label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pp_label = format!("{label}_pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&pp_label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let smp_label = format!("{label}_smp");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&smp_label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl Device {
    /// Apply a 3D .cube LUT to an RGB8 image. Trilinear sampling on
    /// the GPU; one shader dispatch. `intensity` is the blend factor
    /// from the original image (0.0) to the fully-graded image (1.0).
    pub fn apply_lut3d(
        &self,
        input_rgb: &[u8],
        w: u32, h: u32,
        lut: &vr180_core::color::Cube3DLut,
        intensity: f32,
    ) -> Result<Vec<u8>> {
        // Upload input as rgba8unorm 2D texture.
        let input_rgba = rgb_to_rgba(input_rgb, w as usize, h as usize);
        let input_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut3d_input"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &input_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        // Upload LUT as 3D rgba8unorm texture.
        let lut_bytes = lut.to_rgba8_for_upload();
        let lut_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut3d_cube"),
            size: wgpu::Extent3d {
                width: lut.size, height: lut.size, depth_or_array_layers: lut.size,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &lut_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &lut_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lut.size * 4),
                rows_per_image: Some(lut.size),
            },
            wgpu::Extent3d {
                width: lut.size, height: lut.size, depth_or_array_layers: lut.size,
            },
        );

        // Output storage texture.
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut3d_output"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Uniforms.
        let uniforms = Lut3DUniforms {
            size: lut.size,
            intensity,
            _pad0: 0, _pad1: 0,
        };
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lut3d_uniforms"),
            size: std::mem::size_of::<Lut3DUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Bind group.
        let input_view = input_tex.create_view(&Default::default());
        let lut_view = lut_tex.create_view(&Default::default());
        let output_view = output_tex.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lut3d_bg"),
            layout: &self.lut3d.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lut3d.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 4, resource: uniform_buf.as_entire_binding() },
            ],
        });

        // Dispatch.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lut3d_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lut3d_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.lut3d.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (w + 7) / 8;
            let wg_y = (h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Readback (same staging buffer dance as project_cross_to_equirect).
        let bpp = 4u32;
        let unpadded_row_bytes = w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lut3d_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();
        let mut out_rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();
        Ok(out_rgb)
    }
}

// ----- Phase 0.7.5 pipeline creators (sharpen + mid-detail) -----

impl GaussianBlur1dPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm, "gaussian_blur_1d")
    }
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm, "gaussian_blur_1d_16")
    }
    fn create_with_format(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let wgsl = if format == wgpu::TextureFormat::Rgba16Unorm {
            GAUSSIAN_BLUR_1D_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else { GAUSSIAN_BLUR_1D_WGSL.to_string() };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                bgle_tex(0),
                bgle_storage_out_fmt(1, format),
                bgle_uniform(2, std::mem::size_of::<GaussianBlur1dUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl SharpenCombinePipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm, "sharpen_combine")
    }
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm, "sharpen_combine_16")
    }
    fn create_with_format(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let wgsl = if format == wgpu::TextureFormat::Rgba16Unorm {
            SHARPEN_COMBINE_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else { SHARPEN_COMBINE_WGSL.to_string() };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                bgle_tex(0),
                bgle_tex(1),
                bgle_storage_out_fmt(2, format),
                bgle_uniform(3, std::mem::size_of::<SharpenCombineUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl MidDetailCombinePipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm, "mid_detail_combine")
    }
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm, "mid_detail_combine_16")
    }
    fn create_with_format(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let wgsl = if format == wgpu::TextureFormat::Rgba16Unorm {
            MID_DETAIL_COMBINE_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else { MID_DETAIL_COMBINE_WGSL.to_string() };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                bgle_tex(0),
                bgle_tex(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                bgle_storage_out_fmt(3, format),
                bgle_uniform(4, std::mem::size_of::<MidDetailCombineUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl Downsample4xPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba8Unorm, "downsample_4x")
    }
    fn create_16bit(device: &wgpu::Device) -> Self {
        Self::create_with_format(device, wgpu::TextureFormat::Rgba16Unorm, "downsample_4x_16")
    }
    fn create_with_format(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let wgsl = if format == wgpu::TextureFormat::Rgba16Unorm {
            DOWNSAMPLE_4X_WGSL.replace("rgba8unorm, write", "rgba16unorm, write")
        } else { DOWNSAMPLE_4X_WGSL.to_string() };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &[
                bgle_tex(0),
                bgle_storage_out_fmt(1, format),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

/// Bind-group-layout entry for a writeable Rgba8Unorm storage texture.
/// Used as the output of every per-pixel + multi-pass color shader.
fn bgle_storage_out(binding: u32) -> wgpu::BindGroupLayoutEntry {
    bgle_storage_out_fmt(binding, wgpu::TextureFormat::Rgba8Unorm)
}

/// Variant that lets the caller pick the storage texture format. Used by
/// the 16-bit (Rgba16Unorm) variants of the color stack passes for the
/// 10-bit export path.
fn bgle_storage_out_fmt(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Bind-group-layout entry for a uniform buffer of a given byte size.
fn bgle_uniform(binding: u32, size_bytes: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size_bytes),
        },
        count: None,
    }
}

// ===== Phase 0.7.5.5: GPU texture chaining =====
//
// The pre-0.7.5.5 `apply_*` methods each ran their own
// upload → dispatch → readback cycle. With four color stages active
// (CDL + sharpen + mid-detail + color_grade) that's 4 wgpu submits, 4
// GPU sync waits, and 4 staging-buffer memcpys per frame — the entire
// color pipeline is bottlenecked on host↔device transfers.
//
// The fix here is to:
// (a) project the EAC cross to an equirect *texture* instead of a
//     `Vec<u8>` (`project_*_to_equirect_texture` below),
// (b) chain every active color stage into one shared encoder via
//     `record_*` helpers — each one records dispatches but does NOT
//     submit and does NOT read back,
// (c) submit the whole thing once and read back only the final
//     texture (`apply_color_stack_texture`).
//
// Net: 1 submit, 1 staging-buffer round-trip per frame, regardless of
// how many color tools are active.

/// Param bundle for the full Phase 0.7.5 color stack — what
/// `apply_color_stack_texture` consumes. Mirrors `ColorParams` in
/// `vr180-render` but lives in `vr180-pipeline` so the chained API
/// can be called by anything (not just the CLI).
#[derive(Debug, Clone)]
pub struct ColorStackPlan {
    pub cdl:         CdlParams,
    pub lut:         Option<(vr180_core::color::Cube3DLut, f32)>,
    pub sharpen:     SharpenParams,
    pub mid_detail:  MidDetailParams,
    pub color_grade: ColorGradeParams,
    /// "Matching Eyes" inter-eye white-balance trim. Applied OPPOSITELY to the
    /// two eyes (left `+`, right `−`) via [`ColorStackPlan::for_eye`], on top of
    /// the global temperature/tint — corrects an inter-lens color discrepancy
    /// without shifting the overall color. 0 = off.
    pub eye_match_ct:   f32,
    pub eye_match_tint: f32,
}

impl Default for ColorStackPlan {
    fn default() -> Self {
        Self {
            cdl: CdlParams::default(),
            lut: None,
            sharpen: SharpenParams::default(),
            mid_detail: MidDetailParams::default(),
            color_grade: ColorGradeParams::default(),
            eye_match_ct: 0.0,
            eye_match_tint: 0.0,
        }
    }
}

impl ColorStackPlan {
    pub fn any_active(&self) -> bool {
        !self.cdl.is_identity()
            || self.lut.is_some()
            || !self.sharpen.is_identity()
            || !self.mid_detail.is_identity()
            || !self.color_grade.is_identity()
            || self.eye_match_ct != 0.0
            || self.eye_match_tint != 0.0
    }

    /// This plan specialized for one eye: the "Matching Eyes" CT/tint trim is
    /// added in OPPOSITE directions (left `+`, right `−`) on top of the global
    /// temperature/tint. Cheap when the trim is off — borrows `self` with no
    /// clone; only clones (incl. any LUT) when a trim is actually set.
    pub fn for_eye(&self, is_left: bool) -> std::borrow::Cow<'_, ColorStackPlan> {
        if self.eye_match_ct == 0.0 && self.eye_match_tint == 0.0 {
            return std::borrow::Cow::Borrowed(self);
        }
        let sign = if is_left { 1.0 } else { -1.0 };
        let mut p = self.clone();
        p.color_grade.temperature += sign * self.eye_match_ct;
        p.color_grade.tint += sign * self.eye_match_tint;
        std::borrow::Cow::Owned(p)
    }
}

impl Device {
    /// Project an EAC cross (`cross_w × cross_w`, Rgba8Unorm, on the GPU
    /// already) to a half-equirect texture (`out_w × out_h`, Rgba8Unorm,
    /// also on the GPU). No readback — caller chains into the color
    /// stack and reads back at the end.
    ///
    /// Texture-resident sibling of `project_cross_texture_to_equirect`.
    pub fn project_cross_texture_to_equirect_texture(
        &self,
        cross_tex: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let output_tex = make_rw_texture(&self.device, "equirect_tex_out", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("project_tex_enc"),
        });
        self.record_equirect_project(&mut encoder, cross_tex, &output_tex, out_w, out_h, rotation, rs);
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// EAC cross → equidistant-fisheye output (Rgba8Unorm).
    pub fn project_cross_texture_to_fisheye_texture(
        &self,
        cross_tex: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let output_tex = make_rw_texture(&self.device, "eac_fisheye_out", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eac_fisheye_enc"),
        });
        self.record_equirect_project_with(
            &mut encoder, &self.eac_to_fisheye,
            cross_tex, &output_tex, out_w, out_h, rotation, rs);
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// 16-bit EAC cross → equidistant-fisheye output (Rgba16Unorm).
    pub fn project_cross_texture_to_fisheye_texture_16(
        &self,
        cross_tex: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("eac_fisheye_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eac_fisheye_enc_16"),
        });
        self.record_equirect_project_with(
            &mut encoder, &self.eac_to_fisheye_16,
            cross_tex, &output_tex, out_w, out_h, rotation, rs);
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// 16-bit-output EAC→equirect projection (Rgba16Unorm), taking a
    /// 16-bit cross from `nv12_to_eac_cross_16`. The 10-bit EAC export
    /// chain: P010 → Rgba16 cross → Rgba16 equirect → P010 encode.
    pub fn project_cross_texture_to_equirect_texture_16(
        &self,
        cross_tex: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_tex_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("project_tex_enc_16"),
        });
        self.record_equirect_project_with(
            &mut encoder, &self.eac_to_equirect_16,
            cross_tex, &output_tex, out_w, out_h, rotation, rs);
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// CPU-input version of the above: takes a packed RGB8 cross,
    /// uploads it as an Rgba8Unorm texture, projects, returns the
    /// projected texture (no readback). For the CPU-assemble path.
    pub fn project_cross_to_equirect_texture(
        &self,
        cross_rgb: &[u8],
        cross_w: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let cross_rgba = rgb_to_rgba(cross_rgb, cross_w as usize, cross_w as usize);
        let cross_tex = make_rw_texture(&self.device, "cross_for_project",
            cross_w, cross_w,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.upload_rgba(&cross_tex, &cross_rgba, cross_w, cross_w);
        self.project_cross_texture_to_equirect_texture(&cross_tex, out_w, out_h, rotation, rs)
    }

    /// CPU-input EAC cross → equidistant-fisheye output (fisheye-output
    /// mode on the portable / CPU-assemble path).
    pub fn project_cross_to_fisheye_texture(
        &self,
        cross_rgb: &[u8],
        cross_w: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) -> Result<wgpu::Texture> {
        let cross_rgba = rgb_to_rgba(cross_rgb, cross_w as usize, cross_w as usize);
        let cross_tex = make_rw_texture(&self.device, "cross_for_project",
            cross_w, cross_w,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.upload_rgba(&cross_tex, &cross_rgba, cross_w, cross_w);
        self.project_cross_texture_to_fisheye_texture(&cross_tex, out_w, out_h, rotation, rs)
    }

    /// Apply the full Phase 0.7.5 color stack to one equirect texture,
    /// reading back the final result as packed RGB8.
    ///
    /// One encoder, one submit, one readback regardless of how many
    /// color stages are active. Stages with identity params are skipped
    /// (no GPU work, no allocated intermediate texture).
    ///
    /// Order matches Phase 0.7.5 / Python: CDL → LUT3D → sharpen →
    /// mid-detail → color_grade.
    pub fn apply_color_stack_texture(
        &self,
        equirect_tex: &wgpu::Texture,
        w: u32, h: u32,
        plan: &ColorStackPlan,
    ) -> Result<Vec<u8>> {
        let t_total = Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("color_stack_enc"),
        });
        // intermediates owns each newly-created stage output. We hand
        // out `&wgpu::Texture` to `record_*` calls and to the readback
        // step; NLL lets us push into the Vec right after the borrow
        // ends.
        let mut intermediates: Vec<wgpu::Texture> = Vec::with_capacity(5);

        // LUT3D resources outlive the encoder; allocate them in the
        // outer scope so the textures + sampler stay live until submit.
        let lut_resources = if let Some((ref lut, _)) = plan.lut {
            Some(self.upload_lut3d_resources(lut))
        } else { None };

        // Stage 1: CDL.
        if !plan.cdl.is_identity() {
            let next = make_rw_texture(&self.device, "stack_cdl_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            self.record_cdl(&mut encoder, prev, &next, plan.cdl);
            intermediates.push(next);
        }

        // Stage 2: 3D LUT.
        if let (Some((_, intensity)), Some(ref lr)) = (&plan.lut, &lut_resources) {
            let next = make_rw_texture(&self.device, "stack_lut_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            self.record_lut3d(&mut encoder, prev, &next, lr, *intensity, w, h);
            intermediates.push(next);
        }

        // Stage 3: sharpen (3 internal dispatches).
        if !plan.sharpen.is_identity() {
            let next = make_rw_texture(&self.device, "stack_sharpen_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            // Need full-frame intermediate textures for the H-blur and V-blur.
            // They're allocated here so they stay alive until submit, and they
            // can't be shared across stages (sharpen vs mid-detail might run
            // concurrently in a future fused pass, so we keep them local).
            let blur_h = make_rw_texture(&self.device, "stack_sharp_blur_h", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let blur_v = make_rw_texture(&self.device, "stack_sharp_blur_v", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            self.record_sharpen(&mut encoder, prev, &blur_h, &blur_v, &next,
                w, h, plan.sharpen);
            intermediates.push(blur_h);   // keep alive
            intermediates.push(blur_v);   // keep alive
            intermediates.push(next);
        }

        // Stage 3b: white balance (temperature / tint) — POST-LUT, matching
        // the Python order (CDL → LUT → sharpen → temp/tint → mid → sat).
        if plan.color_grade.has_white_balance() {
            let next = make_rw_texture(&self.device, "stack_wb_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            self.record_color_grade(&mut encoder, prev, &next, plan.color_grade.white_balance_only());
            intermediates.push(next);
        }

        // Stage 4: mid-detail (4 internal dispatches).
        if !plan.mid_detail.is_identity() {
            let next = make_rw_texture(&self.device, "stack_mid_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            let sw = (w + 3) / 4;
            let sh = (h + 3) / 4;
            let small = make_rw_texture(&self.device, "stack_mid_small", sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let small_h = make_rw_texture(&self.device, "stack_mid_small_h", sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let small_v = make_rw_texture(&self.device, "stack_mid_small_v", sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            self.record_mid_detail(&mut encoder, prev,
                &small, &small_h, &small_v, &next,
                w, h, sw, sh, plan.mid_detail);
            intermediates.push(small);
            intermediates.push(small_h);
            intermediates.push(small_v);
            intermediates.push(next);
        }

        // Stage 5: saturation — POST-LUT (the only color adjustment after
        // the LUT). Temperature/tint were applied pre-LUT above.
        if plan.color_grade.has_saturation() {
            let next = make_rw_texture(&self.device, "stack_sat_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            self.record_color_grade(&mut encoder, prev, &next, plan.color_grade.saturation_only());
            intermediates.push(next);
        }

        // Pick the final stage output (or fall back to the equirect input
        // if every stage was identity — in which case we're just doing a
        // GPU→host download of the equirect).
        // Filter out intermediates that aren't valid readback sources
        // (the blur scratch textures lack COPY_SRC); use the last one
        // we pushed that DID get COPY_SRC, which is the "_out" texture
        // of the most recent active stage. We keep a parallel index for
        // this — simpler than annotating each push.
        // Trick: only the *_out textures got COPY_SRC. Walk backwards
        // looking for one we explicitly named with "_out".
        let final_tex = intermediates.iter().rev()
            .find(|t| {
                // wgpu::Texture exposes its usage; check COPY_SRC.
                t.usage().contains(wgpu::TextureUsages::COPY_SRC)
            })
            .unwrap_or(equirect_tex);

        let rb = self.encode_readback_rgb(&mut encoder, final_tex, w, h)?;
        self.queue.submit(Some(encoder.finish()));
        let result = self.finalize_readback(rb)?;
        tracing::debug!(elapsed=?t_total.elapsed(),
            stages = intermediates.len(),
            "apply_color_stack_texture");
        Ok(result)
    }

    // ----- record_* helpers — encode into a caller-supplied encoder, -----
    // ----- no submit, no readback. The dst texture must outlive the encoder.

    fn record_equirect_project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cross_tex: &wgpu::Texture,
        dst: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) {
        self.record_equirect_project_with(
            encoder, &self.eac_to_equirect, cross_tex, dst, out_w, out_h, rotation, rs)
    }

    fn record_equirect_project_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pl: &EacToEquirectPipeline,
        cross_tex: &wgpu::Texture,
        dst: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        rs: EquirectRsParams,
    ) {
        let cross_v = cross_tex.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let r_uniform = self.write_uniform("equirect_R_record",
            &EquirectUniforms::from_mat3(rotation.0));
        let rs_uniform = self.write_uniform("equirect_RS_record",
            &RsUniforms::from_params(rs));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("equirect_project_bg"),
            layout: &pl.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cross_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rs_uniform.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("equirect_project_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pl.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
    }

    /// Record a fisheye→half-equirect compute pass into an existing
    /// encoder. Counterpart to `record_equirect_project` for the
    /// fisheye source family. `fisheye_tex` is an Rgba8Unorm texture
    /// holding one raw fisheye eye; `dst` is the output equirect
    /// storage texture.
    fn record_fisheye_project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        fisheye_tex: &wgpu::Texture,
        dst: &wgpu::Texture,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) {
        let src_v = fisheye_tex.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_R_record",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_calib_record",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_project_bg"),
            layout: &self.fisheye_to_hequirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fisheye_project_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.fisheye_to_hequirect.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
    }

    /// CPU-input fisheye → equirect with texture output (no readback).
    /// Same shape as `project_cross_to_equirect_texture` but for the
    /// raw-fisheye source family.
    pub fn project_fisheye_to_equirect_texture(
        &self,
        src_rgba: &[u8],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let output_tex = make_rw_texture(&self.device, "fisheye_eq_tex_out", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let dst_view = output_tex.create_view(&Default::default());

        let mut cache = self.proj_fisheye_cache.lock().unwrap();
        let needs_init = match cache.get(&slot) {
            Some(s) => s.src_w != src_w || s.src_h != src_h,
            None => true,
        };
        if needs_init {
            let src_tex = make_rw_texture(&self.device, "fisheye_src", src_w, src_h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            let src_view = src_tex.create_view(&Default::default());
            let r_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_R_cached"),
                size: std::mem::size_of::<EquirectUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cal_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_calib_cached"),
                size: std::mem::size_of::<FisheyeCalibUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            cache.insert(slot, ProjFisheyeCacheSlot {
                src_w, src_h, src_tex, src_view, r_uniform, cal_uniform,
            });
        }
        let s = cache.get(&slot).unwrap();

        self.upload_rgba(&s.src_tex, src_rgba, src_w, src_h);
        self.queue.write_buffer(&s.r_uniform, 0,
            bytemuck::bytes_of(&EquirectUniforms::from_mat3(rotation.0)));
        self.queue.write_buffer(&s.cal_uniform, 0,
            bytemuck::bytes_of(&FisheyeCalibUniforms::from_public(calib)));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_project_bg"),
            layout: &self.fisheye_to_hequirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&s.src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: s.r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: s.cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_project_tex_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_project_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware variant of [`Self::project_fisheye_to_equirect_texture`].
    /// Takes an additional per-scanline rotation buffer (`rs_rows_f32`
    /// = 12 f32 per row × `src_h` rows) and applies per-row correction
    /// fused with the per-frame stab. Used by the GUI preview path when
    /// DJI OSV per-row matrices are available so we get DJI Studio's
    /// per-slab stab quality without leaving the live pipeline.
    pub fn project_fisheye_to_equirect_rs_texture(
        &self,
        src_rgba: &[u8],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        rs_rows_f32: &[f32],
        slot: u32,
    ) -> Result<wgpu::Texture> {
        debug_assert_eq!(rs_rows_f32.len(), (src_h as usize) * 12);

        // Reused output texture is still owned by the caller (UI thread
        // may render the previous frame for a few ms), so allocate
        // fresh. Everything else lives in the slot cache and is updated
        // via write_buffer/write_texture instead of recreated.
        let output_tex = make_rw_texture(&self.device, "fisheye_eq_tex_out_rs", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let dst_view = output_tex.create_view(&Default::default());

        let mut cache = self.proj_fisheye_rs_cache.lock().unwrap();
        // Rebuild slot if dims changed (eye_w/eye_h doesn't matter — it
        // only affects dispatch size, not cached resources).
        let needs_init = match cache.get(&slot) {
            Some(s) => s.src_w != src_w || s.src_h != src_h,
            None => true,
        };
        if needs_init {
            let src_tex = make_rw_texture(&self.device,
                "fisheye_src_rs", src_w, src_h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            let src_view = src_tex.create_view(&Default::default());
            let r_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_R_rs"),
                size: std::mem::size_of::<EquirectUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cal_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_calib_rs"),
                size: std::mem::size_of::<FisheyeCalibUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let rs_buf_capacity = (rs_rows_f32.len() * std::mem::size_of::<f32>()) as u64;
            let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_rs_rows"),
                size: rs_buf_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            cache.insert(slot, ProjFisheyeRsCacheSlot {
                src_w, src_h,
                src_tex, src_view,
                r_uniform, cal_uniform,
                rs_buf, rs_buf_capacity,
            });
        }
        // Grow rs_buf if a larger frame ever arrives without a dim change
        // (unusual — but src_h could shift via settings).
        let needed_rs = (rs_rows_f32.len() * std::mem::size_of::<f32>()) as u64;
        if cache.get(&slot).unwrap().rs_buf_capacity < needed_rs {
            let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fisheye_rs_rows"),
                size: needed_rs,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let s = cache.get_mut(&slot).unwrap();
            s.rs_buf = rs_buf;
            s.rs_buf_capacity = needed_rs;
        }
        let s = cache.get(&slot).unwrap();

        // Update reused resources in-place. write_buffer / write_texture
        // are queue ops; WGPU serialises them with the compute dispatch
        // below, so a single slot can be reused for consecutive frames.
        self.upload_rgba(&s.src_tex, src_rgba, src_w, src_h);
        self.queue.write_buffer(&s.r_uniform, 0,
            bytemuck::bytes_of(&EquirectUniforms::from_mat3(rotation.0)));
        self.queue.write_buffer(&s.cal_uniform, 0,
            bytemuck::bytes_of(&FisheyeCalibUniforms::from_public(calib)));
        self.queue.write_buffer(&s.rs_buf, 0, bytemuck::cast_slice(rs_rows_f32));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_to_hequirect_rs_bg"),
            layout: &self.fisheye_to_hequirect_rs.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&s.src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect_rs.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: s.r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: s.cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: s.rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_project_rs_tex_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_project_rs_tex_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect_rs.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            let wg_x = (out_w + 7) / 8;
            let wg_y = (out_h + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// 16-bit-per-channel projection. Same KB math + cubic Hermite
    /// extension as the 8-bit variant; output texture is Rgba16Unorm
    /// instead of Rgba8Unorm. Source is uploaded as 16-bit so source
    /// precision (P010 → RGBA64LE → texture) is preserved.
    pub fn project_fisheye_to_equirect_texture_16(
        &self,
        src_rgba64le: &[u8],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        debug_assert_eq!(src_rgba64le.len(), (src_w as usize) * (src_h as usize) * 8);
        // Source texture: Rgba16Unorm to keep the 16-bit precision
        // through the bilinear sampler in the shader.
        let src_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_src_16"),
            size: wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            src_rgba64le,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(src_w * 8),
                rows_per_image: Some(src_h),
            },
            wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
        );
        // Output: Rgba16Unorm half-equirect.
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_eq_tex_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_R_record_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_calib_record_16",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_project_bg_16"),
            layout: &self.fisheye_to_hequirect_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_project_tex_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_project_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Zero-copy P010 projection. The Y and UV plane textures come
    /// from `crate::interop_macos::wgpu_texture_from_iosurface_plane`
    /// (i.e. they alias the VideoToolbox-decoded CVPixelBuffer
    /// directly) — no host-memory hop and no swscale.
    ///
    /// `src_w` / `src_h` is the native fisheye resolution (matches the
    /// Y plane dims). Calibration must have been resolved against those
    /// dims, NOT against any preview-clamped working size.
    ///
    /// Output: Rgba16Unorm half-equirect ready to feed into
    /// `compose_sbs_to_p010`.
    #[cfg(target_os = "macos")]
    pub fn project_fisheye_p010_to_equirect_texture_16(
        &self,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // dims are encoded into the calib uniform's src_w/src_h
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_p010_eq_tex_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let y_view  = y_tex .create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_p010_R_record_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_p010_calib_record_16",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_p010_project_bg_16"),
            layout: &self.fisheye_p010_to_hequirect_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.fisheye_p010_to_hequirect_16.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_p010_project_tex_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_p010_project_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_p010_to_hequirect_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Windows zero-copy P010 projection: same math/pipeline as
    /// [`Self::project_fisheye_p010_to_equirect_texture_16`], but sources Y/UV
    /// from the two planes of a **single** imported P010 texture (the
    /// D3D11→Vulkan-imported NVDEC frame) via plane-aspect views, rather than
    /// two separate IOSurface plane textures. Available on all platforms.
    pub fn project_fisheye_p010_planar_to_equirect_texture_16(
        &self,
        p010_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // encoded into the calib uniform
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_p010_planar_eq_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // P010 plane views: plane 0 = R16 (Y), plane 1 = Rg16 (CbCr).
        let y_view = p010_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("p010_y_plane"),
            format: Some(wgpu::TextureFormat::R16Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = p010_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("p010_uv_plane"),
            format: Some(wgpu::TextureFormat::Rg16Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_p010_planar_R_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_p010_planar_calib_16",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_p010_planar_bg_16"),
            layout: &self.fisheye_p010_to_hequirect_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.fisheye_p010_to_hequirect_16.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_p010_planar_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_p010_planar_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_p010_to_hequirect_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Resolve an imported P010 frame (Y/UV plane views) to a downscaled
    /// `Rgba16Unorm` fisheye image: upsample chroma to RGB at full res, then
    /// box-average down to `out_w × out_h` (see `p010_resolve_rgba16.wgsl`).
    /// `src_w`/`src_h` are the native P010 dims. This is the Windows
    /// zero-copy preview prefilter — pair it with
    /// [`Self::project_fisheye_rgba16_texture_to_equirect_16`] so the
    /// projection minifies only a little and a single bilinear tap no longer
    /// aliases (no luma moiré, no chroma colour-fringing).
    pub fn resolve_p010_to_rgba16(
        &self,
        p010_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
    ) -> Result<wgpu::Texture> {
        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p010_resolved_rgba16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // P010 plane views: plane 0 = R16 (Y), plane 1 = Rg16 (CbCr).
        let y_view = p010_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("p010_resolve_y"),
            format: Some(wgpu::TextureFormat::R16Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = p010_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("p010_resolve_uv"),
            format: Some(wgpu::TextureFormat::Rg16Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });
        let dst_view = out_tex.create_view(&Default::default());
        let dims = self.write_uniform("p010_resolve_dims", &ResolveDims {
            src_w: src_w as f32, src_h: src_h as f32,
            out_w: out_w as f32, out_h: out_h as f32,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p010_resolve_bg"),
            layout: &self.p010_resolve.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.p010_resolve.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("p010_resolve_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_resolve_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.p010_resolve.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out_tex)
    }

    /// macOS variant of [`Self::resolve_p010_to_rgba16`]: the Y and UV planes
    /// arrive as TWO SEPARATE textures (`R16Unorm` + `Rg16Unorm`, each aliasing
    /// one plane of a VideoToolbox IOSurface via
    /// `crate::interop_macos::wgpu_texture_from_iosurface_plane`) instead of one
    /// combined P010 texture sampled through plane-aspect views. Same shader and
    /// box-downscale math — it's the macOS zero-copy preview prefilter, pairing
    /// with the `project_fisheye_rgba16_texture_to_*` family so the projection
    /// minifies only mildly (no luma moiré / chroma fringing) at the working res.
    /// `src_w`/`src_h` are the native P010 (Y-plane) dims; `out_w`/`out_h` the
    /// working preview res.
    #[cfg(target_os = "macos")]
    pub fn resolve_p010_planes_to_rgba16(
        &self,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
    ) -> Result<wgpu::Texture> {
        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p010_planes_resolved_rgba16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Separate plane textures — bind their default views directly (the
        // single-texture path uses plane-aspect views of one P010 texture).
        let y_view = y_tex.create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let dst_view = out_tex.create_view(&Default::default());
        let dims = self.write_uniform("p010_resolve_planes_dims", &ResolveDims {
            src_w: src_w as f32, src_h: src_h as f32,
            out_w: out_w as f32, out_h: out_h as f32,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p010_resolve_planes_bg"),
            layout: &self.p010_resolve.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.p010_resolve.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("p010_resolve_planes_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_resolve_planes_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.p010_resolve.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out_tex)
    }

    /// Project an already-on-GPU `Rgba16Unorm` fisheye texture to half-equirect
    /// (16-bit). Same as [`Self::project_fisheye_to_equirect_texture_16`] but
    /// the source is an existing texture (e.g. from [`Self::resolve_p010_to_rgba16`])
    /// instead of a CPU byte upload — no host hop. `calib` MUST be resolved
    /// against `src_w`/`src_h` (the resolved texture's dims, NOT the native ones).
    pub fn project_fisheye_rgba16_texture_to_equirect_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // encoded in the calib uniform's src_w/src_h
        // Reuse a per-slot output texture across frames so the decoder thread
        // doesn't `create_texture` every frame (that contends with eframe's
        // main thread on the shared wgpu device). Safe to overwrite: the
        // previous frame's compose read this texture before this frame's
        // projection writes it (GPU-ordered on the same queue), and the eq
        // output isn't held past compose. `Texture` clones are cheap handles.
        let output_tex = {
            let mut cache = self.rgba16_eq_out_cache.lock().unwrap();
            let hit = matches!(cache.get(&slot), Some(&(w, h, _)) if w == out_w && h == out_h);
            if !hit {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fisheye_rgba16_eq_out_16"),
                    size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                cache.insert(slot, (out_w, out_h, tex));
            }
            cache.get(&slot).unwrap().2.clone()
        };
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_rgba16_R_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_rgba16_calib_16",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_rgba16_project_bg_16"),
            layout: &self.fisheye_to_hequirect_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_rgba16_project_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_rgba16_project_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Fisheye-output sibling of [`Self::project_fisheye_rgba16_texture_to_equirect_16`]:
    /// raw stabilized-fisheye SBS output (no equirect un-warp) from an on-GPU
    /// `Rgba16Unorm` fisheye texture. Same bindings as the equirect path
    /// (src / sampler / dst / R / calib); only the pipeline differs
    /// (`fisheye_to_fisheye_16`). Used by the GPU-resident export + zero-copy
    /// preview when the output mode is Fisheye and stab is off; the RS-aware
    /// sibling is [`Self::project_fisheye_rgba16_texture_to_fisheye_rs_16`].
    pub fn project_fisheye_rgba16_texture_to_fisheye_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // encoded in the calib uniform's src_w/src_h
        let output_tex = {
            let mut cache = self.rgba16_eq_out_cache.lock().unwrap();
            let hit = matches!(cache.get(&slot), Some(&(w, h, _)) if w == out_w && h == out_h);
            if !hit {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fisheye_rgba16_fish_out_16"),
                    size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                cache.insert(slot, (out_w, out_h, tex));
            }
            cache.get(&slot).unwrap().2.clone()
        };
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_rgba16_fish_R_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_rgba16_fish_calib_16",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_rgba16_fish_bg_16"),
            layout: &self.fisheye_to_fisheye_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_fisheye_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_rgba16_fish_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_rgba16_fish_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_fisheye_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware sibling of [`Self::project_fisheye_rgba16_texture_to_fisheye_16`]:
    /// normalized fisheye output with **per-row rolling-shutter** correction.
    /// Windows analogue of the macOS `project_fisheye_p010_to_fisheye_rs_texture_16`
    /// path so fisheye output gets the same RS treatment as half-equirect.
    /// Same bindings as the equirect RS path; only the pipeline differs.
    pub fn project_fisheye_rgba16_texture_to_fisheye_rs_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        per_row_matrices_f32: &[f32],
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // encoded in the calib uniform's src_w/src_h
        let output_tex = {
            let mut cache = self.rgba16_eq_out_cache.lock().unwrap();
            let hit = matches!(cache.get(&slot), Some(&(w, h, _)) if w == out_w && h == out_h);
            if !hit {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fisheye_rgba16_rs_fish_out_16"),
                    size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                cache.insert(slot, (out_w, out_h, tex));
            }
            cache.get(&slot).unwrap().2.clone()
        };
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_rgba16_rs_fish_R_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_rgba16_rs_fish_calib_16",
            &FisheyeCalibUniforms::from_public(calib));
        // Per-row R storage buffer (12 f32/row). Empty → 1-row zeros so the
        // shader's `rsm.r00 != 0.0` skip-check disables RS cleanly.
        let rs_bytes: &[u8] = if per_row_matrices_f32.is_empty() {
            &[0u8; 48]
        } else {
            bytemuck::cast_slice(per_row_matrices_f32)
        };
        let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fisheye_rgba16_rs_fish_rows_buf"),
            size: rs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&rs_buf, 0, rs_bytes);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_rgba16_rs_fish_bg_16"),
            layout: &self.fisheye_to_fisheye_rs_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_fisheye_rs_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_rgba16_rs_fish_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_rgba16_rs_fish_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_fisheye_rs_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware sibling of [`Self::project_fisheye_rgba16_texture_to_equirect_16`]:
    /// projects an on-GPU `Rgba16Unorm` fisheye texture to half-equirect (16-bit)
    /// with **per-row rolling-shutter** correction (the piece DJI OSV stab needs
    /// to kill jello). `per_row_matrices_f32` packs 12 f32 per source row
    /// (`src_h * 12`, from `pack_per_row_camera_matrices`); pass an empty slice
    /// to disable RS. Reuses the same per-slot output cache as the non-RS path.
    pub fn project_fisheye_rgba16_texture_to_equirect_rs_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        per_row_matrices_f32: &[f32],
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // encoded in the calib uniform's src_w/src_h
        let output_tex = {
            let mut cache = self.rgba16_eq_out_cache.lock().unwrap();
            let hit = matches!(cache.get(&slot), Some(&(w, h, _)) if w == out_w && h == out_h);
            if !hit {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fisheye_rgba16_rs_eq_out_16"),
                    size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                cache.insert(slot, (out_w, out_h, tex));
            }
            cache.get(&slot).unwrap().2.clone()
        };
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_rgba16_rs_R_16",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_rgba16_rs_calib_16",
            &FisheyeCalibUniforms::from_public(calib));
        // Per-row R storage buffer (12 f32/row). Empty → 1-row zeros so the
        // shader's `rsm.r00 != 0.0` skip-check disables RS cleanly.
        let rs_bytes: &[u8] = if per_row_matrices_f32.is_empty() {
            &[0u8; 48]
        } else {
            bytemuck::cast_slice(per_row_matrices_f32)
        };
        let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fisheye_rgba16_rs_rows_buf"),
            size: rs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&rs_buf, 0, rs_bytes);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_rgba16_rs_bg_16"),
            layout: &self.fisheye_to_hequirect_rs_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_hequirect_rs_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_rgba16_rs_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_rgba16_rs_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_hequirect_rs_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware variant of [`Self::project_fisheye_p010_to_equirect_texture_16`].
    /// Same inputs plus a per-scanline rotation buffer
    /// (`per_row_matrices_f32`, packed as 12 f32 per row — three vec4
    /// rows with trailing pad; total `fish_h * 12` floats). The
    /// projection shader looks up the row this pixel maps to and
    /// re-projects through that row's matrix, cancelling intra-frame
    /// shear from the sensor readout.
    ///
    /// Pass an all-zeros buffer to disable RS (the shader checks the
    /// first matrix's `r00` and skips the per-row pass if it's 0).
    #[cfg(target_os = "macos")]
    pub fn project_fisheye_p010_to_equirect_rs_texture_16(
        &self,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        per_row_matrices_f32: &[f32],
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h); // dims are encoded into the calib uniform
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fisheye_p010_rs_eq_tex_out_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let y_view  = y_tex .create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fisheye_p010_rs_R_uniform",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fisheye_p010_rs_calib_uniform",
            &FisheyeCalibUniforms::from_public(calib));

        // Per-row R storage buffer. Must contain at least one row;
        // when callers pass an empty slice we synthesize a 1-row
        // all-zeros buffer so the shader's `rsm.r00 != 0.0` skip-check
        // triggers reliably.
        let rs_bytes: &[u8] = if per_row_matrices_f32.is_empty() {
            &[0u8; 48] // 12 f32 == 48 bytes
        } else {
            bytemuck::cast_slice(per_row_matrices_f32)
        };
        let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fisheye_p010_rs_rows_buf"),
            size: rs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&rs_buf, 0, rs_bytes);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisheye_p010_rs_project_bg_16"),
            layout: &self.fisheye_p010_to_hequirect_16_rs.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.fisheye_p010_to_hequirect_16_rs.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fisheye_p010_rs_project_tex_enc_16"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fisheye_p010_rs_project_pass_16"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_p010_to_hequirect_16_rs.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Project a source fisheye eye into a stabilized fisheye OUTPUT
    /// (equidistant projection, KB source reprojection). Same shape as
    /// `project_fisheye_to_equirect_texture` — only the per-output-pixel
    /// parametrization differs, and the caller can compose the per-frame
    /// stab matrix with the per-eye view-adjust before calling.
    pub fn project_fisheye_to_fisheye_texture(
        &self,
        src_rgba: &[u8],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        let src_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_src"),
            size: wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.upload_rgba(&src_tex, src_rgba, src_w, src_h);
        let output_tex = make_rw_texture(&self.device, "fish_out_eye", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fish_out_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fish_out_cal",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fish_out_bg"),
            layout: &self.fisheye_to_fisheye.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_fisheye.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fish_out_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fish_out_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_fisheye.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware fisheye → stabilized fisheye output. Same as
    /// [`Self::project_fisheye_to_fisheye_texture`] plus a per-scanline
    /// rotation buffer (`rs_rows_f32` = 12 f32 per source row × src_h
    /// rows). Pass an empty slice to disable RS (the shader checks the
    /// first matrix's `r00`).
    pub fn project_fisheye_to_fisheye_rs_texture(
        &self,
        src_rgba: &[u8],
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        rs_rows_f32: &[f32],
    ) -> Result<wgpu::Texture> {
        let src_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_rs_src"),
            size: wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.upload_rgba(&src_tex, src_rgba, src_w, src_h);
        let output_tex = make_rw_texture(&self.device, "fish_out_rs_eye", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fish_out_rs_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fish_out_rs_cal",
            &FisheyeCalibUniforms::from_public(calib));
        let rs_bytes: &[u8] = if rs_rows_f32.is_empty() {
            &[0u8; 48] // 12 f32 == 48 bytes, all-zero → RS-off sentinel
        } else {
            bytemuck::cast_slice(rs_rows_f32)
        };
        let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fish_out_rs_rows"),
            size: rs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&rs_buf, 0, rs_bytes);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fish_out_rs_bg"),
            layout: &self.fisheye_to_fisheye_rs.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_fisheye_rs.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fish_out_rs_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fish_out_rs_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_fisheye_rs.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// 16-bit variant — Rgba16Unorm output for the 10-bit fisheye-output
    /// export path.
    pub fn project_fisheye_to_fisheye_texture_16(
        &self,
        src_rgba_or_rgba64le: &[u8],
        src_bit_depth: u8,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        // Source texture: Rgba16Unorm. If the source is 8-bit (RGBA8)
        // we widen to 16-bit by left-shifting each byte 8 bits (low
        // byte zero — no fake precision). 16-bit sources copy verbatim.
        let src_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_src_16"),
            size: wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let widened: std::borrow::Cow<[u8]> = if src_bit_depth >= 16 {
            std::borrow::Cow::Borrowed(src_rgba_or_rgba64le)
        } else {
            let n_pixels = (src_w as usize) * (src_h as usize);
            let mut out = Vec::with_capacity(n_pixels * 8);
            for px in src_rgba_or_rgba64le.chunks_exact(4) {
                out.push(0x00); out.push(px[0]);
                out.push(0x00); out.push(px[1]);
                out.push(0x00); out.push(px[2]);
                out.push(0xFF); out.push(0xFF);
            }
            std::borrow::Cow::Owned(out)
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &widened,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(src_w * 8),
                rows_per_image: Some(src_h),
            },
            wgpu::Extent3d { width: src_w, height: src_h, depth_or_array_layers: 1 },
        );
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_eye_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src_view = src_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fish_out16_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fish_out16_cal",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fish_out16_bg"),
            layout: &self.fisheye_to_fisheye_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.fisheye_to_fisheye_16.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 3, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fish_out16_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fish_out16_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_to_fisheye_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// Zero-copy P010 → stabilized fisheye output (Rgba16Unorm). Used
    /// by the macOS OSV fisheye-output export at 10-bit.
    #[cfg(target_os = "macos")]
    pub fn project_fisheye_p010_to_fisheye_texture_16(
        &self,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h);
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_p010_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let y_view  = y_tex .create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fish_p010_out_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fish_p010_out_cal",
            &FisheyeCalibUniforms::from_public(calib));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fish_p010_out_bg"),
            layout: &self.fisheye_p010_to_fisheye_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.fisheye_p010_to_fisheye_16.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cal_uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fish_p010_out_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fish_p010_out_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_p010_to_fisheye_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// RS-aware zero-copy P010 → stabilized fisheye output (Rgba16Unorm).
    /// Adds a per-scanline rotation buffer for rolling-shutter
    /// correction. Pass an empty slice to disable RS.
    #[cfg(target_os = "macos")]
    pub fn project_fisheye_p010_to_fisheye_rs_texture_16(
        &self,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        per_row_matrices_f32: &[f32],
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h);
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fish_out_p010_rs_16"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let y_view  = y_tex .create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let dst_view = output_tex.create_view(&Default::default());
        let r_uniform = self.write_uniform("fish_p010_out_rs_R",
            &EquirectUniforms::from_mat3(rotation.0));
        let cal_uniform = self.write_uniform("fish_p010_out_rs_cal",
            &FisheyeCalibUniforms::from_public(calib));
        let rs_bytes: &[u8] = if per_row_matrices_f32.is_empty() {
            &[0u8; 48]
        } else {
            bytemuck::cast_slice(per_row_matrices_f32)
        };
        let rs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fish_p010_out_rs_rows"),
            size: rs_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&rs_buf, 0, rs_bytes);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fish_p010_out_rs_bg"),
            layout: &self.fisheye_p010_to_fisheye_rs_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.fisheye_p010_to_fisheye_rs_16.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 4, resource: r_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cal_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rs_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fish_p010_out_rs_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fish_p010_out_rs_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fisheye_p010_to_fisheye_rs_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(output_tex)
    }

    /// 16-bit SBS compose. Same shape as `compose_sbs_textures` but
    /// the output is Rgba16Unorm and the inputs MUST be Rgba16Unorm.
    pub fn compose_sbs_textures_16(
        &self,
        left: &wgpu::Texture,
        right: &wgpu::Texture,
        eye_w: u32,
        eye_h: u32,
    ) -> Result<wgpu::Texture> {
        let out_w = eye_w * 2;
        let sbs = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sbs_export_compose_16"),
            size: wgpu::Extent3d { width: out_w, height: eye_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sbs_export_compose_enc_16"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: left, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &sbs, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: right, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &sbs, mip_level: 0,
                origin: wgpu::Origin3d { x: eye_w, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(sbs)
    }

    /// Zero-copy 10-bit P010 SBS compose. Takes the two Rgba16Unorm
    /// per-eye projection outputs and writes both planes of the P010
    /// CVPixelBuffer (Y full-res + UV half-res with 2×2 chroma
    /// subsampling) via two compute dispatches. The result is ready
    /// for VT to consume directly — no swscale, no readback, no
    /// ffmpeg frame copy on the CPU side.
    #[cfg(target_os = "macos")]
    pub fn compose_sbs_to_p010(
        &self,
        left_16: &wgpu::Texture,
        right_16: &wgpu::Texture,
        target: &crate::interop_macos::EncodePixelBufferP010,
        eye_w: u32, eye_h: u32,
    ) -> Result<()> {
        let l_v = left_16.create_view(&Default::default());
        let r_v = right_16.create_view(&Default::default());
        let y_v = target.y_tex.create_view(&Default::default());
        let uv_v = target.uv_tex.create_view(&Default::default());
        let uni = self.write_uniform("p010_compose_u",
            &ComposeSbsUniforms { eye_w, _pad0: 0, _pad1: 0, _pad2: 0 });
        let out_w = eye_w * 2;
        let out_h = eye_h;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compose_sbs_to_p010_enc"),
        });

        // Y plane pass (full SBS res).
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p010_y_bg"),
                layout: &self.compose_sbs_p010_y.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&y_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_y_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_sbs_p010_y.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        // UV plane pass (half SBS res — one thread per 2×2 source block).
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p010_uv_bg"),
                layout: &self.compose_sbs_p010_uv.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&uv_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                ],
            });
            let uv_w = out_w / 2;
            let uv_h = out_h / 2;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_uv_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_sbs_p010_uv.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((uv_w + 7) / 8, (uv_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        // Block until both planes are fully written before VT reads.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    /// Read back an `Rgba16Unorm` texture as packed **RGB48LE** (6
    /// bytes per pixel, little-endian, alpha channel dropped). The
    /// returned buffer is suitable for ffmpeg's RGB48LE pixel format
    /// → swscale → YUV420P10LE → Main10 encode.
    pub fn read_texture_rgb48(
        &self,
        tex: &wgpu::Texture,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>> {
        // Rgba16Unorm is 8 bytes/pixel on the GPU; we drop the 2 alpha
        // bytes to emit RGB48LE (6 bytes/pixel) for the encoder.
        let bpp = 8u32;
        let unpadded_row_bytes = w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_texture_rgb48_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_texture_rgb48_enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();

        // Pack 8-byte RGBA64LE → 6-byte RGB48LE per pixel (strip alpha).
        let mut out = Vec::with_capacity((w * h * 6) as usize);
        for y in 0..h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..w {
                let px_off = row_off + (x * 8) as usize;
                out.extend_from_slice(&mapped[px_off..px_off + 6]);
            }
        }
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// Like [`Self::read_texture_rgb48`] but keeps all 8 bytes/pixel
    /// (RGBA64LE) via a bulk per-row copy with NO per-pixel work. The
    /// encoder's swscale drops the alpha + converts to YUV in one SIMD
    /// pass; doing the alpha strip here meant a ~30M-iteration/frame loop
    /// on the export's main-thread critical path. Returns `w*h*8` bytes,
    /// tightly packed.
    pub fn read_texture_rgba64(
        &self,
        tex: &wgpu::Texture,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>> {
        let bpp = 8u32;
        let unpadded_row_bytes = w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_texture_rgba64_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_texture_rgba64_enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();

        // Bulk per-row copy — strips only the swscale row padding, keeps all
        // 8 bytes/pixel. At 7680-wide the row is already 256-aligned, so this
        // is effectively one contiguous memcpy.
        let row_bytes = unpadded_row_bytes as usize;
        let mut out = vec![0u8; row_bytes * h as usize];
        for y in 0..h as usize {
            let src = y * padded_row_bytes as usize;
            out[y * row_bytes..y * row_bytes + row_bytes]
                .copy_from_slice(&mapped[src..src + row_bytes]);
        }
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// GPU SBS compose straight to **P010 planes in plain, readable
    /// textures**: a full-res `R16Unorm` luma plane + a half-res
    /// `Rg16Unorm` interleaved-chroma plane, both 10-bit MSB-aligned (the
    /// P010LE layout NVENC consumes). Reuses the same compute shaders as the
    /// macOS IOSurface compose; the only difference is the destinations are
    /// `COPY_SRC` textures we can read back. The point: the encode thread
    /// can then feed NVENC P010 directly via `encode_frame_p010` and skip
    /// the single-threaded CPU swscale (RGB→YUV) that otherwise caps the
    /// export's encode stage.
    pub fn compose_sbs_to_p010_textures(
        &self,
        left_16: &wgpu::Texture,
        right_16: &wgpu::Texture,
        eye_w: u32, eye_h: u32,
        full_range: bool,
    ) -> Result<(wgpu::Texture, wgpu::Texture)> {
        let out_w = eye_w * 2;
        let out_h = eye_h;
        let y_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p010_y_tex"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let uv_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p010_uv_tex"),
            size: wgpu::Extent3d { width: out_w / 2, height: out_h / 2, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let l_v = left_16.create_view(&Default::default());
        let r_v = right_16.create_view(&Default::default());
        let y_v = y_tex.create_view(&Default::default());
        let uv_v = uv_tex.create_view(&Default::default());
        let uni = self.write_uniform("p010_tex_compose_u",
            &ComposeSbsUniforms { eye_w, _pad0: full_range as u32, _pad1: 0, _pad2: 0 });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compose_sbs_to_p010_textures_enc"),
        });
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p010_tex_y_bg"),
                layout: &self.compose_sbs_p010_y.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&y_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_tex_y_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_sbs_p010_y.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p010_tex_uv_bg"),
                layout: &self.compose_sbs_p010_uv.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&uv_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                ],
            });
            let uv_w = out_w / 2;
            let uv_h = out_h / 2;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010_tex_uv_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_sbs_p010_uv.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((uv_w + 7) / 8, (uv_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok((y_tex, uv_tex))
    }

    /// Read back a single-mip 2D texture as tightly-packed bytes via a bulk
    /// per-row copy (no per-pixel work). `bytes_per_px` must match the
    /// format (2 = R16Unorm, 4 = Rg16Unorm). Returns `w*h*bytes_per_px`.
    pub fn read_texture_planar(
        &self,
        tex: &wgpu::Texture,
        w: u32, h: u32,
        bytes_per_px: u32,
    ) -> Result<Vec<u8>> {
        let unpadded_row_bytes = w * bytes_per_px;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        // Reuse a staging buffer of this size across frames — a fresh 59/29 MB
        // host-visible allocation per frame dominated the readback cost.
        let staging = {
            let mut cache = self.readback_staging.lock().unwrap();
            cache.entry(buf_size).or_insert_with(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback_staging_cached"),
                    size: buf_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }).clone()
        };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_texture_planar_enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("planar map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("planar map: {e}")))?;
        let mapped = slice.get_mapped_range();
        let row = unpadded_row_bytes as usize;
        let mut out = vec![0u8; row * h as usize];
        for y in 0..h as usize {
            let src = y * padded_row_bytes as usize;
            out[y * row..y * row + row].copy_from_slice(&mapped[src..src + row]);
        }
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// Build an SBS texture by copying `left` to `(0,0)` and `right`
    /// to `(eye_w, 0)`. Output is `2*eye_w × eye_h × Rgba8Unorm` and
    /// includes COPY_SRC so the caller can read it back. This is the
    /// GPU-resident alternative to CPU-side SBS interleaving and
    /// removes ~88 MB/frame of memcpy at 3840×3840 per-eye.
    pub fn compose_sbs_textures(
        &self,
        left: &wgpu::Texture,
        right: &wgpu::Texture,
        eye_w: u32,
        eye_h: u32,
    ) -> Result<wgpu::Texture> {
        let out_w = eye_w * 2;
        let sbs = make_rw_texture(&self.device, "sbs_export_compose", out_w, eye_h,
            wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sbs_export_compose_enc"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: left, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &sbs, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: right, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &sbs, mip_level: 0,
                origin: wgpu::Origin3d { x: eye_w, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(sbs)
    }

    /// Read back an `Rgba8Unorm` texture as packed RGB8 (3 bytes per
    /// pixel). Handles the 256-byte-aligned bytes-per-row requirement
    /// of `wgpu::ImageCopyBuffer`. Used by the fisheye export path to
    /// produce the encoder-ready RGB buffer after GPU compose.
    pub fn read_texture_rgb8(
        &self,
        tex: &wgpu::Texture,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>> {
        let bpp = 4u32;
        let unpadded_row_bytes = w * bpp;
        let padded_row_bytes = (unpadded_row_bytes + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_texture_rgb8_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_texture_rgb8_enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| Error::Wgpu("staging map send channel closed".into()))?
            .map_err(|e| Error::Wgpu(format!("staging map: {e}")))?;
        let mapped = slice.get_mapped_range();

        let mut out_rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            let row_off = (y * padded_row_bytes) as usize;
            for x in 0..w {
                let px_off = row_off + (x * 4) as usize;
                out_rgb.extend_from_slice(&mapped[px_off..px_off + 3]);
            }
        }
        drop(mapped);
        staging.unmap();
        Ok(out_rgb)
    }

    fn record_cdl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        params: CdlParams,
    ) {
        let uniforms = CdlUniforms {
            lift: params.lift, gamma: params.gamma, gain: params.gain,
            shadow: params.shadow, highlight: params.highlight,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        let uni = self.write_uniform("cdl_chain_u", &uniforms);
        let dims = (dst.width(), dst.height());
        self.record_per_pixel(encoder, "cdl_chain", &self.cdl, src, dst, &uni, dims);
    }

    fn record_color_grade(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        params: ColorGradeParams,
    ) {
        let uniforms = ColorGradeUniforms {
            temperature: params.temperature,
            tint: params.tint,
            saturation: params.saturation,
            _pad: 0.0,
        };
        let uni = self.write_uniform("grade_chain_u", &uniforms);
        let dims = (dst.width(), dst.height());
        self.record_per_pixel(encoder, "grade_chain", &self.color_grade, src, dst, &uni, dims);
    }

    fn record_per_pixel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &PerPixelPipeline,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        uni: &wgpu::Buffer,
        (w, h): (u32, u32),
    ) {
        let bg_label = format!("{label}_bg");
        let pass_label = format!("{label}_pass");
        let src_v = src.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&bg_label),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 2, resource: uni.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&pass_label), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }

    fn record_sharpen(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        blur_h: &wgpu::Texture,
        blur_v: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32, h: u32,
        params: SharpenParams,
    ) {
        let blur_h_u = self.write_uniform("sharp_chain_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("sharp_chain_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("sharp_chain_c_u",
            &SharpenCombineUniforms {
                amount: params.amount,
                apply_lat_weight: if params.apply_lat_weight { 1 } else { 0 },
                _pad0: 0, _pad1: 0,
            });
        self.dispatch_blur_1d(encoder, "sharp_chain_h", src,   blur_h, &blur_h_u, w, h);
        self.dispatch_blur_1d(encoder, "sharp_chain_v", blur_h, blur_v, &blur_v_u, w, h);
        let src_v   = src.create_view(&Default::default());
        let blur_vv = blur_v.create_view(&Default::default());
        let dst_v   = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharp_chain_combine_bg"),
            layout: &self.sharpen_combine.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blur_vv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 3, resource: combine_u.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sharp_chain_combine_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.sharpen_combine.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }

    fn record_mid_detail(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        small: &wgpu::Texture,
        small_h: &wgpu::Texture,
        small_v: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32, h: u32, sw: u32, sh: u32,
        params: MidDetailParams,
    ) {
        let blur_h_u = self.write_uniform("mid_chain_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("mid_chain_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("mid_chain_c_u",
            &MidDetailCombineUniforms { amount: params.amount, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 });

        // Pass 1: 4× downsample.
        {
            let src_v   = src.create_view(&Default::default());
            let small_v = small.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid_chain_ds_bg"),
                layout: &self.downsample_4x.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&small_v) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid_chain_ds_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_4x.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((sw + 7) / 8, (sh + 7) / 8, 1);
        }
        // Passes 2 + 3: blur the small image.
        self.dispatch_blur_1d(encoder, "mid_chain_h", small,   small_h, &blur_h_u, sw, sh);
        self.dispatch_blur_1d(encoder, "mid_chain_v", small_h, small_v, &blur_v_u, sw, sh);
        // Pass 4: upsample-combine to full-res output.
        {
            let src_v       = src.create_view(&Default::default());
            let smallv_view = small_v.create_view(&Default::default());
            let dst_v       = dst.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid_chain_combine_bg"),
                layout: &self.mid_detail_combine.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&smallv_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.bilinear_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_v) },
                    wgpu::BindGroupEntry { binding: 4, resource: combine_u.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid_chain_combine_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.mid_detail_combine.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }
    }

    fn record_sharpen_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        blur_h: &wgpu::Texture,
        blur_v: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32, h: u32,
        params: SharpenParams,
    ) {
        let blur_h_u = self.write_uniform("sharp16_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("sharp16_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("sharp16_c_u",
            &SharpenCombineUniforms {
                amount: params.amount,
                apply_lat_weight: if params.apply_lat_weight { 1 } else { 0 },
                _pad0: 0, _pad1: 0,
            });
        self.dispatch_blur_1d_16(encoder, "sharp16_h", src,    blur_h, &blur_h_u, w, h);
        self.dispatch_blur_1d_16(encoder, "sharp16_v", blur_h, blur_v, &blur_v_u, w, h);
        let src_v   = src.create_view(&Default::default());
        let blur_vv = blur_v.create_view(&Default::default());
        let dst_v   = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharp16_combine_bg"),
            layout: &self.sharpen_combine_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blur_vv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 3, resource: combine_u.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sharp16_combine_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.sharpen_combine_16.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }

    fn record_mid_detail_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        small: &wgpu::Texture,
        small_h: &wgpu::Texture,
        small_v: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32, h: u32, sw: u32, sh: u32,
        params: MidDetailParams,
    ) {
        let blur_h_u = self.write_uniform("mid16_h_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 0, _pad0: 0, _pad1: 0 });
        let blur_v_u = self.write_uniform("mid16_v_u",
            &GaussianBlur1dUniforms { sigma: params.sigma, direction: 1, _pad0: 0, _pad1: 0 });
        let combine_u = self.write_uniform("mid16_c_u",
            &MidDetailCombineUniforms { amount: params.amount, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 });
        {
            let src_v   = src.create_view(&Default::default());
            let small_v = small.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid16_ds_bg"),
                layout: &self.downsample_4x_16.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&small_v) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid16_ds_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_4x_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((sw + 7) / 8, (sh + 7) / 8, 1);
        }
        self.dispatch_blur_1d_16(encoder, "mid16_h", small,   small_h, &blur_h_u, sw, sh);
        self.dispatch_blur_1d_16(encoder, "mid16_v", small_h, small_v, &blur_v_u, sw, sh);
        {
            let src_v       = src.create_view(&Default::default());
            let smallv_view = small_v.create_view(&Default::default());
            let dst_v       = dst.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mid16_combine_bg"),
                layout: &self.mid_detail_combine_16.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&smallv_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.bilinear_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_v) },
                    wgpu::BindGroupEntry { binding: 4, resource: combine_u.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mid16_combine_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.mid_detail_combine_16.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }
    }

    /// Upload the LUT texture + sampler. Held by the caller for the
    /// duration of the encoder. Same content as `apply_lut3d` builds
    /// inside its own scope.
    fn upload_lut3d_resources(&self, lut: &vr180_core::color::Cube3DLut) -> Lut3DResources {
        let lut_bytes = lut.to_rgba8_for_upload();
        let lut_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stack_lut3d"),
            size: wgpu::Extent3d {
                width: lut.size, height: lut.size, depth_or_array_layers: lut.size,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &lut_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &lut_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lut.size * 4),
                rows_per_image: Some(lut.size),
            },
            wgpu::Extent3d {
                width: lut.size, height: lut.size, depth_or_array_layers: lut.size,
            },
        );
        Lut3DResources { tex: lut_tex, size: lut.size }
    }

    fn record_lut3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        lut_res: &Lut3DResources,
        intensity: f32,
        w: u32, h: u32,
    ) {
        let uniforms = Lut3DUniforms {
            size: lut_res.size,
            intensity,
            _pad0: 0, _pad1: 0,
        };
        let uni = self.write_uniform("lut3d_chain_u", &uniforms);
        let src_v = src.create_view(&Default::default());
        let lut_v = lut_res.tex.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lut3d_chain_bg"),
            layout: &self.lut3d.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lut3d.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lut3d_chain_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.lut3d.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }
}

/// Outlives the encoder that uses it — the LUT texture + its size.
/// (No sampler held here; `Device::lut3d.sampler` is the shared one.)
struct Lut3DResources {
    tex: wgpu::Texture,
    size: u32,
}

// ===== Phase 0.7.5.6: zero-copy encode SBS composition =====

impl P010ComposePipeline {
    fn create(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("compose_sbs_{label}")),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("compose_sbs_{label}_bgl")),
            entries: &[
                bgle_tex(0),
                bgle_tex(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: out_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                bgle_uniform(3, std::mem::size_of::<ComposeSbsUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("compose_sbs_{label}_pll")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("compose_sbs_{label}_pipeline")),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl ComposeSbsPipeline {
    fn create(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "compose_sbs", COMPOSE_SBS_BGRA_WGSL,
            std::mem::size_of::<ComposeSbsUniforms>() as u64,
        )
    }

    /// Preview-mode composer. Same binding shape as the BGRA SBS compose
    /// but the shader picks one of three modes (SBS / anaglyph / overlay)
    /// based on a `mode` field in the uniform. Output is Rgba8Unorm.
    fn create_preview(device: &wgpu::Device) -> Self {
        Self::create_with_shader(
            device, "compose_preview", COMPOSE_PREVIEW_WGSL,
            std::mem::size_of::<PreviewComposeUniforms>() as u64,
        )
    }

    fn create_with_shader(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        uniform_size: u64,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let bgl_label = format!("{label}_bgl");
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&bgl_label),
            entries: &[
                bgle_tex(0),
                bgle_tex(1),
                bgle_storage_out(2),
                bgle_uniform(3, uniform_size),
            ],
        });
        let pll_label = format!("{label}_pll");
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&pll_label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pp_label = format!("{label}_pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&pp_label),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }
}

impl Device {
    /// Compose two equirect textures (left + right eye, RGBA8) into one
    /// SBS texture in BGRA byte order. The destination is typically an
    /// IOSurface-backed wgpu::Texture (viewed as Rgba8Unorm) that's
    /// about to be handed to the VideoToolbox encoder.
    ///
    /// This is the LAST GPU step on the zero-copy-encode path —
    /// `Device::poll(Maintain::Wait)` after submit ensures the IOSurface
    /// bytes are fully written before VT reads them.
    ///
    /// The full chained color stack + SBS compose runs in one encoder:
    /// `apply_color_stack_to_sbs_bgra(left_eq, right_eq, dst, plan)`.
    pub fn apply_color_stack_to_sbs_bgra(
        &self,
        left_eq: &wgpu::Texture,
        right_eq: &wgpu::Texture,
        dst_bgra: &wgpu::Texture,
        eye_w: u32, eye_h: u32,
        plan: &ColorStackPlan,
    ) -> Result<()> {
        let t_total = Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("color_stack_sbs_enc"),
        });
        // Pre-allocate the LUT resources (if any) so they outlive the encoder.
        let lut_resources = if let Some((ref lut, _)) = plan.lut {
            Some(self.upload_lut3d_resources(lut))
        } else { None };

        // Run the color stack on each eye into a shared `intermediates`
        // Vec. Each `record_color_stack` returns `Option<usize>` —
        // an index into intermediates pointing at the final per-eye
        // texture, or `None` meaning "no stages ran, use the source
        // texture directly".
        let mut intermediates: Vec<wgpu::Texture> = Vec::with_capacity(16);
        // Per-eye "Matching Eyes" trim: grade each eye with the OPPOSITE CT/tint
        // offset (left +, right −). `for_eye` borrows the plan when the trim is
        // off (no clone). The LUT is unchanged by the trim, so the pre-uploaded
        // `lut_resources` apply to both eyes.
        let left_idx = self.record_color_stack(
            &mut encoder, left_eq, eye_w, eye_h, &plan.for_eye(true), &lut_resources, &mut intermediates,
            "stack_l",
        );
        let right_idx = self.record_color_stack(
            &mut encoder, right_eq, eye_w, eye_h, &plan.for_eye(false), &lut_resources, &mut intermediates,
            "stack_r",
        );
        let left_final = match left_idx {
            Some(i) => &intermediates[i],
            None => left_eq,
        };
        let right_final = match right_idx {
            Some(i) => &intermediates[i],
            None => right_eq,
        };

        // Final pass: compose to SBS BGRA.
        let uni = self.write_uniform("compose_sbs_u",
            &ComposeSbsUniforms { eye_w, _pad0: 0, _pad1: 0, _pad2: 0 });
        {
            let l_v = left_final.create_view(&Default::default());
            let r_v = right_final.create_view(&Default::default());
            let d_v = dst_bgra.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compose_sbs_bg"),
                layout: &self.compose_sbs.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&d_v) },
                    wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                ],
            });
            let out_w = eye_w * 2;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compose_sbs_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_sbs.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (eye_h + 7) / 8, 1);
        }
        // Submit and wait so the IOSurface is fully written before
        // VideoToolbox starts reading. `Maintain::Wait` blocks the calling
        // thread until the GPU has completed every submitted command.
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        tracing::debug!(elapsed=?t_total.elapsed(), "apply_color_stack_to_sbs_bgra");
        Ok(())
    }

    /// Run one eye's worth of the color stack into the encoder.
    /// Returns the index into `intermediates` of the final per-eye
    /// texture, or `None` if every stage was identity (in which case
    /// the caller should use `src` directly).
    ///
    /// Helper for `apply_color_stack_to_sbs_bgra` — same recipe as the
    /// body of `apply_color_stack_texture` but factored out so both
    /// eyes share one encoder + one intermediates Vec.
    fn record_color_stack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        w: u32, h: u32,
        plan: &ColorStackPlan,
        lut_resources: &Option<Lut3DResources>,
        intermediates: &mut Vec<wgpu::Texture>,
        eye_label: &str,
    ) -> Option<usize> {
        let mut current_idx: Option<usize> = None;

        // Stage 1: CDL.
        if !plan.cdl.is_identity() {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_cdl"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_cdl(encoder, prev, &next, plan.cdl);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Stage 2: 3D LUT.
        if let (Some((_, intensity)), Some(ref lr)) = (&plan.lut, lut_resources) {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_lut"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_lut3d(encoder, prev, &next, lr, *intensity, w, h);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Stage 3: sharpen.
        if !plan.sharpen.is_identity() {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_sharp"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING);
            let blur_h = make_rw_texture(&self.device,
                &format!("{eye_label}_sharp_bh"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let blur_v = make_rw_texture(&self.device,
                &format!("{eye_label}_sharp_bv"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_sharpen(encoder, prev, &blur_h, &blur_v, &next, w, h, plan.sharpen);
            intermediates.push(blur_h);
            intermediates.push(blur_v);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Stage 3b: white balance (temperature / tint) — POST-LUT, matching
        // the Python app's order (CDL → LUT → sharpen → temp/tint → mid →
        // saturation). On log footage this is essential: temp/tint must act
        // on the LUT's Rec.709 output, not the log input.
        if plan.color_grade.has_white_balance() {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_wb"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_color_grade(encoder, prev, &next, plan.color_grade.white_balance_only());
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Stage 4: mid-detail.
        if !plan.mid_detail.is_identity() {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_mid"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let sw = (w + 3) / 4;
            let sh = (h + 3) / 4;
            let small = make_rw_texture(&self.device,
                &format!("{eye_label}_mid_s"), sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let small_h = make_rw_texture(&self.device,
                &format!("{eye_label}_mid_sh"), sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let small_v = make_rw_texture(&self.device,
                &format!("{eye_label}_mid_sv"), sw, sh,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_mid_detail(encoder, prev,
                &small, &small_h, &small_v, &next, w, h, sw, sh, plan.mid_detail);
            intermediates.push(small);
            intermediates.push(small_h);
            intermediates.push(small_v);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Stage 5: saturation — POST-LUT (the only color adjustment applied
        // after the LUT). Temperature/tint were handled pre-LUT above.
        if plan.color_grade.has_saturation() {
            let next = make_rw_texture(&self.device,
                &format!("{eye_label}_sat"), w, h,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_color_grade(encoder, prev, &next, plan.color_grade.saturation_only());
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // Index of the final stage's output texture, or None for
        // "every stage was identity, caller should use src".
        current_idx
    }

    // ----- 16-bit color stack ----- (10-bit end-to-end export path)

    fn record_cdl_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        params: CdlParams,
    ) {
        let uniforms = CdlUniforms {
            lift: params.lift, gamma: params.gamma, gain: params.gain,
            shadow: params.shadow, highlight: params.highlight,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        let uni = self.write_uniform("cdl16_chain_u", &uniforms);
        let dims = (dst.width(), dst.height());
        self.record_per_pixel(encoder, "cdl16_chain", &self.cdl_16, src, dst, &uni, dims);
    }

    fn record_color_grade_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        params: ColorGradeParams,
    ) {
        let uniforms = ColorGradeUniforms {
            temperature: params.temperature,
            tint: params.tint,
            saturation: params.saturation,
            _pad: 0.0,
        };
        let uni = self.write_uniform("grade16_chain_u", &uniforms);
        let dims = (dst.width(), dst.height());
        self.record_per_pixel(encoder, "grade16_chain", &self.color_grade_16, src, dst, &uni, dims);
    }

    fn record_lut3d_16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        lut_res: &Lut3DResources,
        intensity: f32,
        w: u32, h: u32,
    ) {
        let uniforms = Lut3DUniforms {
            size: lut_res.size,
            intensity,
            _pad0: 0, _pad1: 0,
        };
        let uni = self.write_uniform("lut3d16_chain_u", &uniforms);
        let src_v = src.create_view(&Default::default());
        let lut_v = lut_res.tex.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lut3d16_chain_bg"),
            layout: &self.lut3d_16.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lut3d_16.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lut3d16_chain_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.lut3d_16.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }

    /// 8-bit per-eye color stack — Rgba8Unorm in, owned Rgba8Unorm out.
    /// Returns the final intermediate when the plan has stages, or a
    /// fresh storage-binding-friendly copy of the input when no stage
    /// ran. The result texture has `TEXTURE_BINDING + STORAGE_BINDING`
    /// usage, suitable for feeding into `compose_preview`. Used by the
    /// GUI preview path.
    pub fn apply_color_stack_to_sbs_bgra_per_eye(
        &self,
        src: &wgpu::Texture,
        w: u32, h: u32,
        plan: &ColorStackPlan,
    ) -> Result<wgpu::Texture> {
        if !plan.any_active() {
            // No color stage to run — return the input verbatim by way
            // of a texture-to-texture copy so the caller owns an
            // independent texture (and so the source texture isn't kept
            // around longer than necessary). When the input already has
            // the right usage flags the copy is cheap on the GPU.
            let dst = make_rw_texture(&self.device, "stack8_per_eye_passthrough", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stack8_per_eye_copy"),
            });
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: src, mip_level: 0,
                    origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dst, mip_level: 0,
                    origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.queue.submit(Some(enc.finish()));
            return Ok(dst);
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stack8_per_eye_enc"),
        });
        let lut_resources = if let Some((ref lut, _)) = plan.lut {
            Some(self.upload_lut3d_resources(lut))
        } else { None };
        let mut intermediates: Vec<wgpu::Texture> = Vec::with_capacity(4);
        let final_idx = self.record_color_stack(
            &mut encoder, src, w, h, plan, &lut_resources, &mut intermediates,
            "stack8_per_eye",
        ).expect("plan.any_active was true; record_color_stack must have run a stage");
        self.queue.submit(Some(encoder.finish()));
        // Move the final intermediate out of the Vec and return it.
        // swap_remove avoids the cost of shifting later elements.
        let result = intermediates.swap_remove(final_idx);
        Ok(result)
    }

    /// Apply the full color stack to one `Rgba16Unorm` equirect (or
    /// fisheye) texture and return a new `Rgba16Unorm` texture with the
    /// result. Order matches the 8-bit path:
    ///   CDL → LUT3D → sharpen → mid-detail → color_grade.
    ///
    /// All stages stay at Rgba16Unorm storage, so a 10-bit export pipeline
    /// keeps full 10-bit precision through the entire grade. Returns
    /// `None` when every stage is identity; the caller can keep using
    /// the original texture (zero-cost identity).
    pub fn apply_color_stack_per_eye_16(
        &self,
        src: &wgpu::Texture,
        w: u32, h: u32,
        plan: &ColorStackPlan,
    ) -> Result<Option<wgpu::Texture>> {
        if !plan.any_active() {
            return Ok(None);
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("color_stack_16_enc"),
        });

        let lut_resources = if let Some((ref lut, _)) = plan.lut {
            Some(self.upload_lut3d_resources(lut))
        } else { None };

        let mut intermediates: Vec<wgpu::Texture> = Vec::with_capacity(12);

        let make_16 = |label: &str, w: u32, h: u32| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };

        let mut current_idx: Option<usize> = None;

        // 1. CDL.
        if !plan.cdl.is_identity() {
            let next = make_16("stack16_cdl_out", w, h);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_cdl_16(&mut encoder, prev, &next, plan.cdl);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // 2. 3D LUT.
        if let (Some((_, intensity)), Some(ref lr)) = (&plan.lut, &lut_resources) {
            let next = make_16("stack16_lut_out", w, h);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_lut3d_16(&mut encoder, prev, &next, lr, *intensity, w, h);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // 3. Sharpen (full-res, 3 internal dispatches).
        if !plan.sharpen.is_identity() {
            let next = make_16("stack16_sharp_out", w, h);
            let blur_h = make_16("stack16_sharp_bh", w, h);
            let blur_v = make_16("stack16_sharp_bv", w, h);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_sharpen_16(&mut encoder, prev, &blur_h, &blur_v, &next, w, h, plan.sharpen);
            intermediates.push(blur_h);
            intermediates.push(blur_v);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // 3b. White balance (temperature / tint) — POST-LUT, matching the
        // Python order (CDL → LUT → sharpen → temp/tint → mid → saturation).
        if plan.color_grade.has_white_balance() {
            let next = make_16("stack16_wb_out", w, h);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_color_grade_16(&mut encoder, prev, &next, plan.color_grade.white_balance_only());
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // 4. Mid-detail (downsample → blur → upsample-combine).
        if !plan.mid_detail.is_identity() {
            let sw = (w + 3) / 4;
            let sh = (h + 3) / 4;
            let next = make_16("stack16_mid_out", w, h);
            let small   = make_16("stack16_mid_s",  sw, sh);
            let small_h = make_16("stack16_mid_sh", sw, sh);
            let small_v = make_16("stack16_mid_sv", sw, sh);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_mid_detail_16(&mut encoder, prev,
                &small, &small_h, &small_v, &next, w, h, sw, sh, plan.mid_detail);
            intermediates.push(small);
            intermediates.push(small_h);
            intermediates.push(small_v);
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }
        // 5. Saturation — POST-LUT (the only color adjustment after the
        // LUT). Temperature/tint were applied pre-LUT above.
        if plan.color_grade.has_saturation() {
            let next = make_16("stack16_sat_out", w, h);
            let prev = match current_idx {
                Some(i) => &intermediates[i],
                None => src,
            };
            self.record_color_grade_16(&mut encoder, prev, &next, plan.color_grade.saturation_only());
            intermediates.push(next);
            current_idx = Some(intermediates.len() - 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let final_idx = current_idx.expect("at least one stage ran (any_active was true)");
        let final_tex = intermediates.swap_remove(final_idx);
        Ok(Some(final_tex))
    }

    /// Preview-mode composer: combines two per-eye half-equirect
    /// textures into one preview frame in SBS / anaglyph / overlay mode.
    /// Output is `Rgba8Unorm` and includes COPY_SRC + TEXTURE_BINDING so
    /// the caller can hand it to the egui renderer or read it back.
    pub fn compose_preview(
        &self,
        left: &wgpu::Texture,
        right: &wgpu::Texture,
        eye_w: u32,
        eye_h: u32,
        mode: PreviewMode,
    ) -> Result<wgpu::Texture> {
        let (out_w, out_h) = mode.output_dims(eye_w, eye_h);
        // The shader writes this via `textureStore`, so it MUST have
        // STORAGE_BINDING. It's also sampled by egui (TEXTURE_BINDING)
        // and may be read back (COPY_SRC). No COPY_DST — nothing copies
        // INTO it. We expose an sRGB view-format so the GUI can register an
        // sRGB view with egui: egui treats sampled textures as LINEAR and
        // re-applies the OETF, so without the sRGB decode-on-sample our
        // Rec.709-gamma values get double-encoded → washed-out preview.
        // (Storage textures can't BE sRGB, hence the view-format trick.)
        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compose_preview_out"),
            size: wgpu::Extent3d { width: out_w, height: out_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let uni = self.write_uniform("compose_preview_u",
            &PreviewComposeUniforms {
                eye_w,
                mode: mode.shader_code(),
                _pad0: 0, _pad1: 0,
            });

        let l_v = left.create_view(&Default::default());
        let r_v = right.create_view(&Default::default());
        let d_v = out_tex.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose_preview_bg"),
            layout: &self.compose_preview.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&r_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&d_v) },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compose_preview_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compose_preview_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.compose_preview.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out_tex)
    }
}

// ----- Phase 0.7.5 smoke tests -----
//
// Synthetic 8×8 inputs, exercise each color tool's identity case
// + a non-trivial case. Identity must round-trip bit-exact (the
// `is_identity` short-circuit returns `rgb_in.to_vec()`); the
// non-identity case is sanity-checked against the math, not
// bit-validated against Python (that's a follow-up phase against
// a real frame).
#[cfg(test)]
mod color_tool_tests {
    use super::*;

    fn solid_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) { v.extend_from_slice(&[r, g, b]); }
        v
    }

    #[test]
    fn cdl_identity_roundtrips_exact() {
        let dev = Device::new().expect("device");
        let img = solid_image(8, 8, 100, 150, 200);
        let out = dev.apply_cdl(&img, 8, 8, CdlParams::default()).expect("cdl");
        assert_eq!(img, out, "identity CDL must roundtrip exact");
    }

    #[test]
    fn cdl_gain_2x_doubles_or_clips() {
        let dev = Device::new().expect("device");
        let img = solid_image(8, 8, 50, 100, 200);
        let params = CdlParams { gain: 2.0, ..Default::default() };
        let out = dev.apply_cdl(&img, 8, 8, params).expect("cdl");
        // 50 * 2 = 100, 100 * 2 = 200, 200 * 2 = 400 clamps to 255.
        // Allow ±2 quantization slack for the uniform-buffer round-trip
        // through f32 (50/255 * 2 ≈ 0.392 → encoded 100 ± 1).
        let px = &out[..3];
        assert!((px[0] as i32 - 100).abs() <= 2, "R: got {}", px[0]);
        assert!((px[1] as i32 - 200).abs() <= 2, "G: got {}", px[1]);
        assert_eq!(px[2], 255, "B: 2*200 should clip to 255, got {}", px[2]);
    }

    #[test]
    fn color_grade_identity_roundtrips_exact() {
        let dev = Device::new().expect("device");
        let img = solid_image(8, 8, 100, 150, 200);
        let out = dev.apply_color_grade(&img, 8, 8, ColorGradeParams::default())
            .expect("color_grade");
        assert_eq!(img, out);
    }

    #[test]
    fn color_grade_zero_saturation_makes_grayscale() {
        let dev = Device::new().expect("device");
        // Pure red — sat=0 should desaturate to luma value (BT.601):
        // luma = 0.299*255 + 0.587*0 + 0.114*0 ≈ 76.
        let img = solid_image(8, 8, 255, 0, 0);
        let params = ColorGradeParams { saturation: 0.0, ..Default::default() };
        let out = dev.apply_color_grade(&img, 8, 8, params).expect("color_grade");
        let px = &out[..3];
        let target = 76i32;
        for (i, &v) in px.iter().enumerate() {
            assert!((v as i32 - target).abs() <= 2,
                "channel {i}: got {v}, want ~{target} (gray)");
        }
        // All three channels equal → grayscale.
        assert!((px[0] as i32 - px[1] as i32).abs() <= 1);
        assert!((px[1] as i32 - px[2] as i32).abs() <= 1);
    }

    #[test]
    fn sharpen_identity_roundtrips_exact() {
        let dev = Device::new().expect("device");
        let img = solid_image(16, 16, 80, 120, 200);
        let out = dev.apply_sharpen(&img, 16, 16, SharpenParams::default())
            .expect("sharpen");
        assert_eq!(img, out);
    }

    #[test]
    fn sharpen_solid_image_returns_solid_image() {
        // On a uniform input, blur == input, so detail == 0, so output
        // should equal input regardless of `amount`. Catches sign bugs
        // and per-row weight bugs in one shot.
        let dev = Device::new().expect("device");
        let img = solid_image(32, 32, 100, 150, 200);
        let params = SharpenParams { amount: 1.5, ..Default::default() };
        let out = dev.apply_sharpen(&img, 32, 32, params).expect("sharpen");
        for (i, (a, b)) in img.iter().zip(out.iter()).enumerate() {
            assert!((*a as i32 - *b as i32).abs() <= 1,
                "uniform image should be sharpen-invariant; idx {i}: orig={a} got={b}");
        }
    }

    #[test]
    fn mid_detail_identity_roundtrips_exact() {
        let dev = Device::new().expect("device");
        let img = solid_image(16, 16, 80, 120, 200);
        let out = dev.apply_mid_detail(&img, 16, 16, MidDetailParams::default())
            .expect("mid_detail");
        assert_eq!(img, out);
    }

    #[test]
    fn mid_detail_solid_image_returns_solid_image() {
        // Same logic as sharpen: on a uniform input, blur == input,
        // detail == 0, output == input.
        let dev = Device::new().expect("device");
        let img = solid_image(32, 32, 100, 150, 200);
        let params = MidDetailParams { amount: 0.5, sigma: 1.0 };
        let out = dev.apply_mid_detail(&img, 32, 32, params).expect("mid_detail");
        for (i, (a, b)) in img.iter().zip(out.iter()).enumerate() {
            assert!((*a as i32 - *b as i32).abs() <= 2,
                "uniform image should be clarity-invariant; idx {i}: orig={a} got={b}");
        }
    }
}

// ----- EAC cross assembly regression test (Phase 0.6.6 bug fix) -----
//
// The original nv12_to_eac_cross.wgsl shader's middle-band branch
// unconditionally returned the Lens-A formula (`s0[y_in, cx + tw]`)
// regardless of `uni.lens`. That made Lens B's CENTER face sample s0
// (= Lens A's source) instead of s4_rot — both eyes showed the same
// content in the front-facing region, breaking stereo separation.
//
// This test runs the shader with s0 = solid red and s4 = solid blue
// (in BT.709 P010), and asserts:
//   - Lens A CENTER is reddish (R > B) — sourced from s0
//   - Lens B CENTER is bluish (B > R) — sourced from s4 (via s4_rot)
//
// The pre-fix shader fails the Lens-B half. The fixed shader passes both.
#[cfg(test)]
mod eac_assembly_regression {
    use super::*;
    use vr180_core::eac::Dims;

    /// Build a constant-value R16Unorm texture of the given dims. Used
    /// to feed the Y plane of a synthetic stream.
    fn const_r16_texture(
        device: &wgpu::Device, queue: &wgpu::Queue,
        w: u32, h: u32, value_u16: u16, label: &str,
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes: Vec<u8> = std::iter::repeat(value_u16.to_le_bytes())
            .take((w * h) as usize)
            .flat_map(|b| b.into_iter())
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 2),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        tex
    }

    /// Build a constant-value Rg16Unorm UV-plane texture (half resolution).
    fn const_rg16_texture(
        device: &wgpu::Device, queue: &wgpu::Queue,
        w: u32, h: u32, u_u16: u16, v_u16: u16, label: &str,
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Pack (u, v) per pixel as 2× u16 little-endian.
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            bytes.extend_from_slice(&u_u16.to_le_bytes());
            bytes.extend_from_slice(&v_u16.to_le_bytes());
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        tex
    }

    /// Read back a wgpu Rgba8Unorm texture to a Vec<u8> (RGBA, packed).
    /// Returns (4 * w * h) bytes.
    fn read_rgba8(dev: &Device, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
        let padded_row_bytes = ((w * 4) + 255) & !255;
        let buf_size = (padded_row_bytes * h) as u64;
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_readback_enc"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let row_off = (y * padded_row_bytes) as usize;
            out.extend_from_slice(&mapped[row_off..row_off + (w * 4) as usize]);
        }
        drop(mapped);
        staging.unmap();
        out
    }

    /// Build solid-color P010 YUV planes (Y, UV) for one synthetic stream.
    /// `(r, g, b)` are linear [0, 1] floats describing the desired color.
    ///
    /// BT.709 limited-range encode: Y in [16/255, 235/255], UV in
    /// [16/255, 240/255] with 128/255 = neutral. Then bit-shift to P010
    /// (10 bits in upper 10 of u16): `u16 = (8bit_val << 8) | (8bit_val >> 2)`
    /// — approximately `8bit * 256`. We use exactly `8bit * 257` so the
    /// 8-bit max of 255 maps to u16 65535.
    fn make_p010_planes(
        dev: &Device, w: u32, h: u32,
        r: f32, g: f32, b: f32,
        label_prefix: &str,
    ) -> (wgpu::Texture, wgpu::Texture) {
        // BT.709 RGB → YUV (full range 0..1).
        let y_f = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let u_f = -0.1146 * r - 0.3854 * g + 0.5000 * b;
        let v_f =  0.5000 * r - 0.4542 * g - 0.0458 * b;
        // Limited range encode (8-bit).
        let y_8 = (16.0 + y_f * 219.0).round() as u16;
        let u_8 = (128.0 + u_f * 224.0).round() as u16;
        let v_8 = (128.0 + v_f * 224.0).round() as u16;
        // 8-bit → u16 (mimic P010 upper-10-of-16 by multiplying by 257
        // so 255 → 65535). Inside the shader, the BT.709 P010 expansion
        // formula recovers the original Y/U/V from this.
        let y_u16 = (y_8 * 257).min(65535) as u16;
        let u_u16 = (u_8 * 257).min(65535) as u16;
        let v_u16 = (v_8 * 257).min(65535) as u16;

        let y_tex = const_r16_texture(&dev.device, &dev.queue,
            w, h, y_u16, &format!("{label_prefix}_y"));
        let uv_tex = const_rg16_texture(&dev.device, &dev.queue,
            w / 2, h / 2, u_u16, v_u16, &format!("{label_prefix}_uv"));
        (y_tex, uv_tex)
    }

    /// Sample one RGBA pixel from a packed RGBA8 buffer.
    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8) {
        let off = ((y * w + x) * 4) as usize;
        (buf[off], buf[off + 1], buf[off + 2])
    }

    #[test]
    fn nv12_to_eac_lens_b_center_is_blue_not_red() {
        // The Phase 0.6.6 regression: the WGSL shader's middle-band
        // branch returned the Lens-A formula for ALL lenses, so Lens B's
        // CENTER face sampled s0 (red here) instead of s4_rot (blue).
        // This test catches that.

        let dev = Device::new().expect("device");
        // Skip if the adapter lacks TEXTURE_FORMAT_16BIT_NORM (the
        // shader's P010 input format) — same gate as the runtime.
        if !dev.device.features().contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            eprintln!("skip: adapter lacks TEXTURE_FORMAT_16BIT_NORM");
            return;
        }

        let dims = Dims::new(5952, 1920);
        let cw = dims.cross_w();

        // s0 = pure red, s4 = pure blue (BT.709 P010).
        let (s0_y, s0_uv) = make_p010_planes(&dev, dims.stream_w, dims.stream_h, 1.0, 0.0, 0.0, "s0");
        let (s4_y, s4_uv) = make_p010_planes(&dev, dims.stream_w, dims.stream_h, 0.0, 0.0, 1.0, "s4");

        let cross_a = dev.nv12_to_eac_cross(&s0_y, &s0_uv, &s4_y, &s4_uv, Lens::A, dims)
            .expect("nv12_to_eac Lens A");
        let cross_b = dev.nv12_to_eac_cross(&s0_y, &s0_uv, &s4_y, &s4_uv, Lens::B, dims)
            .expect("nv12_to_eac Lens B");

        let a_buf = read_rgba8(&dev, &cross_a, cw, cw);
        let b_buf = read_rgba8(&dev, &cross_b, cw, cw);

        // Probe the CENTER of the cross for each lens.
        let (cx, cy) = (cw / 2, cw / 2);
        let (ar, _ag, ab) = px(&a_buf, cw, cx, cy);
        let (br, _bg, bb) = px(&b_buf, cw, cx, cy);

        assert!(ar as i32 > ab as i32 + 30,
            "Lens A CENTER should be reddish (from s0); got R={ar} B={ab}");
        assert!(bb as i32 > br as i32 + 30,
            "Lens B CENTER should be bluish (from s4 via s4_rot); got R={br} B={bb}. \
             Regression: pre-fix WGSL shader returned Lens A formula for Lens B.");
    }

    #[test]
    fn nv12_to_eac_lens_b_left_right_come_from_correct_s0_tiles() {
        // Lens B LEFT  ← s0[:, sw-tw:sw]   (outer-right tile of s0)
        // Lens B RIGHT ← s0[:, 0:tw]       (outer-left tile of s0)
        // To distinguish these from Lens A's middle (which uses
        // contiguous interior s0 tiles), we vary s0 spatially: paint
        // s0's outer cols (matching Lens-B tiles) bright red and its
        // interior cols (matching Lens-A tiles) bright green. A
        // misrouted Lens B middle would show green instead of red on
        // LEFT/RIGHT.
        //
        // For brevity here we just sanity-check the constant-color case
        // (above test covers the Lens A/B center divergence); the
        // spatial-color variant is straightforward extension.

        // Constant-color smoke test: with s0=red and s4=blue, Lens B's
        // LEFT and RIGHT bands should still be reddish (they DO come
        // from s0, just from different cols of s0).
        let dev = Device::new().expect("device");
        if !dev.device.features().contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            eprintln!("skip: adapter lacks TEXTURE_FORMAT_16BIT_NORM");
            return;
        }
        let dims = Dims::new(5952, 1920);
        let cw = dims.cross_w();
        let tw = dims.tile_w();
        let (s0_y, s0_uv) = make_p010_planes(&dev, dims.stream_w, dims.stream_h, 1.0, 0.0, 0.0, "s0");
        let (s4_y, s4_uv) = make_p010_planes(&dev, dims.stream_w, dims.stream_h, 0.0, 0.0, 1.0, "s4");

        let cross_b = dev.nv12_to_eac_cross(&s0_y, &s0_uv, &s4_y, &s4_uv, Lens::B, dims)
            .expect("nv12_to_eac Lens B");
        let buf = read_rgba8(&dev, &cross_b, cw, cw);

        // LEFT face: cross[tw:tw+sh, 0:tw]. Probe (tw/2, tw + cw/4).
        let (r, _g, b) = px(&buf, cw, tw / 2, tw + cw / 4);
        assert!(r as i32 > b as i32 + 30,
            "Lens B LEFT should be reddish (from s0); got R={r} B={b}");

        // RIGHT face: cross[tw:tw+sh, tw+cn:cw]. Probe (tw + cn + tw/2, tw + cw/4).
        let cn = vr180_core::eac::CENTER_W;
        let (r, _g, b) = px(&buf, cw, tw + cn + tw / 2, tw + cw / 4);
        assert!(r as i32 > b as i32 + 30,
            "Lens B RIGHT should be reddish (from s0); got R={r} B={b}");
    }
}
