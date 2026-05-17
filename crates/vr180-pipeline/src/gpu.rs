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
use std::time::Instant;

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
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    eac_to_equirect: EacToEquirectPipeline,
    lut3d: Lut3DPipeline,
    nv12_to_eac: Nv12ToEacPipeline,
    cdl: PerPixelPipeline,
    color_grade: PerPixelPipeline,
    gaussian_blur: GaussianBlur1dPipeline,
    sharpen_combine: SharpenCombinePipeline,
    mid_detail_combine: MidDetailCombinePipeline,
    downsample_4x: Downsample4xPipeline,
    bilinear_sampler: wgpu::Sampler,
}

#[derive(Debug)]
struct EacToEquirectPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
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

impl Device {
    /// Create a wgpu device using the best available adapter.
    /// Backend is picked by wgpu (Metal on macOS, DX12/Vulkan on Win).
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::Wgpu("no compatible adapter".into()))?;
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
                },
                None,
            )
            .await
            .map_err(|e| Error::Wgpu(format!("request_device: {e}")))?;
        let eac_to_equirect = EacToEquirectPipeline::create(&device);
        let lut3d = Lut3DPipeline::create(&device);
        let nv12_to_eac = Nv12ToEacPipeline::create(&device);
        let cdl = PerPixelPipeline::create(
            &device, "cdl", CDL_WGSL,
            std::mem::size_of::<CdlUniforms>() as u64,
        );
        let color_grade = PerPixelPipeline::create(
            &device, "color_grade", COLOR_GRADE_WGSL,
            std::mem::size_of::<ColorGradeUniforms>() as u64,
        );
        let gaussian_blur = GaussianBlur1dPipeline::create(&device);
        let sharpen_combine = SharpenCombinePipeline::create(&device);
        let mid_detail_combine = MidDetailCombinePipeline::create(&device);
        let downsample_4x = Downsample4xPipeline::create(&device);
        let bilinear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok(Self {
            instance, adapter, device, queue,
            eac_to_equirect, lut3d, nv12_to_eac,
            cdl, color_grade,
            gaussian_blur, sharpen_combine, mid_detail_combine, downsample_4x,
            bilinear_sampler,
        })
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
            wgpu::ImageCopyTexture {
                texture: &input_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &cross_rgba,
            wgpu::ImageDataLayout {
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
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eac_to_equirect_bg"),
            layout: &self.eac_to_equirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.eac_to_equirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
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
            pass.set_bind_group(0, &bind_group, &[]);
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
            wgpu::ImageCopyTexture {
                texture: &output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
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

impl EacToEquirectPipeline {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eac_to_equirect"),
            source: wgpu::ShaderSource::Wgsl(EAC_TO_EQUIRECT_WGSL.into()),
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eac_to_equirect_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eac_to_equirect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cross_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
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

const LUT3D_WGSL: &str = include_str!("shaders/lut3d.wgsl");
const NV12_TO_EAC_WGSL: &str = include_str!("shaders/nv12_to_eac_cross.wgsl");
const CDL_WGSL: &str = include_str!("shaders/cdl.wgsl");
const COLOR_GRADE_WGSL: &str = include_str!("shaders/color_grade.wgsl");
const GAUSSIAN_BLUR_1D_WGSL: &str = include_str!("shaders/gaussian_blur_1d.wgsl");
const SHARPEN_COMBINE_WGSL: &str = include_str!("shaders/sharpen_combine.wgsl");
const MID_DETAIL_COMBINE_WGSL: &str = include_str!("shaders/mid_detail_combine.wgsl");
const DOWNSAMPLE_4X_WGSL: &str = include_str!("shaders/downsample_4x.wgsl");

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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12_to_eac"),
            source: wgpu::ShaderSource::Wgsl(NV12_TO_EAC_WGSL.into()),
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nv12_to_eac_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nv12_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self { pipeline, bind_group_layout, sampler }
    }
}

impl PerPixelPipeline {
    /// Build a generic per-pixel pipeline: 1 input texture + 1 storage
    /// output + 1 uniform buffer. Used by CDL, color_grade, and
    /// (future) saturation, white-point shift, etc. The `uniform_size`
    /// is used for the layout's `min_binding_size` so wgpu's validation
    /// can catch buffer-size mismatches at bind time instead of failing
    /// silently on the GPU.
    fn create(
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pp_label = format!("{label}_pipeline");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&pp_label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
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
            wgpu::ImageCopyTexture {
                texture: &in_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
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
            pass.set_bind_group(0, &bind_group, &[]);
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
            wgpu::ImageCopyTexture {
                texture: &out_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
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
            pass.set_bind_group(0, &bg, &[]);
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
            pass.set_bind_group(0, &bg, &[]);
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
            pass.set_bind_group(0, &bg, &[]);
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
            wgpu::ImageCopyTexture {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
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
        let bg_label = format!("{label}_bg");
        let pass_label = format!("{label}_pass");
        let src_v = src.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&bg_label),
            layout: &self.gaussian_blur.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_v) },
                wgpu::BindGroupEntry { binding: 2, resource: uni.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&pass_label), timestamp_writes: None,
        });
        pass.set_pipeline(&self.gaussian_blur.pipeline);
        pass.set_bind_group(0, &bg, &[]);
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
            wgpu::ImageCopyTexture {
                texture: tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
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
        let cw = dims.cross_w();

        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(match lens { Lens::A => "cross_a", Lens::B => "cross_b" }),
            size: wgpu::Extent3d { width: cw, height: cw, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
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
            layout: &self.nv12_to_eac.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&s0_y_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&s0_uv_v) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&s4_y_v) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&s4_uv_v) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.nv12_to_eac.sampler) },
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
            pass.set_pipeline(&self.nv12_to_eac.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eac_to_equirect_zc_bg"),
            layout: &self.eac_to_equirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cross_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.eac_to_equirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&output_view) },
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
            pass.set_bind_group(0, &bind_group, &[]);
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
            wgpu::ImageCopyTexture {
                texture: &output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lut3d"),
            source: wgpu::ShaderSource::Wgsl(LUT3D_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lut3d_bgl"),
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
            label: Some("lut3d_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lut3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lut3d_smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
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
            wgpu::ImageCopyTexture {
                texture: &input_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &input_rgba,
            wgpu::ImageDataLayout {
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
            wgpu::ImageCopyTexture {
                texture: &lut_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &lut_bytes,
            wgpu::ImageDataLayout {
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
            pass.set_bind_group(0, &bind_group, &[]);
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
            wgpu::ImageCopyTexture {
                texture: &output_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_blur_1d"),
            source: wgpu::ShaderSource::Wgsl(GAUSSIAN_BLUR_1D_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_blur_1d_bgl"),
            entries: &[
                bgle_tex(0),
                bgle_storage_out(1),
                bgle_uniform(2, std::mem::size_of::<GaussianBlur1dUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_blur_1d_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gaussian_blur_1d_pipeline"),
            layout: Some(&pll),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { pipeline, bind_group_layout }
    }
}

impl SharpenCombinePipeline {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharpen_combine"),
            source: wgpu::ShaderSource::Wgsl(SHARPEN_COMBINE_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharpen_combine_bgl"),
            entries: &[
                bgle_tex(0),   // orig
                bgle_tex(1),   // blur
                bgle_storage_out(2),
                bgle_uniform(3, std::mem::size_of::<SharpenCombineUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sharpen_combine_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sharpen_combine_pipeline"),
            layout: Some(&pll),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { pipeline, bind_group_layout }
    }
}

impl MidDetailCombinePipeline {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mid_detail_combine"),
            source: wgpu::ShaderSource::Wgsl(MID_DETAIL_COMBINE_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mid_detail_combine_bgl"),
            entries: &[
                bgle_tex(0),   // orig (full-res)
                bgle_tex(1),   // small blurred (1/4 res)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                bgle_storage_out(3),
                bgle_uniform(4, std::mem::size_of::<MidDetailCombineUniforms>() as u64),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mid_detail_combine_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mid_detail_combine_pipeline"),
            layout: Some(&pll),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { pipeline, bind_group_layout }
    }
}

impl Downsample4xPipeline {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downsample_4x"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_4X_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsample_4x_bgl"),
            entries: &[
                bgle_tex(0),
                bgle_storage_out(1),
            ],
        });
        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("downsample_4x_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("downsample_4x_pipeline"),
            layout: Some(&pll),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { pipeline, bind_group_layout }
    }
}

/// Bind-group-layout entry for a writeable Rgba8Unorm storage texture.
/// Used as the output of every per-pixel + multi-pass color shader.
fn bgle_storage_out(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
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
}

impl Default for ColorStackPlan {
    fn default() -> Self {
        Self {
            cdl: CdlParams::default(),
            lut: None,
            sharpen: SharpenParams::default(),
            mid_detail: MidDetailParams::default(),
            color_grade: ColorGradeParams::default(),
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
    ) -> Result<wgpu::Texture> {
        let output_tex = make_rw_texture(&self.device, "equirect_tex_out", out_w, out_h,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("project_tex_enc"),
        });
        self.record_equirect_project(&mut encoder, cross_tex, &output_tex, out_w, out_h);
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
    ) -> Result<wgpu::Texture> {
        let cross_rgba = rgb_to_rgba(cross_rgb, cross_w as usize, cross_w as usize);
        let cross_tex = make_rw_texture(&self.device, "cross_for_project",
            cross_w, cross_w,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        self.upload_rgba(&cross_tex, &cross_rgba, cross_w, cross_w);
        self.project_cross_texture_to_equirect_texture(&cross_tex, out_w, out_h)
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

        // Stage 5: color grade (temp/tint/sat).
        if !plan.color_grade.is_identity() {
            let next = make_rw_texture(&self.device, "stack_grade_out", w, h,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);
            let prev = intermediates.last().unwrap_or(equirect_tex);
            self.record_color_grade(&mut encoder, prev, &next, plan.color_grade);
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
    ) {
        let cross_v = cross_tex.create_view(&Default::default());
        let dst_v = dst.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("equirect_project_bg"),
            layout: &self.eac_to_equirect.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cross_v) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.eac_to_equirect.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dst_v) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("equirect_project_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.eac_to_equirect.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
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
        pass.set_bind_group(0, &bg, &[]);
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
        pass.set_bind_group(0, &bg, &[]);
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
            pass.set_bind_group(0, &bg, &[]);
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
            pass.set_bind_group(0, &bg, &[]);
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
            wgpu::ImageCopyTexture {
                texture: &lut_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &lut_bytes,
            wgpu::ImageDataLayout {
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
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
    }
}

/// Outlives the encoder that uses it — the LUT texture + its size.
/// (No sampler held here; `Device::lut3d.sampler` is the shared one.)
struct Lut3DResources {
    tex: wgpu::Texture,
    size: u32,
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
