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
        Ok(Self { instance, adapter, device, queue, eac_to_equirect, lut3d, nv12_to_eac })
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
