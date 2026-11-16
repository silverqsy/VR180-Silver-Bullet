//! GPU side of raw (Bayer) video: the mosaic goes up as one `R16Uint`
//! texture per eye and comes back as a camera-linear `Rgba16Unorm` eye the
//! fisheye projection kernels sample like any other 16-bit source
//! (`shaders/bayer_demosaic.wgsl`).

use crate::gpu::Device;
use crate::{Error, Result};

const BAYER_DEMOSAIC_WGSL: &str = include_str!("shaders/bayer_demosaic.wgsl");
const RAW_DEVELOP_WGSL: &str = include_str!("shaders/raw_develop.wgsl");
const RAW_MEASURE_WGSL: &str = include_str!("shaders/raw_measure.wgsl");
const RAW_REDUCE_WGSL: &str = include_str!("shaders/raw_reduce.wgsl");
const RAW_GRID_BUILD_WGSL: &str = include_str!("shaders/raw_grid_build.wgsl");
const RAW_GRID_BLUR_WGSL: &str = include_str!("shaders/raw_grid_blur.wgsl");
const RAW_DEVELOP_LOCAL_WGSL: &str = include_str!("shaders/raw_develop_local.wgsl");

use crate::raw_color::prophoto_luma;
use crate::raw_tone::{
    DevelopConstants, DevelopGeometry, FrameStats, LocalCurves, LocalTone, Measure, RawDevelop, ToneKeys, BASE_FALLBACK, BASE_FLOOR,
    BASE_LO, BASE_RANGE, CLIP_FROM, CLIP_TO, CURVE_LO, LOCAL_LO, LOCAL_PER_STOP, LOCAL_SAMPLES, OUT_LO,
};
use std::sync::{Arc, Mutex};

/// What a mosaic's codes mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayerLayout {
    /// 2×2 pattern, row-major: 0 = R, 1 = G, 2 = B.
    pub cfa: [u8; 4],
    pub black: f32,
    pub white: f32,
}

impl BayerLayout {
    /// Parity `(x, y)` of the red site, `None` for a pattern that is no
    /// Bayer 2×2 (one red, one blue on the opposite corner, two greens).
    pub fn red_site(&self) -> Option<(u32, u32)> {
        let red = self.cfa.iter().position(|&c| c == 0)?;
        let blue = self.cfa.iter().position(|&c| c == 2)?;
        let greens = self.cfa.iter().filter(|&&c| c == 1).count();
        (greens == 2 && red + blue == 3).then_some((red as u32 & 1, red as u32 >> 1))
    }
}

/// Pipelines of the raw path, built on first use.
#[derive(Debug)]
pub(crate) struct RawPipelines {
    demosaic: wgpu::ComputePipeline,
    demosaic_layout: wgpu::BindGroupLayout,
    develop: wgpu::ComputePipeline,
    develop_layout: wgpu::BindGroupLayout,
    measure: wgpu::ComputePipeline,
    measure_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    grid_build: wgpu::ComputePipeline,
    grid_build_layout: wgpu::BindGroupLayout,
    grid_blur: wgpu::ComputePipeline,
    grid_blur_layout: wgpu::BindGroupLayout,
    develop_local: wgpu::ComputePipeline,
    develop_local_layout: wgpu::BindGroupLayout,
    grid_sampler: wgpu::Sampler,
    /// What the live preview's measurements have found so far.
    memory: Arc<Mutex<ToneMemory>>,
}

/// The live preview's tone keys: it must not wait on the GPU, so every
/// frame it renders uses the last measurement that has come back and, when
/// none is on its way, queues one of itself.
#[derive(Default)]
struct ToneMemory {
    latest: Option<FrameStats>,
    in_flight: bool,
    /// Curves to use instead of the ones derived from the frame (a
    /// measuring harness sets them).
    curves: Option<Arc<LocalCurves>>,
    /// Called when a measurement lands that differs from `latest`: a paused
    /// preview draws its frame again with it.
    notify: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for ToneMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToneMemory").field("latest", &self.latest).field("in_flight", &self.in_flight).finish_non_exhaustive()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DevelopParams {
    neutral: [f32; 4],
    cam_to_work: [[f32; 4]; 3],
    work_to_out: [[f32; 4]; 3],
    curve: [f32; 4],
    clip: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeasureParams {
    neutral: [f32; 4],
    cam_to_work: [[f32; 4]; 3],
    clip: [f32; 4],
    cell: [f32; 4],
    luma: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    neutral: [f32; 4],
    cam_to_work: [[f32; 4]; 3],
    clip: [f32; 4],
    luma: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    cell: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DevelopLocalParams {
    neutral: [f32; 4],
    cam_to_work: [[f32; 4]; 3],
    work_to_out: [[f32; 4]; 3],
    black: [f32; 4],
    fed: [f32; 4],
    gains: [f32; 4],
    clip: [f32; 4],
    luma: [f32; 4],
    grid: [f32; 4],
    stage: [f32; 4],
    keyed: [f32; 4],
}

/// Cells along the longer side of the frame the keys are read from: three
/// to a cell of the reduced copy the curves are derived from.
const KEY_CELLS: u32 = 126;
/// Key cells along one side of a cell of that reduced copy.
const STAT_BLOCK: usize = 3;
/// Most taps along one side of a cell.
const KEY_TAPS: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DemosaicParams {
    black: f32,
    scale: f32,
    red_x: u32,
    red_y: u32,
    src_w: i32,
    src_h: i32,
    half: u32,
    _pad: u32,
}

impl RawPipelines {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bayer_demosaic"),
            source: wgpu::ShaderSource::Wgsl(BAYER_DEMOSAIC_WGSL.into()),
        });
        let demosaic_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bayer_demosaic_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<DemosaicParams>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bayer_demosaic_pll"),
            bind_group_layouts: &[Some(&demosaic_layout)],
            immediate_size: 0,
        });
        let demosaic = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bayer_demosaic_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let develop_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw_develop"),
            source: wgpu::ShaderSource::Wgsl(RAW_DEVELOP_WGSL.into()),
        });
        let develop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_develop_bgl"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<DevelopParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let develop_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raw_develop_pll"),
            bind_group_layouts: &[Some(&develop_layout)],
            immediate_size: 0,
        });
        let develop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raw_develop_pipeline"),
            layout: Some(&develop_pll),
            module: &develop_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let measure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw_measure"),
            source: wgpu::ShaderSource::Wgsl(RAW_MEASURE_WGSL.into()),
        });
        let measure_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_measure_bgl"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<MeasureParams>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let measure_pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raw_measure_pll"),
            bind_group_layouts: &[Some(&measure_layout)],
            immediate_size: 0,
        });
        let measure = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raw_measure_pipeline"),
            layout: Some(&measure_pll),
            module: &measure_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Shadows / Highlights: the reduced copy, its bilateral grid, and
        // the development that reads the base from it.
        use wgpu::TextureViewDimension::{D2, D3};
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_reduce_bgl"),
            entries: &[
                bgl_texture(0, D2, true),
                bgl_storage(1, wgpu::TextureFormat::R32Float, D2),
                bgl_uniform(2, std::mem::size_of::<ReduceParams>()),
            ],
        });
        let reduce = compute_pipeline(device, "raw_reduce", RAW_REDUCE_WGSL, "main", &reduce_layout);
        let grid_build_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_grid_build_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::Rgba16Float, D3),
                bgl_uniform(2, std::mem::size_of::<GridParams>()),
            ],
        });
        let grid_build = compute_pipeline(device, "raw_grid_build", RAW_GRID_BUILD_WGSL, "build", &grid_build_layout);
        let grid_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_grid_blur_bgl"),
            entries: &[bgl_texture(0, D3, true), bgl_storage(1, wgpu::TextureFormat::Rgba16Float, D3)],
        });
        let grid_blur = compute_pipeline(device, "raw_grid_blur", RAW_GRID_BLUR_WGSL, "blur", &grid_blur_layout);
        let develop_local_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_develop_local_bgl"),
            entries: &[
                bgl_texture(0, D2, true),
                bgl_storage(1, wgpu::TextureFormat::Rgba16Unorm, D2),
                bgl_uniform(2, std::mem::size_of::<DevelopLocalParams>()),
                bgl_storage_buffer(3),
                bgl_storage_buffer(4),
                bgl_storage_buffer(5),
                bgl_texture(6, D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                bgl_storage_buffer(8),
                bgl_storage_buffer(9),
                bgl_storage_buffer(10),
            ],
        });
        let develop_local = compute_pipeline(device, "raw_develop_local", RAW_DEVELOP_LOCAL_WGSL, "main", &develop_local_layout);
        let grid_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raw_grid_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            demosaic, demosaic_layout, develop, develop_layout, measure, measure_layout,
            reduce, reduce_layout, grid_build, grid_build_layout, grid_blur, grid_blur_layout,
            develop_local, develop_local_layout, grid_sampler, memory: Arc::default(),
        }
    }
}

fn bgl_texture(binding: u32, dim: wgpu::TextureViewDimension, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: dim, multisampled: false },
        count: None,
    }
}

fn bgl_storage(binding: u32, format: wgpu::TextureFormat, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: dim },
        count: None,
    }
}

fn bgl_uniform(binding: u32, size: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size as u64),
        },
        count: None,
    }
}

fn bgl_storage_buffer(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn compute_pipeline(device: &wgpu::Device, label: &str, wgsl: &str, entry: &str, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
    let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pll),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Device {
    fn raw_pipelines(&self) -> &RawPipelines {
        self.raw.get_or_init(|| RawPipelines::create(&self.device))
    }

    /// An empty mosaic texture (`R16Uint`) of one eye.
    pub fn new_bayer_texture(&self, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bayer_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Replace a mosaic texture's samples (`w × h` codes, row-major).
    pub fn write_bayer_texture(&self, tex: &wgpu::Texture, samples: &[u16]) {
        self.write_bayer_bytes(tex, bytemuck::cast_slice(samples));
    }

    /// [`Self::write_bayer_texture`] from the codes' little-endian bytes.
    pub fn write_bayer_bytes(&self, tex: &wgpu::Texture, samples: &[u8]) {
        let (w, h) = (tex.width(), tex.height());
        debug_assert_eq!(samples.len(), (w as usize) * (h as usize) * 2);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            samples,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 2), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    /// An empty camera-linear eye (`Rgba16Unorm`) a demosaic writes and the
    /// 16-bit projection kernels read.
    pub fn new_linear_eye_texture(&self, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bayer_linear_eye"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Demosaic `bayer` into `out`: at the mosaic's own size (`out` = the
    /// mosaic's size, Malvar-He-Cutler) or at half size (`out` = half the
    /// mosaic's size, one pixel per 2×2 site group). The result is the
    /// camera's linear RGB with the sensor's white level at 1.0.
    pub fn demosaic_bayer(&self, bayer: &wgpu::Texture, layout: &BayerLayout, out: &wgpu::Texture) -> Result<()> {
        let (sw, sh) = (bayer.width(), bayer.height());
        let half = match (out.width(), out.height()) {
            (w, h) if (w, h) == (sw, sh) => 0,
            (w, h) if (w * 2, h * 2) == (sw, sh) => 1,
            (w, h) => return Err(Error::Ffmpeg(format!("demosaic: a {sw}x{sh} mosaic cannot fill a {w}x{h} eye"))),
        };
        let Some((red_x, red_y)) = layout.red_site() else {
            return Err(Error::Ffmpeg(format!("demosaic: {:?} is not a Bayer pattern", layout.cfa)));
        };
        let span = (layout.white - layout.black).max(1.0);
        let params = self.write_uniform("bayer_demosaic_params", &DemosaicParams {
            black: layout.black,
            scale: 1.0 / span,
            red_x,
            red_y,
            src_w: sw as i32,
            src_h: sh as i32,
            half,
            _pad: 0,
        });
        let pipes = self.raw_pipelines();
        let src_view = bayer.create_view(&Default::default());
        let dst_view = out.create_view(&Default::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayer_demosaic_bg"),
            layout: &pipes.demosaic_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bayer_demosaic") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bayer_demosaic"), timestamp_writes: None });
            pass.set_pipeline(&pipes.demosaic);
            pass.set_bind_group(0, Some(&bind), &[]);
            pass.dispatch_workgroups(out.width().div_ceil(8), out.height().div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

fn row4(r: [f32; 3]) -> [f32; 4] {
    [r[0], r[1], r[2], 0.0]
}

/// The working space's luminance weights, with `w` in the fourth place.
fn luma4(w: f32) -> [f32; 4] {
    let l = prophoto_luma();
    [l[0] as f32, l[1] as f32, l[2] as f32, w]
}

/// What a mapped measurement says of the frame: `bytes` holds the cells of
/// every eye measured (`cx × cy` each, eye after eye), 16 bytes (four
/// `f32`: the mean scene RGB and the mean log₂ luminance) each.
fn stats_from_cells(bytes: &[u8], cx: usize, cy: usize) -> Option<FrameStats> {
    let word = |c: &[u8], i: usize| f32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]);
    let cells: Vec<[f32; 3]> = bytes.chunks_exact(16).map(|c| [0, 1, 2].map(|i| word(c, i))).collect();
    let keys = ToneKeys::from_cells(&cells)?;
    // The reduced copy the curves come from: blocks of key cells, averaged in log.
    let logs: Vec<f32> = bytes.chunks_exact(16).map(|c| word(c, 3)).collect();
    let mut levels = Vec::new();
    for eye in logs.chunks_exact(cx * cy) {
        for by in (0..cy).step_by(STAT_BLOCK) {
            for bx in (0..cx).step_by(STAT_BLOCK) {
                let (mut sum, mut n) = (0f32, 0u32);
                for y in by..(by + STAT_BLOCK).min(cy) {
                    for x in bx..(bx + STAT_BLOCK).min(cx) {
                        sum += eye[y * cx + x];
                        n += 1;
                    }
                }
                levels.push(sum / n as f32);
            }
        }
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The brightest channel of the cells at each level.
    let mut peaks = vec![0f32; LOCAL_SAMPLES];
    for (cell, level) in cells.iter().zip(&logs) {
        let at = (((level - LOCAL_LO) * LOCAL_PER_STOP).round().max(0.0) as usize).min(LOCAL_SAMPLES - 1);
        peaks[at] = peaks[at].max(cell[0].max(cell[1]).max(cell[2]));
    }
    Some(FrameStats { keys, levels: levels.into(), peaks: peaks.into() })
}

/// Two measurements say the same about a frame (nothing to draw again for).
fn same_stats(a: &FrameStats, b: &FrameStats) -> bool {
    let stops = |x: f32, y: f32| (x.max(1e-9).log2() - y.max(1e-9).log2()).abs();
    if stops(a.keys.max, b.keys.max) >= 0.01 || stops(a.keys.dark, b.keys.dark) >= 0.03 || a.levels.len() != b.levels.len() {
        return false;
    }
    // The levels at a few ranks.
    let n = a.levels.len();
    n == 0 || (0..=8).all(|i| { let k = i * (n - 1) / 8; (a.levels[k] - b.levels[k]).abs() < 0.02 })
}

impl Device {
    /// Whom to tell when the live preview's tone keys change (see
    /// [`Measure::Latest`]): the preview worker, so a paused frame is drawn
    /// again with the keys of its own measurement. `None` clears it and
    /// forgets what was measured (another clip).
    pub fn set_raw_tone_notifier(&self, notify: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut m) = self.raw_pipelines().memory.lock() {
            m.latest = None;
            m.notify = notify;
        }
    }

    /// Queue the reduced copies of the given eyes and return the buffer
    /// they are copied into (16 bytes per cell, eye after eye).
    fn submit_tone_measurement(&self, eyes: &[&wgpu::Texture], geometry: &DevelopGeometry) -> (wgpu::Buffer, usize, usize) {
        let pipes = self.raw_pipelines();
        // The frame the keys describe is all the eyes side by side.
        let (ew, eh) = (eyes[0].width(), eyes[0].height());
        let long = (ew * eyes.len() as u32).max(eh) as f32;
        let cell = long / KEY_CELLS as f32;
        let (cx, cy) = (((ew as f32 / cell).round() as u32).max(1), ((eh as f32 / cell).round() as u32).max(1));
        // A row of the copy must be a multiple of 256 bytes: the cells'
        // texture is that wide (16 texels), the cells past `cx` are not read.
        let padded = cx.div_ceil(16) * 16;
        let eye_bytes = (padded * cy * 16) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_measure_staging"),
            size: eye_bytes * eyes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self.write_uniform("raw_measure_params", &MeasureParams {
            neutral: row4(geometry.neutral),
            cam_to_work: geometry.camera_to_work.map(row4),
            clip: [CLIP_FROM, CLIP_TO, 0.0, 0.0],
            cell: [ew as f32 / cx as f32, eh as f32 / cy as f32, cell.ceil().clamp(1.0, KEY_TAPS as f32), 0.0],
            luma: luma4(BASE_FLOOR),
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw_measure") });
        for (n, eye) in eyes.iter().enumerate() {
            let cells = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("raw_measure_cells"),
                size: wgpu::Extent3d { width: padded, height: cy, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let (src_view, dst_view) = (eye.create_view(&Default::default()), cells.create_view(&Default::default()));
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raw_measure_bg"),
                layout: &pipes.measure_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raw_measure"), timestamp_writes: None });
                pass.set_pipeline(&pipes.measure);
                pass.set_bind_group(0, Some(&bind), &[]);
                pass.dispatch_workgroups(cx.div_ceil(8), cy.div_ceil(8), 1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &cells, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout { offset: eye_bytes * n as u64, bytes_per_row: Some(padded * 16), rows_per_image: Some(cy) },
                },
                wgpu::Extent3d { width: padded, height: cy, depth_or_array_layers: 1 },
            );
        }
        self.queue.submit(Some(encoder.finish()));
        (staging, cx as usize, cy as usize)
    }

    /// The cells of a mapped measurement without the padding of its rows:
    /// 16 bytes each, `cx × cy` an eye, eye after eye.
    fn measured_cells(bytes: &[u8], cx: usize, cy: usize) -> Vec<u8> {
        let padded = cx.div_ceil(16) * 16;
        let mut out = Vec::with_capacity(bytes.len());
        for eye in bytes.chunks_exact(padded * cy * 16) {
            for row in eye.chunks_exact(padded * 16) {
                out.extend_from_slice(&row[..cx * 16]);
            }
        }
        out
    }

    /// Curves to use for Shadows / Highlights instead of the ones derived
    /// from each frame (`None`: derive them again). For measurements.
    pub fn set_raw_local_curves(&self, curves: Option<Arc<LocalCurves>>) {
        if let Ok(mut m) = self.raw_pipelines().memory.lock() {
            m.curves = curves;
        }
    }

    /// What a frame's rendering is derived from: its eyes (camera-linear,
    /// as they enter the colour stack) under `stage`'s white balance.
    /// [`Measure::Wait`] measures this frame and waits; [`Measure::Latest`]
    /// returns what the last finished measurement found and queues one of
    /// this frame when none is on its way.
    pub fn raw_frame_stats(&self, eyes: &[&wgpu::Texture], stage: &RawDevelop) -> FrameStats {
        if eyes.is_empty() {
            return FrameStats::unknown();
        }
        let geometry = stage.geometry();
        let memory = self.raw_pipelines().memory.clone();
        match stage.measure {
            Measure::Wait => {
                let (staging, cx, cy) = self.submit_tone_measurement(eyes, &geometry);
                let (tx, rx) = std::sync::mpsc::channel();
                staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                match rx.recv() {
                    Ok(Ok(())) => {
                        let stats = stats_from_cells(&Self::measured_cells(&staging.slice(..).get_mapped_range(), cx, cy), cx, cy);
                        staging.unmap();
                        stats.unwrap_or_else(FrameStats::unknown)
                    }
                    _ => {
                        tracing::warn!("raw development: the frame's tone keys could not be read back");
                        FrameStats::unknown()
                    }
                }
            }
            Measure::Latest => {
                let (latest, start) = match memory.lock() {
                    Ok(mut m) => {
                        let start = !m.in_flight;
                        m.in_flight = true;
                        (m.latest.clone(), start)
                    }
                    Err(_) => (None, false),
                };
                if start {
                    let (staging, cx, cy) = self.submit_tone_measurement(eyes, &geometry);
                    let buffer = staging.clone();
                    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        let stats = r.ok().and_then(|()| stats_from_cells(&Self::measured_cells(&buffer.slice(..).get_mapped_range(), cx, cy), cx, cy));
                        buffer.unmap();
                        let notify = memory.lock().ok().and_then(|mut m| {
                            m.in_flight = false;
                            let stats = stats?;
                            let changed = !m.latest.as_ref().is_some_and(|old| same_stats(old, &stats));
                            m.latest = Some(stats);
                            changed.then(|| m.notify.clone()).flatten()
                        });
                        if let Some(notify) = notify {
                            notify();
                        }
                    });
                }
                latest.unwrap_or_else(FrameStats::unknown)
            }
        }
    }

    /// The tone keys of a frame (see [`Self::raw_frame_stats`]).
    pub fn raw_tone_keys(&self, eyes: &[&wgpu::Texture], stage: &RawDevelop) -> ToneKeys {
        self.raw_frame_stats(eyes, stage).keys
    }

    /// The per-frame constants of `stage` for the given eyes (measured as
    /// the stage asks). The eyes are one frame: side by side when two.
    pub fn raw_develop_constants(&self, eyes: &[&wgpu::Texture], stage: &RawDevelop) -> DevelopConstants {
        let stats = self.raw_frame_stats(eyes, stage);
        let curves = self.raw_pipelines().memory.lock().ok().and_then(|m| m.curves.clone());
        let mut k = stage.constants_for(&stats, curves.as_deref());
        if let (Some(local), Some(eye)) = (k.local.as_mut(), eyes.first()) {
            local.long_side = (eye.width() * eyes.len() as u32).max(eye.height());
        }
        k
    }

    /// Record the bilateral grid of one eye for a frame's local tone: the
    /// reduced copy, the gathered grid, the smoothed grid. Returns the
    /// smoothed grid and its cell in pixels of the eye.
    fn record_raw_base(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, k: &DevelopConstants, local: &LocalTone) -> (wgpu::Texture, f32) {
        let pipes = self.raw_pipelines();
        let long = if local.long_side > 0 { local.long_side } else { src.width().max(src.height()) };
        let (n, cell) = crate::raw_local::base_layout(long);
        let (rw, rh) = (src.width().div_ceil(n), src.height().div_ceil(n));
        let (gx, gy, gz) = crate::raw_local::grid_dims(rw, rh, cell);
        let reduced = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raw_reduced"),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let grid = |label: &str| self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: gx, height: gy, depth_or_array_layers: gz },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let (gathered, smoothed) = (grid("raw_grid_gathered"), grid("raw_grid"));
        let reduce_params = self.write_uniform("raw_reduce_params", &ReduceParams {
            neutral: [k.geometry.neutral[0], k.geometry.neutral[1], k.geometry.neutral[2], k.gain],
            cam_to_work: k.geometry.camera_to_work.map(row4),
            clip: [CLIP_FROM, CLIP_TO, BASE_FLOOR, 0.0],
            luma: luma4(n as f32),
        });
        let grid_params = self.write_uniform("raw_grid_params", &GridParams { cell: [cell, BASE_LO, 1.0 / BASE_RANGE, 0.0] });
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (src_v, red_v, gat_v, smo_v) = (view(src), view(&reduced), view(&gathered), view(&smoothed));
        fn tex<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let reduce_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_reduce_bg"),
            layout: &pipes.reduce_layout,
            entries: &[tex(0, &src_v), tex(1, &red_v), wgpu::BindGroupEntry { binding: 2, resource: reduce_params.as_entire_binding() }],
        });
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_grid_build_bg"),
            layout: &pipes.grid_build_layout,
            entries: &[tex(0, &red_v), tex(1, &gat_v), wgpu::BindGroupEntry { binding: 2, resource: grid_params.as_entire_binding() }],
        });
        let blur_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_grid_blur_bg"),
            layout: &pipes.grid_blur_layout,
            entries: &[tex(0, &gat_v), tex(1, &smo_v)],
        });
        for (label, pipeline, bind, (x, y)) in [
            ("raw_reduce", &pipes.reduce, &reduce_bg, (rw, rh)),
            ("raw_grid_build", &pipes.grid_build, &build_bg, (gx, gy)),
            ("raw_grid_blur", &pipes.grid_blur, &blur_bg, (gx, gy)),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), 1);
        }
        (smoothed, cell * n as f32)
    }

    fn curve_buffer(&self, label: &str, samples: &[f32]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(samples);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// The development with Shadows / Highlights: the eye's base, then the
    /// stage that reads it.
    fn record_raw_develop_local(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, k: &DevelopConstants, local: &LocalTone) {
        let (grid, cell_px) = self.record_raw_base(encoder, src, k, local);
        let lo = CURVE_LO as f32;
        let last = (local.black.len() - 1) as f32;
        let out_lo = OUT_LO as f32;
        let fed_last = (local.fed.len() - 1) as f32;
        let stage_last = (local.tone.len() - 1) as f32;
        let params = self.write_uniform("raw_develop_local_params", &DevelopLocalParams {
            neutral: [k.geometry.neutral[0], k.geometry.neutral[1], k.geometry.neutral[2], k.gain],
            cam_to_work: k.geometry.camera_to_work.map(row4),
            work_to_out: k.geometry.work_to_out.map(row4),
            black: [lo, last / (local.black_hi - lo), last, 2f32.powf(lo)],
            fed: [out_lo, fed_last / (local.fed_hi - out_lo), fed_last, 2f32.powf(out_lo)],
            gains: [LOCAL_LO, LOCAL_PER_STOP, (local.gain.len() - 1) as f32, local.floor],
            clip: [CLIP_FROM, CLIP_TO, BASE_FLOOR, BASE_FALLBACK],
            luma: luma4(cell_px),
            grid: [BASE_LO, 1.0 / BASE_RANGE, local.white.log2(), 0.0],
            stage: [stage_last / -out_lo, stage_last, 0.0, 0.0],
            keyed: [local.whites.floor, local.blacks.floor, if local.whites.keyed { 1.0 } else { 0.0 }, if local.blacks.keyed { 1.0 } else { 0.0 }],
        });
        let (pre, outc, gains, whites, blacks, tone) = (
            self.curve_buffer("raw_develop_black", &local.black),
            self.curve_buffer("raw_develop_fed", &local.fed),
            self.curve_buffer("raw_develop_gains", &local.gain),
            self.curve_buffer("raw_develop_whites", &local.whites.map),
            self.curve_buffer("raw_develop_blacks", &local.blacks.map),
            self.curve_buffer("raw_develop_tone", &local.tone),
        );
        let pipes = self.raw_pipelines();
        let (src_view, dst_view, grid_view) = (src.create_view(&Default::default()), dst.create_view(&Default::default()), grid.create_view(&Default::default()));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_develop_local_bg"),
            layout: &pipes.develop_local_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pre.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: outc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gains.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&grid_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&pipes.grid_sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: whites.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: blacks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: tone.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raw_develop_local"), timestamp_writes: None });
        pass.set_pipeline(&pipes.develop_local);
        pass.set_bind_group(0, Some(&bind), &[]);
        pass.dispatch_workgroups(dst.width().div_ceil(8), dst.height().div_ceil(8), 1);
    }

    /// Record the raw development stage: `src` (a camera-linear eye) →
    /// `dst` (`Rgba16Unorm`, the same size, display-referred).
    pub(crate) fn record_raw_develop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        k: &DevelopConstants,
    ) {
        if let Some(local) = &k.local {
            return self.record_raw_develop_local(encoder, src, dst, k, local);
        }
        let lo = CURVE_LO as f32;
        let last = (k.curve.len() - 1) as f32;
        let params = self.write_uniform("raw_develop_params", &DevelopParams {
            neutral: [k.geometry.neutral[0], k.geometry.neutral[1], k.geometry.neutral[2], k.gain],
            cam_to_work: k.geometry.camera_to_work.map(row4),
            work_to_out: k.geometry.work_to_out.map(row4),
            curve: [lo, last / (k.curve_hi - lo), last, 2f32.powf(lo)],
            clip: [CLIP_FROM, CLIP_TO, k.floor, 0.0],
        });
        let bytes: &[u8] = bytemuck::cast_slice(&k.curve);
        let curve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_develop_curve"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&curve, 0, bytes);
        let pipes = self.raw_pipelines();
        let src_view = src.create_view(&Default::default());
        let dst_view = dst.create_view(&Default::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_develop_bg"),
            layout: &pipes.develop_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: curve.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raw_develop"), timestamp_writes: None });
        pass.set_pipeline(&pipes.develop);
        pass.set_bind_group(0, Some(&bind), &[]);
        pass.dispatch_workgroups(dst.width().div_ceil(8), dst.height().div_ceil(8), 1);
    }

    /// Develop one camera-linear eye into a new display-referred texture,
    /// with the constants given (see [`Self::raw_develop_constants`]).
    pub fn develop_raw_eye(&self, src: &wgpu::Texture, k: &DevelopConstants) -> wgpu::Texture {
        let dst = self.new_linear_eye_texture(src.width(), src.height());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw_develop") });
        self.record_raw_develop(&mut encoder, src, &dst, k);
        self.queue.submit(Some(encoder.finish()));
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU stage against its CPU twin on random camera pixels.
    #[test]
    #[ignore = "needs a GPU"]
    fn gpu_development_matches_the_cpu_one() {
        use crate::raw_color::RawCamera;
        use crate::raw_tone::RawSettings;
        let dev = Device::new().expect("device");
        let (w, h) = (64u32, 64u32);
        let mut s = 99u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        let px: Vec<u16> = (0..w * h).flat_map(|i| {
            // Mostly dim values like real frames, some near the clip point.
            let scale = if i % 7 == 0 { 65535 } else { 9000 };
            [(next() % scale) as u16, (next() % scale) as u16, (next() % scale) as u16, u16::MAX]
        }).collect();
        let src = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&px),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        for settings in [
            RawSettings::default(),
            RawSettings { exposure: 1.5, ..Default::default() },
            RawSettings { exposure: -2.0, temperature: Some(3200.0), tint: Some(12.0), ..Default::default() },
            RawSettings { blacks: 80.0, whites: -60.0, ..Default::default() },
            RawSettings { blacks: -50.0, exposure: 1.0, ..Default::default() },
            RawSettings { blacks: -80.0, whites: -100.0, ..Default::default() },
        ] {
            let stage = RawDevelop { camera: RawCamera::X6, settings, measure: Measure::Wait };
            // The keys the GPU reads off the picture are the CPU's own of the same cells.
            let keys = dev.raw_tone_keys(&[&src], &stage);
            let g = stage.geometry();
            let cell = (w / 8) as usize;          // 64 wide -> 128 cells would be half a pixel: here 8 px per cell is asked below
            let _ = cell;
            let scenes: Vec<[f32; 3]> = px.chunks_exact(4).map(|c| g.scene([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0))).collect();
            let brightest = scenes.iter().flat_map(|c| c.iter().copied()).fold(0f32, f32::max);
            assert!(keys.max <= brightest * 1.001 && keys.max > brightest * 0.2, "{settings:?}: max {} of {brightest}", keys.max);
            assert!(keys.dark >= 0.0 && keys.dark < keys.max);
            let k = stage.constants(keys);
            let out = dev.develop_raw_eye(&src, &k);
            let got = dev.read_texture_rgba64(&out, w, h).unwrap();
            let got: Vec<u16> = got.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let mut worst = 0f32;
            for (g, c) in got.chunks_exact(4).zip(px.chunks_exact(4)) {
                let want = k.develop([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0));
                for ch in 0..3 {
                    worst = worst.max((g[ch] as f32 / 65535.0 - want[ch]).abs());
                }
            }
            assert!(worst < 1.5e-3, "{settings:?}: worst difference {worst}");
        }
    }

    /// Shadows / Highlights on the GPU against the CPU twin: the base read
    /// from the bilateral grid, the gain of that base, the black-subtracted
    /// signal through it.
    #[test]
    #[ignore = "needs a GPU"]
    fn gpu_local_tone_matches_the_cpu_one() {
        use crate::raw_color::RawCamera;
        use crate::raw_local::{base_layout, reduce_log, BaseGrid};
        use crate::raw_tone::RawSettings;
        let dev = Device::new().expect("device");
        let (w, h) = (192u32, 128u32);
        let mut s = 7u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        // Regions of different brightness with texture inside, and a few specks.
        let px: Vec<u16> = (0..w * h).flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let region = match (x / 48, y / 64) { (0, _) => 300.0, (1, 0) => 2500.0, (1, _) => 900.0, (2, _) => 12000.0, _ => 40000.0 };
            let texture = 0.7 + 0.6 * ((next() % 1000) as f32 / 1000.0);
            let speck = if next() % 211 == 0 { 20.0 } else { 1.0 };
            let v = (region * texture * speck).min(65535.0);
            [(v * 0.55) as u16, v as u16, (v * 0.4) as u16, u16::MAX]
        }).collect();
        let src = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&px),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        for settings in [
            RawSettings { shadows: 100.0, ..Default::default() },
            RawSettings { highlights: -100.0, exposure: 1.0, ..Default::default() },
            RawSettings { shadows: -40.0, highlights: 60.0, ..Default::default() },
            RawSettings { shadows: 70.0, blacks: 60.0, whites: -40.0, ..Default::default() },
        ] {
            let stage = RawDevelop { camera: RawCamera::X6, settings, measure: Measure::Wait };
            let mut k = dev.raw_develop_constants(&[&src], &stage);
            let local = k.local.as_mut().expect("local stage");
            // A long side that gives more than one pixel per texel of the reduced copy.
            local.long_side = 2048;
            assert!(local.gain.iter().any(|g| g.abs() > 0.2), "{settings:?}: no gain");
            let (n, cell) = base_layout(2048);
            let lum: Vec<f32> = px.chunks_exact(4).map(|c| k.base_luminance([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0))).collect();
            let (red, rw, rh) = reduce_log(&lum, w as usize, h as usize, n as usize);
            let grid = BaseGrid::build(&red, rw, rh, cell);
            let out = dev.develop_raw_eye(&src, &k);
            let got = dev.read_texture_rgba64(&out, w, h).unwrap();
            let got: Vec<u16> = got.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let (mut worst, mut sum) = (0f32, 0f64);
            for (i, (g, c)) in got.chunks_exact(4).zip(px.chunks_exact(4)).enumerate() {
                let (x, y) = ((i as u32 % w) as f32, (i as u32 / w) as f32);
                let base = grid.base(x, y, n as f32, lum[i].log2());
                let want = k.develop_local([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0), base);
                for ch in 0..3 {
                    let d = (g[ch] as f32 / 65535.0 - want[ch]).abs();
                    worst = worst.max(d);
                    sum += d as f64;
                }
            }
            let mean = sum / (3 * w * h) as f64;
            // The grid is 16-bit float on the GPU: a few thousandths at most.
            assert!(mean < 6e-4 && worst < 8e-3, "{settings:?}: mean {mean}, worst {worst}");
            // And the stage did something: against the plain development.
            let plain = stage.constants(dev.raw_tone_keys(&[&src], &stage));
            let moved = px.chunks_exact(4).zip(got.chunks_exact(4)).filter(|(c, g)| {
                (plain.develop([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0))[1] - g[1] as f32 / 65535.0).abs() > 0.02
            }).count();
            assert!(moved > (w * h / 10) as usize, "{settings:?}: only {moved} pixels moved");
        }
    }

    /// The reduced copy is the mean of each cell, eye after eye.
    #[test]
    #[ignore = "needs a GPU"]
    fn tone_keys_are_read_off_the_reduced_copy() {
        use crate::raw_color::RawCamera;
        use crate::raw_tone::RawSettings;
        let dev = Device::new().expect("device");
        // Two 252 px eyes: 63 cells of 4 px along each. Left: a flat grey
        // with one bright cell and twelve dark ones; right: flat.
        let (w, h) = (252u32, 252u32);
        let stage = RawDevelop { camera: RawCamera::X6, settings: RawSettings::default(), measure: Measure::Wait };
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let eye = |level_at: &dyn Fn(u32, u32) -> f32| -> wgpu::Texture {
            let px: Vec<u16> = (0..w * h).flat_map(|i| {
                let l = level_at(i % w, i / w);
                [(n[0] * l * 65535.0) as u16, (n[1] * l * 65535.0) as u16, (n[2] * l * 65535.0) as u16, u16::MAX]
            }).collect();
            let t = dev.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_eye"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            dev.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&px),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            t
        };
        let left = eye(&|x, y| if (40..44).contains(&x) && (80..84).contains(&y) { 0.6 } else if x < 48 && y < 4 { 0.01 } else { 0.2 });
        let right = eye(&|_, _| 0.2);
        let stats = dev.raw_frame_stats(&[&left, &right], &stage);
        let keys = stats.keys;
        // 7938 cells; the brightest is the 0.6 one, the darkest one of the
        // twelve dark ones.
        assert!((keys.max - 0.6).abs() < 2e-3, "max {}", keys.max);
        assert!((keys.dark - 0.01).abs() < 5e-4, "dark {}", keys.dark);
        // The brightest value at each level: the grey's, the bright cell's
        // and the dark cells', each at its own level.
        let peak_at = |level: f32| stats.peaks[((level.log2() - LOCAL_LO) * LOCAL_PER_STOP).round() as usize];
        assert!((peak_at(0.2) - 0.2).abs() < 1e-3 && (peak_at(0.6) - keys.max).abs() < 1e-6 && (peak_at(0.01) - 0.01).abs() < 1e-4 && peak_at(0.05) == 0.0);
        assert_eq!(stats.peaks.iter().filter(|p| **p > 0.0).count(), 3);
        // The levels the curves come from: blocks of 3 × 3 cells averaged
        // in log, 21 × 21 an eye, sorted. Nearly all are the grey's.
        assert_eq!(stats.levels.len(), 2 * 21 * 21);
        let grey = 0.2f32.log2();
        assert!((stats.levels[441] - grey).abs() < 0.01, "{} vs {grey}", stats.levels[441]);
        assert!(stats.levels.windows(2).all(|w| w[0] <= w[1]));
        // The four blocks over the dark cells: a third of each is dark.
        let dark_block = (2.0 * grey + 0.01f32.log2()) / 3.0;
        assert!((stats.levels[0] - dark_block).abs() < 0.02, "{} vs {dark_block}", stats.levels[0]);
        // The block with the bright cell: one cell of nine.
        let bright_block = (8.0 * grey + 0.6f32.log2()) / 9.0;
        assert!((stats.levels[881] - bright_block).abs() < 0.02, "{} vs {bright_block}", stats.levels[881]);
        // One eye alone is a frame of its own: 126 cells of 2 px.
        let mono = dev.raw_tone_keys(&[&right], &stage);
        assert!((mono.max - 0.2).abs() < 1e-3 && (mono.dark - 0.2).abs() < 1e-3, "{mono:?}");
    }

    #[test]
    fn the_red_site_of_a_pattern() {
        let l = |cfa| BayerLayout { cfa, black: 64.0, white: 15408.0 };
        assert_eq!(l([1, 2, 0, 1]).red_site(), Some((0, 1)), "GBRG: red at the start of the second row");
        assert_eq!(l([0, 1, 1, 2]).red_site(), Some((0, 0)), "RGGB");
        assert_eq!(l([1, 0, 2, 1]).red_site(), Some((1, 0)), "GRBG");
        assert_eq!(l([2, 1, 1, 0]).red_site(), Some((1, 1)), "BGGR");
        assert_eq!(l([0, 2, 1, 1]).red_site(), None);
        assert_eq!(l([1, 1, 1, 1]).red_site(), None);
    }

    /// The GPU kernels against the CPU demosaic of the stills path
    /// (`raw_develop`), on a random mosaic.
    #[test]
    #[ignore = "needs a GPU"]
    fn gpu_demosaic_matches_the_cpu_one() {
        use crate::raw_develop::{develop_linear, Half, RawParams};
        let dev = Device::new().expect("device");
        let (w, h) = (256u32, 128u32);
        let mut s = 12345u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        let mosaic: Vec<u16> = (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            (2000 + x * 20 + y * 31 + next() % 700) as u16
        }).collect();
        let layout = BayerLayout { cfa: [1, 2, 0, 1], black: 64.0, white: 15408.0 };
        // The CPU path develops one half of a side-by-side frame: put the
        // mosaic in the left half.
        let mut sbs = vec![64u16; (2 * w * h) as usize];
        for y in 0..h as usize {
            sbs[y * 2 * w as usize..y * 2 * w as usize + w as usize].copy_from_slice(&mosaic[y * w as usize..(y + 1) * w as usize]);
        }
        let params = RawParams { width: 2 * w, height: h, cfa: layout.cfa, white_level: layout.white, black_level: [layout.black; 4], as_shot_neutral: [1.0; 3] };
        let tex = dev.new_bayer_texture(w, h);
        dev.write_bayer_texture(&tex, &mosaic);
        for half in [false, true] {
            let cpu = develop_linear(&sbs, &params, &[], Half::Left, half);
            let (ow, oh) = if half { (w / 2, h / 2) } else { (w, h) };
            assert_eq!((cpu.width, cpu.height), (ow, oh));
            let out = dev.new_linear_eye_texture(ow, oh);
            dev.demosaic_bayer(&tex, &layout, &out).unwrap();
            let gpu = dev.read_texture_rgba64(&out, ow, oh).unwrap();
            let gpu: Vec<u16> = gpu.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let mut worst = 0i32;
            for (px, (g, c)) in gpu.chunks_exact(4).zip(cpu.rgba.chunks_exact(4)).enumerate() {
                // The CPU half reflects at the seam of its side-by-side
                // frame exactly as the GPU does at the edge of its eye.
                for ch in 0..3 {
                    let d = (g[ch] as i32 - c[ch] as i32).abs();
                    if d > worst { worst = d; }
                    assert!(d <= 2, "half {half} pixel {px} channel {ch}: gpu {} cpu {}", g[ch], c[ch]);
                }
            }
            eprintln!("half {half}: worst difference {worst} of 65535");
        }
    }
}
