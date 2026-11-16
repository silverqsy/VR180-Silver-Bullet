//! GPU side of raw (Bayer) video: the mosaic goes up as one `R16Uint`
//! texture per eye and comes back as a camera-linear `Rgba16Unorm` eye the
//! fisheye projection kernels sample like any other 16-bit source
//! (`shaders/bayer_demosaic.wgsl`).

use crate::gpu::{Device, EquirectRotation, FisheyeCalib};
use crate::{Error, Result};

const BAYER_MEASURE_WGSL: &str = include_str!("shaders/bayer_measure.wgsl");
const BAYER_GREEN_WGSL: &str = include_str!("shaders/bayer_green.wgsl");
const BAYER_DEMOSAIC_WGSL: &str = include_str!("shaders/bayer_demosaic.wgsl");
const RAW_DEVELOP_WGSL: &str = include_str!("shaders/raw_develop.wgsl");
const RAW_MEASURE_WGSL: &str = include_str!("shaders/raw_measure.wgsl");
const RAW_REDUCE_WGSL: &str = include_str!("shaders/raw_reduce.wgsl");
const RAW_GRID_BUILD_WGSL: &str = include_str!("shaders/raw_grid_build.wgsl");
const RAW_GRID_BLUR_WGSL: &str = include_str!("shaders/raw_grid_blur.wgsl");
const RAW_DEVELOP_LOCAL_WGSL: &str = include_str!("shaders/raw_develop_local.wgsl");
const RAW_SHARPEN_LEVEL_WGSL: &str = include_str!("shaders/raw_sharpen_level.wgsl");
const RAW_SHARPEN_BLUR_WGSL: &str = include_str!("shaders/raw_sharpen_blur.wgsl");
const RAW_SHARPEN_APPLY_WGSL: &str = include_str!("shaders/raw_sharpen_apply.wgsl");

use crate::raw_color::prophoto_luma;
use crate::raw_sharpen::SharpenConstants;
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
    /// The camera's values of a neutral (green = 1): what the mosaic is
    /// balanced by while it is interpolated (see [`crate::bayer`]). The
    /// result is not white balanced by it.
    pub neutral: [f32; 3],
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
    demosaic_measure: wgpu::ComputePipeline,
    demosaic_measure_layout: wgpu::BindGroupLayout,
    demosaic_green: wgpu::ComputePipeline,
    demosaic_green_layout: wgpu::BindGroupLayout,
    demosaic: wgpu::ComputePipeline,
    demosaic_layout: wgpu::BindGroupLayout,
    /// The demosaic's work planes, kept from frame to frame.
    demosaic_work: Mutex<Option<DemosaicWork>>,
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
    sharpen_level: wgpu::ComputePipeline,
    sharpen_edge: wgpu::ComputePipeline,
    sharpen_level_layout: wgpu::BindGroupLayout,
    sharpen_blur: wgpu::ComputePipeline,
    sharpen_blur_layout: wgpu::BindGroupLayout,
    sharpen_apply: wgpu::ComputePipeline,
    sharpen_apply_layout: wgpu::BindGroupLayout,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SharpenLevelParams {
    luma: [f32; 4],
    floor: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SharpenBlurParams {
    dir: [i32; 4],
    coeff: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SharpenApplyParams {
    luma: [f32; 4],
    soft: [f32; 4],
    fade: [f32; 4],
    mask: [f32; 4],
    table: [f32; 4],
}

/// What a raw clip's development is read from when the picture being drawn
/// is another view of the clip. The rendering is derived from the frame it
/// is given (its darkest and brightest parts, the spread of its levels), so
/// the same moment as a reframed view or a lens frame would come out
/// developed differently from its half-equirect. One frame stands for the
/// moment instead: its half-equirect, which every view takes its keys and
/// curves from. See [`Device::raw_reference`].
#[derive(Debug, Clone)]
pub struct RawReference {
    /// The frame's eyes as half-equirects: camera-linear, turned as the
    /// view drawn is.
    pub eyes: Vec<wgpu::Texture>,
    /// The long side the reference frame has at the scale the view drawn
    /// shows the scene at (at its centre). The local stage sizes its
    /// neighbourhoods from it, so they cover the same part of the scene in
    /// every view.
    pub long_side: u32,
}

/// One eye of a [`RawReference`] to draw: the camera-linear source eye, the
/// rotation and per-row matrices of the view, and the eye's calibration as
/// the half-equirect resolves it.
pub struct RawReferenceEye<'a> {
    pub src: &'a wgpu::Texture,
    pub rotation: EquirectRotation,
    pub calib: FisheyeCalib,
    pub rows: Option<&'a [f32]>,
}

/// The side of a reference eye, pixels: 16 to a cell of the measurement.
pub const RAW_REFERENCE_SIDE: u32 = 1008;

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
    /// 1 / the neutral's red and blue, then the two themselves.
    balance: [f32; 4],
}

/// The demosaic's two work planes for a mosaic of `size`: the measures of
/// the first pass (`Rg32Float`) and the green of the second (`R32Float`).
#[derive(Debug)]
struct DemosaicWork {
    size: (u32, u32),
    measure: wgpu::Texture,
    green: wgpu::Texture,
}

impl RawPipelines {
    fn create(device: &wgpu::Device) -> Self {
        use wgpu::TextureViewDimension::{D2, D3};
        let mosaic = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: D2, multisampled: false },
            count: None,
        };
        let demosaic_params = bgl_uniform(2, std::mem::size_of::<DemosaicParams>());
        let demosaic_measure_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bayer_measure_bgl"),
            entries: &[mosaic, bgl_storage(1, wgpu::TextureFormat::Rg32Float, D2), demosaic_params],
        });
        let demosaic_measure = compute_pipeline(device, "bayer_measure", BAYER_MEASURE_WGSL, "main", &demosaic_measure_layout);
        let demosaic_green_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bayer_green_bgl"),
            entries: &[mosaic, bgl_storage(1, wgpu::TextureFormat::R32Float, D2), demosaic_params, bgl_texture(3, D2, false)],
        });
        let demosaic_green = compute_pipeline(device, "bayer_green", BAYER_GREEN_WGSL, "main", &demosaic_green_layout);
        let demosaic_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bayer_demosaic_bgl"),
            entries: &[mosaic, bgl_storage(1, wgpu::TextureFormat::Rgba16Unorm, D2), demosaic_params, bgl_texture(3, D2, false)],
        });
        let demosaic = compute_pipeline(device, "bayer_demosaic", BAYER_DEMOSAIC_WGSL, "main", &demosaic_layout);
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
        // Sharpening of the developed picture: its level, the kernel's
        // blurs summed, the change put back.
        let sharpen_level_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_sharpen_level_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::R32Float, D2),
                bgl_uniform(2, std::mem::size_of::<SharpenLevelParams>()),
            ],
        });
        let sharpen_level = compute_pipeline(device, "raw_sharpen_level", RAW_SHARPEN_LEVEL_WGSL, "level", &sharpen_level_layout);
        let sharpen_edge = compute_pipeline(device, "raw_sharpen_edge", RAW_SHARPEN_LEVEL_WGSL, "edge", &sharpen_level_layout);
        let sharpen_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_sharpen_blur_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::R32Float, D2),
                bgl_uniform(2, std::mem::size_of::<SharpenBlurParams>()),
                bgl_storage_buffer(3),
                bgl_texture(4, D2, false),
                bgl_texture(5, D2, false),
            ],
        });
        let sharpen_blur = compute_pipeline(device, "raw_sharpen_blur", RAW_SHARPEN_BLUR_WGSL, "main", &sharpen_blur_layout);
        let sharpen_apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_sharpen_apply_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::Rgba16Unorm, D2),
                bgl_uniform(2, std::mem::size_of::<SharpenApplyParams>()),
                bgl_texture(3, D2, false),
                bgl_texture(4, D2, false),
                bgl_texture(5, D2, false),
                bgl_storage_buffer(6),
            ],
        });
        let sharpen_apply = compute_pipeline(device, "raw_sharpen_apply", RAW_SHARPEN_APPLY_WGSL, "main", &sharpen_apply_layout);
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
            demosaic_measure, demosaic_measure_layout, demosaic_green, demosaic_green_layout,
            demosaic, demosaic_layout, demosaic_work: Mutex::new(None), develop, develop_layout, measure, measure_layout,
            reduce, reduce_layout, grid_build, grid_build_layout, grid_blur, grid_blur_layout,
            develop_local, develop_local_layout, grid_sampler,
            sharpen_level, sharpen_edge, sharpen_level_layout, sharpen_blur, sharpen_blur_layout, sharpen_apply, sharpen_apply_layout,
            memory: Arc::default(),
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
    /// mosaic's size) or at half size (`out` = half the mosaic's size, each
    /// pixel the mean of the 2×2 it covers). The result is the camera's
    /// linear RGB with the sensor's white level at 1.0. The method is
    /// [`crate::bayer`]'s: three passes (measure, green, colour).
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
        let (nr, nb) = (layout.neutral[0].max(1e-3), layout.neutral[2].max(1e-3));
        let params = self.write_uniform("bayer_demosaic_params", &DemosaicParams {
            black: layout.black,
            scale: 1.0 / span,
            red_x,
            red_y,
            src_w: sw as i32,
            src_h: sh as i32,
            half,
            _pad: 0,
            balance: [1.0 / nr, 1.0 / nb, nr, nb],
        });
        let pipes = self.raw_pipelines();
        // The work planes of a mosaic this size, made once.
        let mut work = pipes.demosaic_work.lock().map_err(|_| Error::Ffmpeg("demosaic: work planes poisoned".into()))?;
        if work.as_ref().map(|w| w.size) != Some((sw, sh)) {
            let plane = |label: &str, format: wgpu::TextureFormat| self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            *work = Some(DemosaicWork {
                size: (sw, sh),
                measure: plane("bayer_measure", wgpu::TextureFormat::Rg32Float),
                green: plane("bayer_green", wgpu::TextureFormat::R32Float),
            });
        }
        let planes = work.as_ref().expect("made above");
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (src_v, dst_v, measure_v, green_v) = (view(bayer), view(out), view(&planes.measure), view(&planes.green));
        fn tex<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let uniform = wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() };
        let measure_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayer_measure_bg"),
            layout: &pipes.demosaic_measure_layout,
            entries: &[tex(0, &src_v), tex(1, &measure_v), uniform.clone()],
        });
        let green_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayer_green_bg"),
            layout: &pipes.demosaic_green_layout,
            entries: &[tex(0, &src_v), tex(1, &green_v), uniform.clone(), tex(3, &measure_v)],
        });
        let colour_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayer_demosaic_bg"),
            layout: &pipes.demosaic_layout,
            entries: &[tex(0, &src_v), tex(1, &dst_v), uniform, tex(3, &green_v)],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bayer_demosaic") });
        for (label, pipeline, bind, (w, h)) in [
            ("bayer_measure", &pipes.demosaic_measure, &measure_bind, (sw, sh)),
            ("bayer_green", &pipes.demosaic_green, &green_bind, (sw, sh)),
            ("bayer_demosaic", &pipes.demosaic, &colour_bind, (out.width(), out.height())),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
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

    /// The reference frame of a raw clip's frame, for a view that is not
    /// the half-equirect: every eye drawn as a half-equirect of
    /// [`RAW_REFERENCE_SIDE`]. `view_px_per_rad`: the density of the view
    /// being drawn, at its centre ([`FisheyeCalib::center_px_per_rad`]).
    /// `slot`: the projection cache slot of the first eye (the next ones
    /// follow it); the textures are the slots' own, good until the next
    /// reference is drawn there.
    pub fn raw_reference(&self, eyes: &[RawReferenceEye<'_>], view_px_per_rad: f32, slot: u32) -> Result<RawReference> {
        let side = RAW_REFERENCE_SIDE;
        let mut drawn = Vec::with_capacity(eyes.len());
        for (n, eye) in eyes.iter().enumerate() {
            let (sw, sh) = (eye.src.width(), eye.src.height());
            let slot = slot + n as u32;
            drawn.push(match eye.rows.filter(|rows| !rows.is_empty()) {
                Some(rows) => self.project_fisheye_rgba16_texture_to_equirect_rs_16(
                    eye.src, sw, sh, side, side, eye.rotation, eye.calib, rows, slot)?,
                None => self.project_fisheye_rgba16_texture_to_equirect_16(
                    eye.src, sw, sh, side, side, eye.rotation, eye.calib, slot)?,
            });
        }
        // A half-equirect eye is π radians wide; the frame is its eyes side by side.
        let long_side = (std::f32::consts::PI * view_px_per_rad * eyes.len() as f32).round().max(1.0) as u32;
        Ok(RawReference { eyes: drawn, long_side })
    }

    /// The per-frame constants of `stage` for the given eyes (measured as
    /// the stage asks). The eyes are one frame: side by side when two.
    /// `reference`: the frame to read instead, when the eyes are another
    /// view of the clip than its half-equirect. `pixel_scale`: pixels of
    /// these eyes to one pixel of the frame an export writes (what the
    /// sharpening's radius is counted in).
    pub fn raw_develop_constants(&self, eyes: &[&wgpu::Texture], reference: Option<&RawReference>, stage: &RawDevelop, pixel_scale: f32) -> DevelopConstants {
        let reference = reference.filter(|r| !r.eyes.is_empty());
        let stats = match reference {
            Some(r) => self.raw_frame_stats(&r.eyes.iter().collect::<Vec<_>>(), stage),
            None => self.raw_frame_stats(eyes, stage),
        };
        tracing::debug!(
            "raw development: dark {:.5} max {:.4}, {} levels (reference: {})",
            stats.keys.dark, stats.keys.max, stats.levels.len(), reference.is_some());
        let curves = self.raw_pipelines().memory.lock().ok().and_then(|m| m.curves.clone());
        let mut k = stage.constants_for(&stats, curves.as_deref());
        if let (Some(local), Some(eye)) = (k.local.as_mut(), eyes.first()) {
            local.long_side = match reference {
                Some(r) => r.long_side,
                None => (eye.width() * eyes.len() as u32).max(eye.height()),
            };
        }
        k.sharpen = stage.settings.sharpen.constants(pixel_scale);
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

    /// Record the sharpening of a developed eye: `src` (the development's
    /// output, sRGB-encoded) → `dst` (`Rgba16Unorm`, the same size). The
    /// level plane, the kernel's five blurs summed into the change of that
    /// level, Masking's edge strength when it is on, and the pass that puts
    /// the change back. CPU twin: [`crate::raw_sharpen::sharpen_picture`].
    pub(crate) fn record_raw_sharpen(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, k: &SharpenConstants) {
        use crate::raw_sharpen::{blur_taps, LIMIT_SAMPLES, LUMA, LUMA_FLOOR, MASK_HI, MASK_LO, MASK_SAMPLES};
        let pipes = self.raw_pipelines();
        let (w, h) = (src.width(), src.height());
        let plane = |label: &str| self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let (level, tmp, sum_a, sum_b) = (plane("raw_sharpen_level"), plane("raw_sharpen_tmp"), plane("raw_sharpen_sum_a"), plane("raw_sharpen_sum_b"));
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (src_v, dst_v, level_v, tmp_v, sum_v) = (view(src), view(dst), view(&level), view(&tmp), [view(&sum_a), view(&sum_b)]);
        fn tex<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let groups = (w.div_ceil(8), h.div_ceil(8));
        let run = |encoder: &mut wgpu::CommandEncoder, label: &str, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        };
        // The level plane (and, for Masking, the size of its gradient).
        let level_params = self.write_uniform("raw_sharpen_level_params", &SharpenLevelParams {
            luma: [LUMA[0], LUMA[1], LUMA[2], 1.0 / k.gamma],
            floor: [LUMA_FLOOR, 0.0, 0.0, 0.0],
        });
        let level_bind = |from: &wgpu::TextureView, to: &wgpu::TextureView| self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_sharpen_level_bg"),
            layout: &pipes.sharpen_level_layout,
            entries: &[tex(0, from), tex(1, to), wgpu::BindGroupEntry { binding: 2, resource: level_params.as_entire_binding() }],
        });
        run(encoder, "raw_sharpen_level", &pipes.sharpen_level, &level_bind(&src_v, &level_v));
        // One pass of a blur: `mode` as the shader's (0 = the blur itself).
        let blur = |encoder: &mut wgpu::CommandEncoder, from: &wgpu::TextureView, to: &wgpu::TextureView, sum: &wgpu::TextureView,
                    taps: &wgpu::Buffer, side: usize, vertical: bool, mode: i32, coeff: f32| {
            let params = self.write_uniform("raw_sharpen_blur_params", &SharpenBlurParams {
                dir: [if vertical { 0 } else { 1 }, if vertical { 1 } else { 0 }, side as i32, mode],
                coeff: [coeff, 0.0, 0.0, 0.0],
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raw_sharpen_blur_bg"),
                layout: &pipes.sharpen_blur_layout,
                entries: &[
                    tex(0, from), tex(1, to),
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: taps.as_entire_binding() },
                    tex(4, &level_v), tex(5, sum),
                ],
            });
            run(encoder, "raw_sharpen_blur", &pipes.sharpen_blur, &bind);
        };
        // The kernel: each blur along x into `tmp`, then along y into the
        // sum (the two sum planes take turns).
        let mut held = 0usize;
        for (n, (sigma, coeff)) in k.sigmas.iter().zip(&k.coeff).enumerate() {
            let taps = blur_taps(*sigma);
            let buffer = self.curve_buffer("raw_sharpen_taps", &taps);
            blur(encoder, &level_v, &tmp_v, &level_v, &buffer, taps.len() - 1, false, 0, 0.0);
            let next = 1 - held;
            blur(encoder, &tmp_v, &sum_v[next], if n == 0 { &level_v } else { &sum_v[held] }, &buffer, taps.len() - 1, true, if n == 0 { 2 } else { 1 }, *coeff);
            held = next;
        }
        // Masking: the gradient's size, spread, into the sum plane not held.
        let free = 1 - held;
        if let Some(m) = &k.mask {
            run(encoder, "raw_sharpen_edge", &pipes.sharpen_edge, &level_bind(&level_v, &sum_v[free]));
            let taps = blur_taps(m.sigma);
            let buffer = self.curve_buffer("raw_sharpen_mask_taps", &taps);
            blur(encoder, &sum_v[free], &tmp_v, &level_v, &buffer, taps.len() - 1, false, 0, 0.0);
            blur(encoder, &tmp_v, &sum_v[free], &level_v, &buffer, taps.len() - 1, true, 0, 0.0);
        }
        // The tables: the two limits, then the mask's curve.
        let mut tables = Vec::with_capacity(2 * LIMIT_SAMPLES + MASK_SAMPLES);
        tables.extend_from_slice(&k.down);
        tables.extend_from_slice(&k.lift);
        match &k.mask {
            Some(m) => tables.extend_from_slice(&m.curve),
            None => tables.resize(2 * LIMIT_SAMPLES + MASK_SAMPLES, 1.0),
        }
        let tables = self.curve_buffer("raw_sharpen_tables", &tables);
        let per_log = (MASK_SAMPLES - 1) as f32 / (MASK_HI - MASK_LO);
        let params = self.write_uniform("raw_sharpen_apply_params", &SharpenApplyParams {
            luma: [LUMA[0], LUMA[1], LUMA[2], k.gamma],
            soft: [k.soft, k.q, k.halo, if k.mask.is_some() { 1.0 } else { 0.0 }],
            fade: k.fade,
            mask: match &k.mask {
                Some(m) => [m.gain, 1.0 / m.threshold, MASK_LO, per_log],
                None => [1.0, 1.0, MASK_LO, per_log],
            },
            table: [(LIMIT_SAMPLES - 1) as f32, (MASK_SAMPLES - 1) as f32, k.window as f32, 0.0],
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_sharpen_apply_bg"),
            layout: &pipes.sharpen_apply_layout,
            entries: &[
                tex(0, &src_v), tex(1, &dst_v),
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                tex(3, &level_v), tex(4, &sum_v[held]),
                tex(5, if k.mask.is_some() { &sum_v[free] } else { &level_v }),
                wgpu::BindGroupEntry { binding: 6, resource: tables.as_entire_binding() },
            ],
        });
        run(encoder, "raw_sharpen_apply", &pipes.sharpen_apply, &bind);
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
            RawSettings { contrast: 70.0, ..Default::default() },
            RawSettings { contrast: -50.0, whites: 40.0, ..Default::default() },
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
            let mut k = dev.raw_develop_constants(&[&src], None, &stage, 1.0);
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

    /// A view that is not the half-equirect is developed from its
    /// reference frame: the keys are the reference's, and the local stage
    /// is sized from it.
    #[test]
    #[ignore = "needs a GPU"]
    fn a_view_is_developed_from_its_reference() {
        use crate::raw_color::RawCamera;
        use crate::raw_tone::RawSettings;
        let dev = Device::new().expect("device");
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let eye = |w: u32, h: u32, level_at: &dyn Fn(u32, u32) -> f32| -> wgpu::Texture {
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
        // The view: a flat grey with a black border (a lens frame's). The
        // reference: the scene, from a deep shadow to a bright wall.
        let view = eye(320, 240, &|x, y| if x < 40 || y < 30 { 0.0 } else { 0.2 });
        let scene = |x: u32, _y: u32| 0.004 * 2f32.powf(x as f32 / 252.0 * 7.0);
        let (ref_l, ref_r) = (eye(252, 252, &scene), eye(252, 252, &scene));
        let settings = RawSettings { shadows: 60.0, highlights: -40.0, ..Default::default() };
        let stage = RawDevelop { camera: RawCamera::X6, settings, measure: Measure::Wait };
        let reference = RawReference { eyes: vec![ref_l.clone(), ref_r.clone()], long_side: 1234 };
        let of_view = dev.raw_develop_constants(&[&view, &view], None, &stage, 1.0);
        let of_reference = dev.raw_develop_constants(&[&ref_l, &ref_r], None, &stage, 1.0);
        let with_reference = dev.raw_develop_constants(&[&view, &view], Some(&reference), &stage, 1.0);
        // Read off the reference, not off the view's own eyes.
        assert_eq!(with_reference.curve, of_reference.curve);
        assert_ne!(with_reference.curve, of_view.curve);
        let (a, b, c) = (with_reference.local.expect("local"), of_reference.local.expect("local"), of_view.local.expect("local"));
        assert_eq!(a.gain, b.gain);
        assert_eq!((a.white, a.black.clone(), a.tone.clone()), (b.white, b.black.clone(), b.tone.clone()));
        assert_ne!(a.gain, c.gain);
        // The neighbourhoods: from the reference's scale, where the others
        // take their own frame's long side.
        assert_eq!((a.long_side, b.long_side, c.long_side), (1234, 504, 640));
        // An empty reference is none.
        let none = RawReference { eyes: Vec::new(), long_side: 99 };
        assert_eq!(dev.raw_develop_constants(&[&view, &view], Some(&none), &stage, 1.0).local.expect("local").long_side, 640);
    }

    /// The sharpening on the GPU against its CPU twin: a picture with
    /// edges hard and soft, texture, colour, deep shadows and near-white
    /// ground, at several settings, with and without Masking, and at a
    /// preview's scale.
    #[test]
    #[ignore = "needs a GPU"]
    fn gpu_sharpening_matches_the_cpu_one() {
        use crate::raw_sharpen::{sharpen_picture, RawSharpen};
        let dev = Device::new().expect("device");
        let (w, h) = (96u32, 80u32);
        let mut s = 7u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s % 10_000) as f32 / 10_000.0 };
        let picture: Vec<[f32; 3]> = (0..w * h).map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            // Columns of ground from near black to near white, a hard edge
            // between each pair, a soft ramp across the middle rows, some
            // texture, and a colour that changes down the picture.
            let ground = [0.02f32, 0.09, 0.2, 0.42, 0.63, 0.8, 0.93, 0.985][(x as usize / 12).min(7)];
            let ramp = if (30.0..50.0).contains(&y) { 0.25 * ((x - 48.0) / 20.0).tanh() } else { 0.0 };
            let v = (ground + ramp + 0.03 * (next() - 0.5)).clamp(0.0, 1.0);
            let tint = y / h as f32;
            [v * (1.0 - 0.3 * tint), v, v * (0.7 + 0.3 * tint)]
        }).collect();
        let px: Vec<u16> = picture.iter().flat_map(|p| [(p[0] * 65535.0).round() as u16, (p[1] * 65535.0).round() as u16, (p[2] * 65535.0).round() as u16, u16::MAX]).collect();
        let exact: Vec<[f32; 3]> = px.chunks_exact(4).map(|c| [c[0], c[1], c[2]].map(|v| v as f32 / 65535.0)).collect();
        let src = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_developed"),
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
        for (settings, scale) in [
            (RawSharpen { amount: 40.0, radius: 1.0, detail: 25.0, masking: 0.0 }, 1.0f32),
            (RawSharpen { amount: 100.0, radius: 2.0, detail: 55.0, masking: 0.0 }, 1.0),
            (RawSharpen { amount: 150.0, radius: 3.0, detail: 100.0, masking: 0.0 }, 1.0),
            (RawSharpen { amount: 70.0, radius: 0.5, detail: 0.0, masking: 0.0 }, 1.0),
            (RawSharpen { amount: 100.0, radius: 1.0, detail: 25.0, masking: 40.0 }, 1.0),
            (RawSharpen { amount: 60.0, radius: 1.4, detail: 40.0, masking: 85.0 }, 1.0),
            (RawSharpen { amount: 80.0, radius: 1.0, detail: 25.0, masking: 30.0 }, 0.25),
            (RawSharpen { amount: 80.0, radius: 3.0, detail: 60.0, masking: 0.0 }, 2.0),
            // Low Detail: the halo rule, over one pixel and over more.
            (RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }, 1.0),
            (RawSharpen { amount: 100.0, radius: 2.0, detail: 10.0, masking: 20.0 }, 1.0),
            (RawSharpen { amount: 150.0, radius: 1.0, detail: 0.0, masking: 0.0 }, 2.0),
            (RawSharpen { amount: 60.0, radius: 3.0, detail: 5.0, masking: 0.0 }, 2.6),
        ] {
            let k = settings.constants(scale).expect("on");
            let dst = dev.new_linear_eye_texture(w, h);
            let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test_sharpen") });
            dev.record_raw_sharpen(&mut encoder, &src, &dst, &k);
            dev.queue.submit(Some(encoder.finish()));
            let bytes = dev.read_texture_rgb48(&dst, w, h).expect("readback");
            let got: Vec<f32> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0).collect();
            let want = sharpen_picture(&exact, w as usize, h as usize, &k);
            let mut worst = 0f32;
            let mut moved = 0usize;
            for (i, (g, wv)) in got.chunks_exact(3).zip(&want).enumerate() {
                for c in 0..3 {
                    worst = worst.max((g[c] - wv[c]).abs());
                }
                if (wv[1] - exact[i][1]).abs() > 2e-3 { moved += 1; }
            }
            // A few 16-bit codes: the two do the same sums in another order.
            assert!(worst < 6e-4, "{settings:?} at scale {scale}: worst difference {worst}");
            assert!(moved > (w * h / 20) as usize, "{settings:?} at scale {scale}: only {moved} pixels changed");
        }
    }

    #[test]
    fn the_red_site_of_a_pattern() {
        let l = |cfa| BayerLayout { cfa, black: 64.0, white: 15408.0, neutral: [1.0; 3] };
        assert_eq!(l([1, 2, 0, 1]).red_site(), Some((0, 1)), "GBRG: red at the start of the second row");
        assert_eq!(l([0, 1, 1, 2]).red_site(), Some((0, 0)), "RGGB");
        assert_eq!(l([1, 0, 2, 1]).red_site(), Some((1, 0)), "GRBG");
        assert_eq!(l([2, 1, 1, 0]).red_site(), Some((1, 1)), "BGGR");
        assert_eq!(l([0, 2, 1, 1]).red_site(), None);
        assert_eq!(l([1, 1, 1, 1]).red_site(), None);
    }

    /// The GPU kernels against the CPU demosaic of the stills path
    /// (`raw_develop`), on a random mosaic.
    /// What a frame's demosaic costs at the X6's size (prints; run with
    /// `--include-ignored --nocapture`).
    #[test]
    #[ignore]
    fn demosaic_timing() {
        let dev = Device::new().expect("device");
        let side = 3840u32;
        let mut s = 99u32;
        let mosaic: Vec<u16> = (0..side * side).map(|i| { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (1000 + (i % side) + s % 900) as u16 }).collect();
        let tex = dev.new_bayer_texture(side, side);
        dev.write_bayer_texture(&tex, &mosaic);
        for half in [false, true] {
            let out_side = if half { side / 2 } else { side };
            let out = dev.new_linear_eye_texture(out_side, out_side);
            // Once to build the pipelines and the work planes, then timed.
            dev.demosaic_bayer(&tex, &crate::x6_raw::BAYER, &out).unwrap();
            let _ = dev.read_texture_rgba64(&out, 8, 8).unwrap();
            let t = std::time::Instant::now();
            let n = 20;
            for _ in 0..n {
                dev.demosaic_bayer(&tex, &crate::x6_raw::BAYER, &out).unwrap();
            }
            let _ = dev.read_texture_rgba64(&out, 8, 8).unwrap();
            eprintln!("demosaic of one {side}x{side} eye, half {half}: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }
    }

    #[test]
    fn gpu_demosaic_matches_the_cpu_one() {
        let dev = Device::new().expect("device");
        let (w, h) = (256u32, 128u32);
        let mut s = 12345u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        // A slope with noise, a bright band across it and one down it, and
        // a patch that reaches the white level.
        let mosaic: Vec<u16> = (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            let band = if (40..44).contains(&y) || (100..103).contains(&x) { 4000 } else { 0 };
            let top = if x > 200 && y > 90 { 9000 } else { 0 };
            (2000 + x * 20 + y * 31 + next() % 700 + band + top).min(16383) as u16
        }).collect();
        let tex = dev.new_bayer_texture(w, h);
        dev.write_bayer_texture(&tex, &mosaic);
        for layout in [crate::x6_raw::BAYER, BayerLayout { cfa: [0, 1, 1, 2], black: 256.0, white: 16000.0, neutral: [0.47, 1.0, 0.61] }] {
            for half in [false, true] {
                let cpu = crate::bayer::demosaic(&mosaic, w as usize, h as usize, &layout, half);
                let (ow, oh) = if half { (w / 2, h / 2) } else { (w, h) };
                assert_eq!(cpu.len(), (ow * oh) as usize);
                let out = dev.new_linear_eye_texture(ow, oh);
                dev.demosaic_bayer(&tex, &layout, &out).unwrap();
                let gpu = dev.read_texture_rgba64(&out, ow, oh).unwrap();
                let gpu: Vec<u16> = gpu.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                let mut worst = 0i32;
                for (px, (g, c)) in gpu.chunks_exact(4).zip(&cpu).enumerate() {
                    for ch in 0..3 {
                        let d = (g[ch] as i32 - (c[ch] * 65535.0).round() as i32).abs();
                        if d > worst { worst = d; }
                        assert!(d <= 2, "{:?} half {half} pixel {px} channel {ch}: gpu {} cpu {}", layout.cfa, g[ch], c[ch] * 65535.0);
                    }
                    assert_eq!(g[3], u16::MAX);
                }
                eprintln!("{:?} half {half}: worst difference {worst} of 65535", layout.cfa);
            }
        }
    }

}
