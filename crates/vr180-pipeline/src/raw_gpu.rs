//! GPU side of raw (Bayer) video: the mosaic goes up as one `R16Uint`
//! texture per eye and comes back as a camera-linear `Rgba16Unorm` eye the
//! fisheye projection kernels sample like any other 16-bit source
//! (`shaders/bayer_demosaic.wgsl`).

use crate::gpu::Device;
use crate::{Error, Result};

const BAYER_DEMOSAIC_WGSL: &str = include_str!("shaders/bayer_demosaic.wgsl");
const RAW_DEVELOP_WGSL: &str = include_str!("shaders/raw_develop.wgsl");

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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DevelopParams {
    camera_white: [f32; 4],
    cam_to_work: [[f32; 4]; 3],
    work_to_out: [[f32; 4]; 3],
    ramp: [f32; 4],
    shoulder: [f32; 4],
    misc: [f32; 4],
}

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
        Self { demosaic, demosaic_layout, develop, develop_layout }
    }
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

impl Device {
    /// Record the raw development stage: `src` (a camera-linear eye) →
    /// `dst` (`Rgba16Unorm`, the same size, display-referred).
    pub(crate) fn record_raw_develop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        stage: &crate::raw_tone::RawDevelop,
    ) {
        let k = stage.constants();
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let params = self.write_uniform("raw_develop_params", &DevelopParams {
            camera_white: row(k.camera_white),
            cam_to_work: k.camera_to_work.map(row),
            work_to_out: k.work_to_out.map(row),
            ramp: [k.ramp.slope, k.ramp.black, k.ramp.radius, k.ramp.q_scale],
            shoulder: [k.shoulder.slope, k.shoulder.a, k.shoulder.b, k.shoulder.c],
            misc: [(k.curve.len() - 1) as f32, 0.0, 0.0, 0.0],
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

    /// Develop one camera-linear eye into a new display-referred texture.
    pub fn develop_raw_eye(&self, src: &wgpu::Texture, stage: &crate::raw_tone::RawDevelop) -> wgpu::Texture {
        let dst = self.new_linear_eye_texture(src.width(), src.height());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw_develop") });
        self.record_raw_develop(&mut encoder, src, &dst, stage);
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
        use crate::raw_tone::{RawDevelop, RawSettings};
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
        ] {
            let stage = RawDevelop { camera: RawCamera::X6, settings };
            let k = stage.constants();
            let out = dev.develop_raw_eye(&src, &stage);
            let got = dev.read_texture_rgba64(&out, w, h).unwrap();
            let got: Vec<u16> = got.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let mut worst = 0f32;
            for (g, c) in got.chunks_exact(4).zip(px.chunks_exact(4)) {
                let want = k.develop([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0));
                for ch in 0..3 {
                    worst = worst.max((g[ch] as f32 / 65535.0 - want[ch]).abs());
                }
            }
            assert!(worst < 4e-4, "{settings:?}: worst difference {worst}");
        }
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
