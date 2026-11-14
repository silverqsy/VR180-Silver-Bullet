//! The fisheye projection: one eye of a fisheye source (DJI OSV, Insta360,
//! SBS fisheye, Blackmagic RAW, Kandao) to one eye of the output, with the
//! per-frame rotation and, optionally, the per-row rolling-shutter
//! correction.
//!
//! One kernel source (`shaders/fisheye_project.wgsl`), composed into the
//! builds of [`FisheyeKernel`]; one pipeline per build
//! ([`ProjectionKernels`]); one entry that runs any of them:
//! [`Device::project_eye`], which also puts a CPU-decoded picture on the
//! GPU first.

use super::{Device, EquirectRotation, EquirectUniforms, FisheyeCalib, FisheyeCalibUniforms};
use crate::{Error, Result};

/// WGSL source of the fisheye projection kernel — every build of it. The
/// lens models, the masks and the ray generators live here once;
/// [`FisheyeKernel::wgsl`] composes a build. CPU reference for testing:
/// `crates/vr180-fisheye/src/projection.rs`.
const FISHEYE_PROJECT_WGSL: &str = include_str!("../shaders/fisheye_project.wgsl");

/// One build of the fisheye projection kernel: the flags of
/// `shaders/fisheye_project.wgsl` plus the width of what it stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FisheyeKernel {
    /// The source is a Y plane + a CbCr plane (a decoded surface sampled in
    /// place, converted to RGB in the kernel) instead of an RGBA texture.
    pub(super) yuv: bool,
    /// Per-row rolling-shutter correction (one more binding: the rows).
    pub(super) rs: bool,
    /// Equidistant fisheye output instead of half-equirect / reframed.
    pub(super) fisheye_out: bool,
    /// Stores `Rgba16Unorm` instead of `Rgba8Unorm`. A plane-sourced build
    /// only exists wide: its source is a 10-bit frame.
    pub(super) wide: bool,
}

impl FisheyeKernel {
    /// Every build there is a pipeline for.
    fn all() -> impl Iterator<Item = Self> {
        (0..16usize).map(Self::from_index).filter(|k| k.wide || !k.yuv)
    }

    fn index(self) -> usize {
        self.yuv as usize | (self.rs as usize) << 1 | (self.fisheye_out as usize) << 2 | (self.wide as usize) << 3
    }

    fn from_index(i: usize) -> Self {
        Self { yuv: i & 1 != 0, rs: i & 2 != 0, fisheye_out: i & 4 != 0, wide: i & 8 != 0 }
    }

    fn out_format(self) -> wgpu::TextureFormat {
        if self.wide { wgpu::TextureFormat::Rgba16Unorm } else { wgpu::TextureFormat::Rgba8Unorm }
    }

    fn label(self) -> String {
        format!(
            "fisheye_project_{}_{}{}_{}",
            if self.yuv { "planes" } else { "rgba" },
            if self.fisheye_out { "fisheye" } else { "equirect" },
            if self.rs { "_rs" } else { "" },
            if self.wide { 16 } else { 8 },
        )
    }

    /// The composed WGSL of this build.
    pub(super) fn wgsl(self) -> String {
        let mut flags: Vec<&str> = Vec::with_capacity(4);
        if self.yuv { flags.push("YUV"); }
        if self.rs { flags.push("RS"); }
        if self.fisheye_out { flags.push("FISHEYE_OUT"); }
        // As the builds were written: only this one hands the corrected ray
        // on to the frame-cover test.
        if self.yuv && self.rs && !self.fisheye_out { flags.push("RS_UPDATES_RAY"); }
        let text = crate::shader_compose::compose(FISHEYE_PROJECT_WGSL, &flags);
        // The source names the storage format of its family: a plane build
        // is written wide, an RGBA build narrow.
        if self.wide && !self.yuv {
            text.replace("rgba8unorm, write", "rgba16unorm, write")
        } else {
            text
        }
    }
}

/// A build of the kernel, compiled. Bindings, in order: the source texture
/// (two for a plane build: luma, chroma), the sampler, the output storage
/// texture, the per-frame rotation, the calibration and, for a
/// rolling-shutter build, the per-row matrices.
#[derive(Debug)]
struct ProjectionPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl ProjectionPipeline {
    fn create(device: &wgpu::Device, kernel: FisheyeKernel) -> Self {
        let label = kernel.label();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(kernel.wgsl().into()),
        });
        let uniform = |size: usize| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size as u64),
        };
        let mut types = vec![
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            };
            if kernel.yuv { 2 } else { 1 }
        ];
        types.push(wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering));
        types.push(wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: kernel.out_format(),
            view_dimension: wgpu::TextureViewDimension::D2,
        });
        types.push(uniform(std::mem::size_of::<EquirectUniforms>()));
        types.push(uniform(std::mem::size_of::<FisheyeCalibUniforms>()));
        if kernel.rs {
            types.push(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            });
        }
        let entries: Vec<wgpu::BindGroupLayoutEntry> = types.into_iter().enumerate()
            .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &entries,
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

/// The compiled builds of the projection kernel, by [`FisheyeKernel`].
#[derive(Debug)]
pub(super) struct ProjectionKernels {
    table: [Option<ProjectionPipeline>; 16],
}

impl ProjectionKernels {
    pub(super) fn create(device: &wgpu::Device) -> Self {
        let mut table: [Option<ProjectionPipeline>; 16] = Default::default();
        for kernel in FisheyeKernel::all() {
            table[kernel.index()] = Some(ProjectionPipeline::create(device, kernel));
        }
        Self { table }
    }

    fn get(&self, kernel: FisheyeKernel) -> Option<&ProjectionPipeline> {
        self.table[kernel.index()].as_ref()
    }
}

/// One eye's source picture, as the kernel samples it.
#[derive(Debug, Clone, Copy)]
pub enum EyePicture<'a> {
    /// An RGBA texture, 8 or 16 bits a channel.
    Rgba(&'a wgpu::TextureView),
    /// A 4:2:0 frame sampled in place: its luma plane and its half-size
    /// chroma plane. The calibration says how the samples are encoded and
    /// where the eye sits in the planes.
    Planes { y: &'a wgpu::TextureView, uv: &'a wgpu::TextureView },
    /// `w × h` RGBA8 pixels from the CPU, uploaded for this call into the
    /// texture kept for `slot`: a caller that uploads every frame (the
    /// CPU-decode preview and export) does not create a texture every
    /// frame. The slot stays locked until the projection is submitted, so
    /// two threads on one slot cannot read each other's upload; eyes
    /// projected in the same frame still need a slot each.
    Bytes8 { rgba: &'a [u8], w: u32, h: u32, slot: u32 },
    /// `w × h` pixels from the CPU, uploaded as a 16-bit texture of their
    /// own: `bit_depth` 16 is RGBA64LE, copied as it is; 8 is RGBA8, each
    /// byte repeated (255 becomes 65535, the same unorm value; a plain shift
    /// leaves white at 65280, 0.39 % dark).
    Bytes16 { bytes: &'a [u8], bit_depth: u8, w: u32, h: u32 },
}

/// Per-row correction is applied to both eyes or to neither: the rows of
/// one eye alone are dropped.
pub fn rows_for_both<'a>(
    left: Option<&'a [f32]>, right: Option<&'a [f32]>,
) -> (Option<&'a [f32]>, Option<&'a [f32]>) {
    match (left, right) {
        (Some(l), Some(r)) => (Some(l), Some(r)),
        _ => (None, None),
    }
}

/// Where a projected eye lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeOutput {
    /// A texture of its own, the caller's to keep.
    New,
    /// The texture kept for this slot, written again on every call: for a
    /// caller that projects every frame and is done with the last result
    /// before it asks for the next. Concurrent outputs need distinct slots.
    Slot(u32),
}

/// A request to [`Device::project_eye`].
#[derive(Debug, Clone, Copy)]
pub struct EyeProjection<'a> {
    pub picture: EyePicture<'a>,
    pub out_w: u32,
    pub out_h: u32,
    /// Store 16 bits a channel (`Rgba16Unorm`) instead of 8.
    pub wide: bool,
    /// Equidistant fisheye out instead of half-equirect / the reframed view.
    pub fisheye_out: bool,
    pub rotation: EquirectRotation,
    pub calib: FisheyeCalib,
    /// Per-row matrices, 12 `f32` a source row: selects the rolling-shutter
    /// build of the kernel. An empty slice still selects it, with the one
    /// all-zero row it reads as "no correction".
    pub rs_rows: Option<&'a [f32]>,
    pub output: EyeOutput,
}

impl<'a> EyeProjection<'a> {
    /// Half-equirect (or the reframed view the calibration describes),
    /// 8 bits a channel, no per-row correction, into a new texture.
    pub fn new(
        picture: EyePicture<'a>,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
    ) -> Self {
        Self {
            picture, out_w, out_h, wide: false, fisheye_out: false,
            rotation, calib, rs_rows: None, output: EyeOutput::New,
        }
    }

    pub fn wide(mut self) -> Self { self.wide = true; self }
    pub fn fisheye_out(mut self, on: bool) -> Self { self.fisheye_out = on; self }
    pub fn rs_rows(mut self, rows: Option<&'a [f32]>) -> Self { self.rs_rows = rows; self }
    pub fn slot(mut self, slot: u32) -> Self { self.output = EyeOutput::Slot(slot); self }
}

const OUT_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC);
const SRC_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::COPY_DST);

fn texture(
    device: &wgpu::Device, label: &str, w: u32, h: u32,
    format: wgpu::TextureFormat, usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

impl Device {
    /// Project one eye: run the build of the kernel the request selects
    /// (source kind, output kind, width, per-row correction) from the
    /// picture into a texture of the output size, and return that texture.
    pub fn project_eye(&self, p: &EyeProjection<'_>) -> Result<wgpu::Texture> {
        match p.picture {
            EyePicture::Rgba(_) | EyePicture::Planes { .. } => self.run_projection(p),
            EyePicture::Bytes8 { rgba, w, h, slot } => {
                debug_assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4);
                let mut cache = self.eye_upload_cache.lock().unwrap();
                let hit = matches!(cache.get(&slot), Some((cw, ch, _)) if *cw == w && *ch == h);
                if !hit {
                    cache.insert(slot, (w, h,
                        texture(&self.device, "eye_uploaded_slot", w, h, wgpu::TextureFormat::Rgba8Unorm, SRC_USAGE)));
                }
                let tex = &cache.get(&slot).unwrap().2;
                self.write_eye(tex, rgba, w, h);
                let view = tex.create_view(&Default::default());
                // `cache` is held to the end of this arm: see `Bytes8`.
                self.run_projection(&EyeProjection { picture: EyePicture::Rgba(&view), ..*p })
            }
            EyePicture::Bytes16 { bytes, bit_depth, w, h } => {
                let widened: std::borrow::Cow<[u8]> = if bit_depth >= 16 {
                    std::borrow::Cow::Borrowed(bytes)
                } else {
                    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 8);
                    for px in bytes.chunks_exact(4) {
                        out.extend_from_slice(&[px[0], px[0], px[1], px[1], px[2], px[2], 0xFF, 0xFF]);
                    }
                    std::borrow::Cow::Owned(out)
                };
                debug_assert_eq!(widened.len(), (w as usize) * (h as usize) * 8);
                let tex = texture(&self.device, "eye_uploaded_16", w, h, wgpu::TextureFormat::Rgba16Unorm, SRC_USAGE);
                self.write_eye(&tex, &widened, w, h);
                let view = tex.create_view(&Default::default());
                self.run_projection(&EyeProjection { picture: EyePicture::Rgba(&view), ..*p })
            }
        }
    }

    /// Project a frame's eyes. A frame with one eye (`right` absent) returns
    /// that eye twice.
    pub fn project_eyes(
        &self,
        left: &EyeProjection<'_>,
        right: Option<&EyeProjection<'_>>,
    ) -> Result<(wgpu::Texture, wgpu::Texture)> {
        let l = self.project_eye(left)?;
        let r = match right {
            Some(right) => self.project_eye(right)?,
            None => l.clone(),
        };
        Ok((l, r))
    }

    /// [`Self::project_eye`] for a picture that is on the GPU.
    fn run_projection(&self, p: &EyeProjection<'_>) -> Result<wgpu::Texture> {
        let kernel = FisheyeKernel {
            yuv: matches!(p.picture, EyePicture::Planes { .. }),
            rs: p.rs_rows.is_some(),
            fisheye_out: p.fisheye_out,
            wide: p.wide,
        };
        let pl = self.projection.get(kernel)
            .ok_or_else(|| Error::Wgpu(format!("no projection kernel for {kernel:?}")))?;
        let format = kernel.out_format();
        let out = match p.output {
            EyeOutput::New => texture(&self.device, "eye_projected", p.out_w, p.out_h, format, OUT_USAGE),
            // Kept per slot so a thread that projects every frame does not
            // `create_texture` every frame (that contends with the UI thread
            // on the shared device). Safe to write again: whoever read the
            // last result did so before this dispatch, on the same queue.
            EyeOutput::Slot(slot) => {
                let mut cache = self.rgba16_eq_out_cache.lock().unwrap();
                let hit = matches!(cache.get(&slot),
                    Some((w, h, t)) if *w == p.out_w && *h == p.out_h && t.format() == format);
                if !hit {
                    cache.insert(slot, (p.out_w, p.out_h,
                        texture(&self.device, "eye_projected_slot", p.out_w, p.out_h, format, OUT_USAGE)));
                }
                cache.get(&slot).unwrap().2.clone()
            }
        };
        let out_view = out.create_view(&Default::default());
        let rotation = self.write_uniform("eye_rotation", &EquirectUniforms::from_mat3(p.rotation.0));
        let calib = self.write_uniform("eye_calib", &FisheyeCalibUniforms::from_public(p.calib));
        let rows = p.rs_rows.map(|rows| {
            // At least one row: an all-zero one trips the kernel's
            // `r00 != 0` test and leaves the per-row pass out.
            let bytes: &[u8] = if rows.is_empty() { &[0u8; 48] } else { bytemuck::cast_slice(rows) };
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("eye_rs_rows"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            buf
        });

        let mut resources: Vec<wgpu::BindingResource> = Vec::with_capacity(7);
        match p.picture {
            EyePicture::Rgba(view) => resources.push(wgpu::BindingResource::TextureView(view)),
            EyePicture::Planes { y, uv } => {
                resources.push(wgpu::BindingResource::TextureView(y));
                resources.push(wgpu::BindingResource::TextureView(uv));
            }
            EyePicture::Bytes8 { .. } | EyePicture::Bytes16 { .. } =>
                unreachable!("project_eye uploads bytes before it gets here"),
        }
        resources.push(wgpu::BindingResource::Sampler(&pl.sampler));
        resources.push(wgpu::BindingResource::TextureView(&out_view));
        resources.push(rotation.as_entire_binding());
        resources.push(calib.as_entire_binding());
        if let Some(rows) = &rows {
            resources.push(rows.as_entire_binding());
        }
        let entries: Vec<wgpu::BindGroupEntry> = resources.into_iter().enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding: binding as u32, resource })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eye_projection_bg"),
            layout: &pl.bind_group_layout,
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eye_projection_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eye_projection_pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&pl.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((p.out_w + 7) / 8, (p.out_h + 7) / 8, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    fn write_eye(&self, tex: &wgpu::Texture, bytes: &[u8], w: u32, h: u32) {
        self.queue.write_texture(
            tex.as_image_copy(), bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes.len() as u32 / h),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    // ── What the Windows arms still call ────────────────────────────
    //
    // The D3D11 zero-copy preview and the two D3D11 exports project an
    // on-GPU `Rgba16Unorm` eye into the output kept for `slot` through
    // these four names. They cannot be compiled on a Mac, so they keep
    // their calls until those arms move onto the shared render flow.
    // `src_w` / `src_h` are part of the calibration and are not read.

    pub fn project_fisheye_rgba16_texture_to_equirect_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h);
        let view = src_tex.create_view(&Default::default());
        self.project_eye(&EyeProjection::new(EyePicture::Rgba(&view), out_w, out_h, rotation, calib)
            .wide().slot(slot))
    }

    pub fn project_fisheye_rgba16_texture_to_fisheye_16(
        &self,
        src_tex: &wgpu::Texture,
        src_w: u32, src_h: u32,
        out_w: u32, out_h: u32,
        rotation: EquirectRotation,
        calib: FisheyeCalib,
        slot: u32,
    ) -> Result<wgpu::Texture> {
        let _ = (src_w, src_h);
        let view = src_tex.create_view(&Default::default());
        self.project_eye(&EyeProjection::new(EyePicture::Rgba(&view), out_w, out_h, rotation, calib)
            .wide().fisheye_out(true).slot(slot))
    }

    /// `per_row_matrices_f32`: `src_h * 12` floats; an empty slice turns the
    /// correction off.
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
        let _ = (src_w, src_h);
        let view = src_tex.create_view(&Default::default());
        self.project_eye(&EyeProjection::new(EyePicture::Rgba(&view), out_w, out_h, rotation, calib)
            .wide().fisheye_out(true).rs_rows(Some(per_row_matrices_f32)).slot(slot))
    }

    /// `per_row_matrices_f32`: `src_h * 12` floats, from
    /// `pack_per_row_camera_matrices`; an empty slice turns the correction
    /// off.
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
        let _ = (src_w, src_h);
        let view = src_tex.create_view(&Default::default());
        self.project_eye(&EyeProjection::new(EyePicture::Rgba(&view), out_w, out_h, rotation, calib)
            .wide().rs_rows(Some(per_row_matrices_f32)).slot(slot))
    }
}

#[cfg(test)]
mod fisheye_kernel_tests {
    use super::FisheyeKernel;

    /// Every build parses and validates as WGSL on its own.
    #[test]
    fn every_build_is_valid_wgsl() {
        for kernel in FisheyeKernel::all() {
            let module = wgpu::naga::front::wgsl::parse_str(&kernel.wgsl())
                .unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            ).validate(&module).unwrap_or_else(|e| panic!("{kernel:?}: {e:?}"));
        }
    }

    /// Twelve builds: every RGBA one at both widths, the plane ones wide.
    #[test]
    fn the_builds_and_their_indices() {
        let all: Vec<FisheyeKernel> = FisheyeKernel::all().collect();
        assert_eq!(all.len(), 12);
        assert!(all.iter().all(|k| k.wide || !k.yuv));
        for k in &all {
            assert_eq!(FisheyeKernel::from_index(k.index()), *k);
            let stores_wide = k.wgsl().contains("rgba16unorm, write");
            assert_eq!(stores_wide, k.wide, "{k:?}");
        }
    }
}

/// Every build of the projection kernel through every kind of picture, on
/// fixed synthetic pictures, each output reduced to a hash. It pins what the projection code computes
/// across a change to that code, on one machine:
///
/// ```text
/// VR180_FINGERPRINTS=/some/file cargo test -p vr180-pipeline --lib \
///     projection_fingerprints -- --ignored
/// ```
///
/// The first run writes the file, later runs compare against it. The hashes
/// belong to the GPU and driver they were made on, so the file is not kept
/// in the repository and the test does not run by default.
#[cfg(test)]
mod projection_fingerprints {
    use super::super::{Device, EquirectRotation, FisheyeCalib};
    use super::{EyePicture, EyeProjection};
    use crate::Result;

    const SW: u32 = 160;
    const SH: u32 = 144;
    const OW: u32 = 136;
    const OH: u32 = 120;

    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// Repeatable noise: bilinear taps of it move with the smallest change
    /// in a sampling position.
    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n).map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        }).collect()
    }

    fn texture(dev: &Device, format: wgpu::TextureFormat, w: u32, h: u32, bytes: &[u8]) -> wgpu::Texture {
        let tex = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fingerprint_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        dev.queue.write_texture(
            tex.as_image_copy(), bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0, bytes_per_row: Some(bytes.len() as u32 / h), rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        tex
    }

    fn kb() -> FisheyeCalib {
        FisheyeCalib::new(46.0, 46.5, 81.3, 70.6, [0.02, -0.01, 0.003, -0.0005], SW as f32, SH as f32, 78.0)
    }

    /// The factory-model branch of the kernels, as a crop with a view mask.
    fn omni() -> FisheyeCalib {
        FisheyeCalib::new_omni(
            235.0, 235.4, 79.6, 71.2, 2.45543,
            [1.28, -0.81, 1.15, 12.03, 0.0], [1e-4, -2e-4, 0.0, 0.0], [1e-4, 0.0, -1e-4, 0.0],
            SW as f32, SH as f32,
        ).with_frame_crop().with_view_mask([0.7, 0.6], 0.1, 0.05)
    }

    fn reframed() -> FisheyeCalib {
        kb().with_reframe(90.0, 0.5, OW as f32 / OH as f32)
    }

    fn rotation() -> EquirectRotation {
        let (y, p, r) = (7.0f32.to_radians(), -4.0f32.to_radians(), 3.0f32.to_radians());
        let (cy, sy, cp, sp, cr, sr) = (y.cos(), y.sin(), p.cos(), p.sin(), r.cos(), r.sin());
        EquirectRotation([
            cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp,
            cp * sr, cp * cr, -sp,
            -sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp,
        ])
    }

    /// One small rotation per source row, in the kernels' 12-float layout.
    fn rows() -> Vec<f32> {
        let mut out = Vec::with_capacity(SH as usize * 12);
        for i in 0..SH {
            let a = 0.004 * (i as f32 / SH as f32 - 0.5);
            let (c, s) = (a.cos(), a.sin());
            out.extend_from_slice(&[c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0]);
        }
        out
    }

    struct Run<'a> {
        dev: &'a Device,
        lines: Vec<(String, u64)>,
    }

    impl Run<'_> {
        fn eight(&mut self, name: &str, tex: Result<wgpu::Texture>) {
            let tex = tex.unwrap_or_else(|e| panic!("{name}: {e}"));
            let px = self.dev.read_texture_rgb8(&tex, OW, OH).expect("readback");
            self.lines.push((name.to_string(), fnv(&px)));
        }
        fn sixteen(&mut self, name: &str, tex: Result<wgpu::Texture>) {
            let tex = tex.unwrap_or_else(|e| panic!("{name}: {e}"));
            let px = self.dev.read_texture_rgba64(&tex, OW, OH).expect("readback");
            self.lines.push((name.to_string(), fnv(&px)));
        }
    }

    #[test]
    #[ignore = "needs VR180_FINGERPRINTS and the GPU the file was made on"]
    fn every_projection_entry_keeps_its_output() {
        let path = std::env::var("VR180_FINGERPRINTS").expect("VR180_FINGERPRINTS=<file>");
        let dev = Device::new().expect("device");
        let (sw, sh) = (SW, SH);
        let rgba8 = noise((sw * sh * 4) as usize, 11);
        let rgba64 = noise((sw * sh * 8) as usize, 23);
        let tex8 = texture(&dev, wgpu::TextureFormat::Rgba8Unorm, sw, sh, &rgba8);
        let tex16 = texture(&dev, wgpu::TextureFormat::Rgba16Unorm, sw, sh, &rgba64);
        let rows = rows();
        let mut run = Run { dev: &dev, lines: Vec::new() };

        let (v8, v16) = (tex8.create_view(&Default::default()), tex16.create_view(&Default::default()));
        fn req<'a>(picture: EyePicture<'a>, calib: FisheyeCalib) -> EyeProjection<'a> {
            EyeProjection::new(picture, OW, OH, rotation(), calib)
        }
        let bytes8 = |slot: u32| EyePicture::Bytes8 { rgba: &rgba8, w: sw, h: sh, slot };

        // RGBA8 bytes in.
        run.eight("bytes8_equirect_readback kb", dev.project_eye(&req(bytes8(2), kb())));
        run.eight("bytes8_equirect kb", dev.project_eye(&req(bytes8(0), kb())));
        run.eight("bytes8_equirect omni", dev.project_eye(&req(bytes8(0), omni())));
        run.eight("bytes8_equirect reframed slot1", dev.project_eye(&req(bytes8(1), reframed())));
        run.eight("bytes8_equirect_rs kb", dev.project_eye(&req(bytes8(0), kb()).rs_rows(Some(&rows))));
        run.eight("bytes8_equirect_rs omni", dev.project_eye(&req(bytes8(0), omni()).rs_rows(Some(&rows))));
        run.eight("bytes8_fisheye kb", dev.project_eye(&req(bytes8(0), kb()).fisheye_out(true)));
        run.eight("bytes8_fisheye omni", dev.project_eye(&req(bytes8(0), omni()).fisheye_out(true)));
        run.eight("bytes8_fisheye_rs kb", dev.project_eye(&req(bytes8(0), kb()).fisheye_out(true).rs_rows(Some(&rows))));
        run.eight("bytes8_fisheye_rs no rows", dev.project_eye(&req(bytes8(0), kb()).fisheye_out(true).rs_rows(Some(&[]))));

        // An RGBA8 texture in. The last of each: rows asked for and none
        // given, which a caller filters to "no correction".
        for (fisheye_out, out) in [(false, "equirect"), (true, "fisheye")] {
            let from_tex8 = |calib| req(EyePicture::Rgba(&v8), calib).fisheye_out(fisheye_out);
            run.eight(&format!("tex8_{out} kb"), dev.project_eye(&from_tex8(kb())));
            run.eight(&format!("tex8_{out} omni"), dev.project_eye(&from_tex8(omni())));
            run.eight(&format!("tex8_{out}_rs kb"), dev.project_eye(&from_tex8(kb()).rs_rows(Some(&rows))));
            run.eight(&format!("tex8_{out}_rs no rows"), dev.project_eye(&from_tex8(kb()).rs_rows(None)));
        }

        // 16-bit bytes in, and 8-bit bytes widened.
        let bytes16 = EyePicture::Bytes16 { bytes: &rgba64, bit_depth: 16, w: sw, h: sh };
        let widened = EyePicture::Bytes16 { bytes: &rgba8, bit_depth: 8, w: sw, h: sh };
        run.sixteen("bytes16_equirect kb", dev.project_eye(&req(bytes16, kb()).wide()));
        run.sixteen("bytes16_equirect omni", dev.project_eye(&req(bytes16, omni()).wide()));
        run.sixteen("bytes16_fisheye kb", dev.project_eye(&req(bytes16, kb()).wide().fisheye_out(true)));
        run.sixteen("bytes8_fisheye_16 kb", dev.project_eye(&req(widened, kb()).wide().fisheye_out(true)));

        // A 16-bit RGBA texture in, the output kept per slot.
        let from_tex16 = |calib, slot| req(EyePicture::Rgba(&v16), calib).wide().slot(slot);
        for (calib, lens) in [(kb(), "kb"), (omni(), "omni"), (reframed(), "reframed")] {
            run.sixteen(&format!("tex16_equirect {lens}"), dev.project_eye(&from_tex16(calib, 0)));
            run.sixteen(&format!("tex16_fisheye {lens}"), dev.project_eye(&from_tex16(calib, 1).fisheye_out(true)));
            run.sixteen(&format!("tex16_equirect_rs {lens}"), dev.project_eye(&from_tex16(calib, 0).rs_rows(Some(&rows))));
            run.sixteen(&format!("tex16_fisheye_rs {lens}"), dev.project_eye(&from_tex16(calib, 1).fisheye_out(true).rs_rows(Some(&rows))));
        }
        run.sixteen("tex16_equirect_rs no rows", dev.project_eye(&from_tex16(kb(), 0).rs_rows(Some(&[]))));
        run.sixteen("tex16_fisheye_rs no rows", dev.project_eye(&from_tex16(kb(), 1).fisheye_out(true).rs_rows(Some(&[]))));

        // Luma + chroma planes in: a frame two eyes wide, so the second eye
        // is read in place through the calibration's plane offset.
        {
            let y = texture(&dev, wgpu::TextureFormat::R16Unorm, sw * 2, sh, &noise((sw * 2 * sh * 2) as usize, 37));
            let uv = texture(&dev, wgpu::TextureFormat::Rg16Unorm, sw, sh / 2, &noise((sw * (sh / 2) * 4) as usize, 41));
            let (y, uv) = (y.create_view(&Default::default()), uv.create_view(&Default::default()));
            let planes = |calib| req(EyePicture::Planes { y: &y, uv: &uv }, calib).wide();
            let second = |mut c: FisheyeCalib| { c.src_x0 = sw as f32; c };
            for (calib, lens) in [(kb(), "kb"), (omni(), "omni"), (second(kb()), "kb second eye"), (reframed(), "reframed")] {
                run.sixteen(&format!("planes_equirect {lens}"), dev.project_eye(&planes(calib)));
                run.sixteen(&format!("planes_fisheye {lens}"), dev.project_eye(&planes(calib).fisheye_out(true)));
                run.sixteen(&format!("planes_equirect_rs {lens}"), dev.project_eye(&planes(calib).rs_rows(Some(&rows))));
                run.sixteen(&format!("planes_fisheye_rs {lens}"), dev.project_eye(&planes(calib).fisheye_out(true).rs_rows(Some(&rows))));
            }
            run.sixteen("planes_equirect_rs no rows", dev.project_eye(&planes(kb()).rs_rows(Some(&[]))));
            run.sixteen("planes_fisheye_rs no rows", dev.project_eye(&planes(kb()).fisheye_out(true).rs_rows(Some(&[]))));
        }

        let text: String = run.lines.iter().map(|(n, h)| format!("{h:016x}  {n}\n")).collect();
        match std::fs::read_to_string(&path) {
            Ok(kept) => {
                let differ: Vec<&str> = text.lines().zip(kept.lines())
                    .filter(|(a, b)| a != b).map(|(a, _)| a).collect();
                assert!(differ.is_empty() && text.lines().count() == kept.lines().count(),
                    "{} of {} fingerprints differ from {path}:\n{}",
                    differ.len(), text.lines().count(), differ.join("\n"));
            }
            Err(_) => {
                std::fs::write(&path, &text).expect("write fingerprints");
                println!("wrote {} fingerprints to {path}", run.lines.len());
            }
        }
    }
}
