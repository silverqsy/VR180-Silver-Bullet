//! The linear-light tone stage ([`crate::tone_linear`]) on the GPU: the
//! frame's measurement, its Shadows / Highlights base and the stage itself,
//! for a picture that arrives as code values. The counterpart of
//! [`crate::raw_gpu`]'s development of a raw frame, with which it shares
//! the reading of a measurement and the bilateral grid's two passes.

use std::sync::{Arc, Mutex};

use crate::gpu::{CdlParams, ColorStackPlan, Device};
use crate::raw_gpu::{
    bgl_storage, bgl_storage_buffer, bgl_texture, bgl_uniform, compute_pipeline, same_stats, stats_from_cells, stats_within,
    GridParams, KEY_CELLS, KEY_TAPS,
};
use crate::raw_tone::{
    FrameStats, Measure, BASE_FALLBACK, BASE_FLOOR, BASE_LO, BASE_RANGE, LOCAL_LO, LOCAL_PER_STOP, OUT_LO,
};
use crate::tone_linear::{flat_gains, LightTables, LinearTone, Shown, ToneConstants, ToneLight, LUMA};

const TONE_MEASURE_WGSL: &str = include_str!("shaders/tone_measure.wgsl");
const TONE_REDUCE_WGSL: &str = include_str!("shaders/tone_reduce.wgsl");
const TONE_LINEAR_WGSL: &str = include_str!("shaders/tone_linear.wgsl");

/// How much wider than between two frames of one size the live preview's
/// measurements may differ between two sizes of one picture before the
/// frame on screen is drawn again for it.
const ACROSS_SIZES: f32 = 6.0;

/// What the stage keeps between frames.
#[derive(Default)]
struct Memory {
    /// What the live preview's last measurement found (it must not wait on
    /// the GPU: see [`Measure::Latest`]), and the size of the eyes it was
    /// made on.
    latest: Option<(FrameStats, [u32; 3])>,
    in_flight: bool,
    /// Called when a measurement lands that differs from `latest`.
    notify: Option<Arc<dyn Fn() + Send + Sync>>,
    /// The tables of the last frame's light and rendering: every frame of
    /// a clip has the same ones while its grade stands.
    tables: Option<(ToneLight, Shown, LightTables)>,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory").field("latest", &self.latest).field("in_flight", &self.in_flight).finish_non_exhaustive()
    }
}

/// What a device holds for the stage: its pipelines, made when a frame
/// first needs them, and what it keeps between frames.
#[derive(Debug, Default)]
pub(crate) struct ToneState {
    pipelines: std::sync::OnceLock<TonePipelines>,
    memory: Arc<Mutex<Memory>>,
}

/// The stage's pipelines.
#[derive(Debug)]
struct TonePipelines {
    measure: wgpu::ComputePipeline,
    measure_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    /// The stage writing `Rgba8Unorm` and `Rgba16Unorm`.
    apply: wgpu::ComputePipeline,
    apply_layout: wgpu::BindGroupLayout,
    apply_16: wgpu::ComputePipeline,
    apply_16_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeasureParams {
    cell: [f32; 4],
    luma: [f32; 4],
    table: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    luma: [f32; 4],
    light: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ApplyParams {
    light: [f32; 4],
    luma: [f32; 4],
    gains: [f32; 4],
    grid: [f32; 4],
    stage: [f32; 4],
    floors: [f32; 4],
    keyed: [f32; 4],
}

impl TonePipelines {
    fn create(device: &wgpu::Device) -> Self {
        use wgpu::TextureViewDimension::{D2, D3};
        let measure_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tone_measure_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::Rgba32Float, D2),
                bgl_uniform(2, std::mem::size_of::<MeasureParams>()),
                bgl_storage_buffer(3),
            ],
        });
        let measure = compute_pipeline(device, "tone_measure", TONE_MEASURE_WGSL, "main", &measure_layout);
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tone_reduce_bgl"),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, wgpu::TextureFormat::R32Float, D2),
                bgl_uniform(2, std::mem::size_of::<ReduceParams>()),
                bgl_storage_buffer(3),
            ],
        });
        let reduce = compute_pipeline(device, "tone_reduce", TONE_REDUCE_WGSL, "main", &reduce_layout);
        let apply_layout_for = |label: &str, format: wgpu::TextureFormat| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                bgl_texture(0, D2, false),
                bgl_storage(1, format, D2),
                bgl_uniform(2, std::mem::size_of::<ApplyParams>()),
                bgl_storage_buffer(3),
                bgl_storage_buffer(4),
                bgl_texture(5, D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                bgl_storage_buffer(7),
                bgl_storage_buffer(8),
                bgl_storage_buffer(9),
                bgl_storage_buffer(10),
            ],
        });
        let apply_layout = apply_layout_for("tone_linear_bgl", wgpu::TextureFormat::Rgba8Unorm);
        let apply = compute_pipeline(device, "tone_linear", TONE_LINEAR_WGSL, "main", &apply_layout);
        let apply_16_layout = apply_layout_for("tone_linear_16_bgl", wgpu::TextureFormat::Rgba16Unorm);
        let apply_16 = compute_pipeline(
            device, "tone_linear_16", &TONE_LINEAR_WGSL.replace("rgba8unorm, write", "rgba16unorm, write"), "main", &apply_16_layout);
        Self { measure, measure_layout, reduce, reduce_layout, apply, apply_layout, apply_16, apply_16_layout }
    }
}

/// The CDL of one channel: `shaders/cdl.wgsl`, in the same order.
fn cdl_channel(x: f32, p: &CdlParams) -> f32 {
    let hermite = |t: f32| t * t * (3.0 - 2.0 * t);
    let mut x = x + p.lift * (1.0 - x);
    x *= p.gain;
    x += p.shadow * hermite((1.0 - 2.0 * x).clamp(0.0, 1.0)) * 0.6;
    x += p.highlight * hermite((2.0 * x - 1.0).clamp(0.0, 1.0)) * 0.6;
    x.clamp(0.0, 1.0).powf(1.0 / p.gamma.max(0.0001))
}

/// A grey through a 3D LUT at `intensity`, as `shaders/lut3d.wgsl` takes
/// it (the cube's eight lattice values around it); the mean of the three
/// channels that come out.
fn lut_grey(lut: &vr180_core::color::Cube3DLut, intensity: f32, v: f32) -> f32 {
    let n = lut.size as usize;
    if n < 2 {
        return v;
    }
    let v = v.clamp(0.0, 1.0);
    let pos = v * (n - 1) as f32;
    let cell = (pos.floor() as usize).min(n - 2);
    let t = pos - cell as f32;
    // The lattice value at (r, g, b), red varying fastest.
    let at = |r: usize, g: usize, b: usize| -> f32 {
        let i = ((b * n + g) * n + r) * 3;
        (lut.data[i] + lut.data[i + 1] + lut.data[i + 2]) / 3.0
    };
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let (i, j) = (cell, cell + 1);
    let graded = mix(
        mix(mix(at(i, i, i), at(j, i, i), t), mix(at(i, j, i), at(j, j, i), t), t),
        mix(mix(at(i, i, j), at(j, i, j), t), mix(at(i, j, j), at(j, j, j), t), t),
        t,
    );
    mix(v, graded, intensity)
}

impl ColorStackPlan {
    /// The neutral curve of what renders the picture after the tone stage:
    /// the CDL, then the LUT. (White balance and saturation, which follow,
    /// leave a grey a grey.)
    pub fn shown(&self) -> Shown {
        let cdl = (!self.cdl.is_identity()).then_some(self.cdl);
        Shown::of(|code| {
            let after_cdl = cdl.as_ref().map_or(code, |p| cdl_channel(code, p));
            match &self.lut {
                Some((lut, intensity)) => lut_grey(lut, *intensity, after_cdl),
                None => after_cdl,
            }
        })
    }

    /// The tone stage of this plan that does something.
    pub fn active_tone(&self) -> Option<&LinearTone> {
        self.tone.as_ref().filter(|t| !t.is_identity())
    }
}

impl Device {
    fn tone_pipelines(&self) -> &TonePipelines {
        self.tone.pipelines.get_or_init(|| TonePipelines::create(&self.device))
    }

    /// Whom to tell when the live preview's measurement of a frame changes
    /// (see [`Measure::Latest`]): the preview worker, so a paused frame is
    /// drawn again with the levels of its own measurement. `None` clears it
    /// and forgets what was measured (another clip). The counterpart of
    /// [`Self::set_raw_tone_notifier`] for a clip that is not raw.
    pub fn set_linear_tone_notifier(&self, notify: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut m) = self.tone.memory.lock() {
            m.latest = None;
            m.notify = notify;
        }
    }

    /// The tables of a light and a rendering: the last frame's where they
    /// are the same.
    fn linear_tone_tables(&self, light: ToneLight, shown: &Shown) -> LightTables {
        let memory = &self.tone.memory;
        if let Ok(m) = memory.lock() {
            if let Some((l, s, tables)) = &m.tables {
                if *l == light && s == shown {
                    return tables.clone();
                }
            }
        }
        let tables = LightTables::of(light, shown);
        if let Ok(mut m) = memory.lock() {
            m.tables = Some((light, shown.clone(), tables.clone()));
        }
        tables
    }

    /// Queue the reduced copies of the given eyes (code values, light
    /// through `decode`) and return the buffer they are copied into: 16
    /// bytes per cell, eye after eye. The frame is all the eyes side by
    /// side, as for a raw frame.
    fn submit_linear_tone_measurement(&self, eyes: &[&wgpu::Texture], decode: &[f32]) -> (wgpu::Buffer, usize, usize) {
        let pipes = self.tone_pipelines();
        let (ew, eh) = (eyes[0].width(), eyes[0].height());
        let long = (ew * eyes.len() as u32).max(eh) as f32;
        let cell = long / KEY_CELLS as f32;
        let (cx, cy) = (((ew as f32 / cell).round() as u32).max(1), ((eh as f32 / cell).round() as u32).max(1));
        // A row of the copy must be a multiple of 256 bytes.
        let padded = cx.div_ceil(16) * 16;
        let eye_bytes = (padded * cy * 16) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tone_measure_staging"),
            size: eye_bytes * eyes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self.write_uniform("tone_measure_params", &MeasureParams {
            cell: [ew as f32 / cx as f32, eh as f32 / cy as f32, cell.ceil().clamp(1.0, KEY_TAPS as f32), 0.0],
            luma: [LUMA[0], LUMA[1], LUMA[2], BASE_FLOOR],
            table: [(decode.len() - 1) as f32, 0.0, 0.0, 0.0],
        });
        let table = self.curve_buffer("tone_measure_decode", decode);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tone_measure") });
        for (n, eye) in eyes.iter().enumerate() {
            let cells = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tone_measure_cells"),
                size: wgpu::Extent3d { width: padded, height: cy, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let (src_view, dst_view) = (eye.create_view(&Default::default()), cells.create_view(&Default::default()));
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tone_measure_bg"),
                layout: &pipes.measure_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: table.as_entire_binding() },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tone_measure"), timestamp_writes: None });
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

    /// What a frame's tone is derived from: its eyes as they enter the
    /// colour stack, in the light `decode` gives their codes.
    /// [`Measure::Wait`] measures this frame and waits; [`Measure::Latest`]
    /// returns what the last finished measurement found and queues one of
    /// this frame when none is on its way.
    pub fn linear_tone_stats(&self, eyes: &[&wgpu::Texture], decode: &[f32], measure: Measure) -> FrameStats {
        if eyes.is_empty() {
            return FrameStats::unknown();
        }
        let memory = self.tone.memory.clone();
        match measure {
            Measure::Wait => {
                let (staging, cx, cy) = self.submit_linear_tone_measurement(eyes, decode);
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
                        tracing::warn!("tone: the frame's levels could not be read back");
                        FrameStats::unknown()
                    }
                }
            }
            Measure::Latest => {
                let (latest, start) = match memory.lock() {
                    Ok(mut m) => {
                        let start = !m.in_flight;
                        m.in_flight = true;
                        (m.latest.as_ref().map(|(stats, _)| stats.clone()), start)
                    }
                    Err(_) => (None, false),
                };
                if start {
                    let (staging, cx, cy) = self.submit_linear_tone_measurement(eyes, decode);
                    let buffer = staging.clone();
                    let size = [eyes[0].width(), eyes[0].height(), eyes.len() as u32];
                    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        let stats = r.ok().and_then(|()| stats_from_cells(&Self::measured_cells(&buffer.slice(..).get_mapped_range(), cx, cy), cx, cy));
                        buffer.unmap();
                        let notify = memory.lock().ok().and_then(|mut m| {
                            m.in_flight = false;
                            let stats = stats?;
                            // The frame on screen and its enlargement are
                            // one picture at two sizes, measured at other
                            // places: what they find differs a little, and
                            // each would have the other drawn again without
                            // end. Across sizes only a real change counts.
                            let changed = !m.latest.as_ref().is_some_and(|(old, at)| {
                                if *at == size { same_stats(old, &stats) } else { stats_within(old, &stats, ACROSS_SIZES) }
                            });
                            m.latest = Some((stats, size));
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

    /// The constants of `plan`'s tone stage for one frame, `None` when the
    /// plan has none that does anything. `eyes`: the picture as it enters
    /// the stack, one frame (side by side when two): both eyes are then
    /// toned alike.
    pub fn linear_tone_constants(&self, eyes: &[&wgpu::Texture], plan: &ColorStackPlan) -> Option<ToneConstants> {
        let stage = plan.active_tone()?;
        let started = std::time::Instant::now();
        let shown = plan.shown();
        let tables = self.linear_tone_tables(stage.light, &shown);
        let tabled = started.elapsed();
        // A clip's frames are all toned by the clip's statistics; a frame
        // that has none is measured by itself.
        let stats = match &plan.clip_stats {
            _ if !stage.settings.needs_stats() => FrameStats::unknown(),
            Some(clip) => clip.tone(&tables.decode),
            None => self.linear_tone_stats(eyes, &tables.decode, stage.measure),
        };
        let measured = started.elapsed();
        let mut k = stage.constants_with(&tables, &stats, &shown, None);
        tracing::debug!(
            "tone: dark {:.5} max {:.4}, {} levels (tables {:.2?}, measured in {:.2?}, constants {:.2?})",
            stats.keys.dark, stats.keys.max, stats.levels.len(), tabled, measured - tabled, started.elapsed() - measured);
        if let Some(eye) = eyes.first() {
            k.long_side = (eye.width() * eyes.len() as u32).max(eye.height());
        }
        Some(k)
    }

    /// The plan's tone stage on a host picture (`w × h`, three bytes a
    /// pixel): eyes of `eye_w` side by side are toned as one frame, each
    /// with its own base. For the 8-bit path that grades on the host; a
    /// plan without an active stage returns the picture as it is.
    pub fn apply_linear_tone_rgb8(&self, rgb: &[u8], w: u32, h: u32, eye_w: u32, plan: &ColorStackPlan) -> crate::Result<Vec<u8>> {
        if plan.active_tone().is_none() || w == 0 || h == 0 {
            return Ok(rgb.to_vec());
        }
        let eyes = if eye_w > 0 && eye_w < w && w % eye_w == 0 { w / eye_w } else { 1 };
        let eye_w = w / eyes;
        let size = wgpu::Extent3d { width: eye_w, height: h, depth_or_array_layers: 1 };
        let texture = |label: &str, usage: wgpu::TextureUsages| self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        });
        let sources: Vec<wgpu::Texture> = (0..eyes).map(|eye| {
            let mut rgba = Vec::with_capacity((eye_w * h * 4) as usize);
            for row in rgb.chunks_exact((w * 3) as usize) {
                for px in row[(eye * eye_w * 3) as usize..((eye + 1) * eye_w * 3) as usize].chunks_exact(3) {
                    rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            let t = texture("tone_host_src", wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &rgba,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(eye_w * 4), rows_per_image: Some(h) },
                size);
            t
        }).collect();
        let Some(k) = self.linear_tone_constants(&sources.iter().collect::<Vec<_>>(), plan) else {
            return Ok(rgb.to_vec());
        };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tone_host") });
        let toned: Vec<wgpu::Texture> = sources.iter().map(|src| {
            let dst = texture("tone_host_dst", wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
            self.record_linear_tone(&mut encoder, src, &dst, &k, false);
            dst
        }).collect();
        self.queue.submit(Some(encoder.finish()));
        let mut out = vec![0u8; rgb.len()];
        for (eye, t) in toned.iter().enumerate() {
            let eye_rgb = self.read_texture_rgb8(t, eye_w, h)?;
            let (from, len) = (eye * eye_w as usize * 3, eye_w as usize * 3);
            for (row, eye_row) in out.chunks_exact_mut((w * 3) as usize).zip(eye_rgb.chunks_exact(len)) {
                row[from..from + len].copy_from_slice(eye_row);
            }
        }
        Ok(out)
    }

    /// Record the bilateral grid of one eye for a frame's Shadows /
    /// Highlights: the reduced copy (light through `decode`, exposed), then
    /// the raw stage's own gather and smoothing. Returns the smoothed grid
    /// and its cell in pixels of the eye.
    fn record_linear_tone_base(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, k: &ToneConstants, decode: &wgpu::Buffer) -> (wgpu::Texture, f32) {
        let (pipes, raw) = (self.tone_pipelines(), self.raw_pipelines());
        let long = if k.long_side > 0 { k.long_side } else { src.width().max(src.height()) };
        let (n, cell) = crate::raw_local::base_layout(long);
        let (rw, rh) = (src.width().div_ceil(n), src.height().div_ceil(n));
        let (gx, gy, gz) = crate::raw_local::grid_dims(rw, rh, cell);
        let reduced = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone_reduced"),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let grid = |label: &str| self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: gx, height: gy, depth_or_array_layers: gz },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let (gathered, smoothed) = (grid("tone_grid_gathered"), grid("tone_grid"));
        let reduce_params = self.write_uniform("tone_reduce_params", &ReduceParams {
            luma: [LUMA[0], LUMA[1], LUMA[2], n as f32],
            light: [(k.decode.len() - 1) as f32, k.gain, BASE_FLOOR, 0.0],
        });
        let grid_params = self.write_uniform("tone_grid_params", &GridParams { cell: [cell, BASE_LO, 1.0 / BASE_RANGE, 0.0] });
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (src_v, red_v, gat_v, smo_v) = (view(src), view(&reduced), view(&gathered), view(&smoothed));
        fn tex<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let reduce_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_reduce_bg"),
            layout: &pipes.reduce_layout,
            entries: &[
                tex(0, &src_v), tex(1, &red_v),
                wgpu::BindGroupEntry { binding: 2, resource: reduce_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: decode.as_entire_binding() },
            ],
        });
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_grid_build_bg"),
            layout: &raw.grid_build_layout,
            entries: &[tex(0, &red_v), tex(1, &gat_v), wgpu::BindGroupEntry { binding: 2, resource: grid_params.as_entire_binding() }],
        });
        let blur_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_grid_blur_bg"),
            layout: &raw.grid_blur_layout,
            entries: &[tex(0, &gat_v), tex(1, &smo_v)],
        });
        for (label, pipeline, bind, (x, y)) in [
            ("tone_reduce", &pipes.reduce, &reduce_bg, (rw, rh)),
            ("tone_grid_build", &raw.grid_build, &build_bg, (gx, gy)),
            ("tone_grid_blur", &raw.grid_blur, &blur_bg, (gx, gy)),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), 1);
        }
        (smoothed, cell * n as f32)
    }

    /// Record the tone stage: `src` (an eye, code values) → `dst` (the
    /// same size; `Rgba16Unorm` when `wide`, else `Rgba8Unorm`).
    pub(crate) fn record_linear_tone(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, k: &ToneConstants, wide: bool) {
        let decode = self.curve_buffer("tone_decode", &k.decode);
        // The frame's base, or one empty cell: with gains of zero that is a
        // gain of exactly 1.
        let (grid, cell_px, gains) = match &k.local {
            Some(gains) => {
                let (grid, cell_px) = self.record_linear_tone_base(encoder, src, k, &decode);
                (grid, cell_px, self.curve_buffer("tone_gains", gains))
            }
            None => {
                let empty = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("tone_grid_none"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                (empty, 1.0, self.curve_buffer("tone_gains", &flat_gains()))
            }
        };
        let out_lo = OUT_LO as f32;
        // The three stages' tables are of one length.
        let stage_last = (k.whites.change.len() - 1) as f32;
        let gains_last = (crate::raw_tone::LOCAL_SAMPLES - 1) as f32;
        let params = self.write_uniform("tone_linear_params", &ApplyParams {
            light: [(k.decode.len() - 1) as f32, k.gain, BASE_FLOOR, BASE_FALLBACK],
            luma: [LUMA[0], LUMA[1], LUMA[2], cell_px],
            gains: [LOCAL_LO, LOCAL_PER_STOP, gains_last, 0.0],
            grid: [BASE_LO, 1.0 / BASE_RANGE, 0.0, 0.0],
            stage: [out_lo, stage_last / -out_lo, stage_last, 2f32.powf(out_lo)],
            floors: [k.whites.floor, k.blacks.floor, k.contrast.floor, (k.encode.len() - 1) as f32],
            keyed: [if k.whites.keyed { 1.0 } else { 0.0 }, if k.blacks.keyed { 1.0 } else { 0.0 }, k.below, 0.0],
        });
        let (whites, blacks, contrast, encode) = (
            self.curve_buffer("tone_whites", &k.whites.change),
            self.curve_buffer("tone_blacks", &k.blacks.change),
            self.curve_buffer("tone_contrast", &k.contrast.change),
            self.curve_buffer("tone_encode", &k.encode),
        );
        let (pipes, raw) = (self.tone_pipelines(), self.raw_pipelines());
        let (pipeline, layout) = if wide { (&pipes.apply_16, &pipes.apply_16_layout) } else { (&pipes.apply, &pipes.apply_layout) };
        let (src_view, dst_view, grid_view) = (src.create_view(&Default::default()), dst.create_view(&Default::default()), grid.create_view(&Default::default()));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone_linear_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: decode.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gains.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&grid_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&raw.grid_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: whites.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: blacks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: contrast.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: encode.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tone_linear"), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bind), &[]);
        pass.dispatch_workgroups(dst.width().div_ceil(8), dst.height().div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tone_linear::ToneSettings;
    use vr180_core::color::Cube3DLut;
    use vr180_core::LogCurve;

    const ILOG: LogCurve = LogCurve::INSTA360_ILOG;

    fn texture16(dev: &Device, w: u32, h: u32, px: &[u16]) -> wgpu::Texture {
        let t = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone_test_src"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(px),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        t
    }

    fn read16(dev: &Device, t: &wgpu::Texture, w: u32, h: u32) -> Vec<u16> {
        dev.read_texture_rgba64(t, w, h).expect("readback").chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
    }

    /// A LUT like a log clip's, the same curve on every channel: black up
    /// to a tenth of the codes, a steep middle, a long shoulder.
    fn shoulder(code: f32) -> f32 {
        let t = ((code - 0.09) / 0.91).clamp(0.0, 1.0);
        1.0 - 0.9 * (1.0 - t).powf(3.2) - 0.1 * (1.0 - t)
    }

    fn lut() -> Cube3DLut {
        let n = 33usize;
        let at = |i: usize| shoulder(i as f32 / (n - 1) as f32);
        let mut data = Vec::with_capacity(n * n * n * 3);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    data.extend_from_slice(&[at(x), at(y), at(z)]);
                }
            }
        }
        Cube3DLut { size: n as u32, data, title: None, domain_min: [0.0; 3], domain_max: [1.0; 3] }
    }

    fn stage(settings: ToneSettings, light: ToneLight) -> LinearTone {
        LinearTone { settings, light, pre_stops: 0.0, measure: Measure::Wait }
    }

    /// Pixels like a frame's: regions of different brightness with texture
    /// inside, coloured, and a few specks.
    fn scene(w: u32, h: u32) -> Vec<u16> {
        let mut s = 7u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        (0..w * h).flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let region = match (x / (w / 4), y / (h / 2)) { (0, _) => 9000.0, (1, 0) => 21000.0, (1, _) => 14000.0, (2, _) => 34000.0, _ => 52000.0 };
            let texture = 0.85 + 0.3 * ((next() % 1000) as f32 / 1000.0);
            let speck = if next() % 211 == 0 { 1.6 } else { 1.0 };
            let v = (region * texture * speck).min(65535.0);
            [(v * 0.92) as u16, v as u16, (v * 0.8) as u16, u16::MAX]
        }).collect()
    }

    /// The neutral curve a plan gives is what its CDL and LUT show a grey as.
    #[test]
    fn the_curve_of_a_plan_is_what_its_stages_show() {
        let dev = Device::new().expect("device");
        let (w, h) = (256u32, 64u32);
        let px: Vec<u16> = (0..w * h).flat_map(|i| { let v = (i * 4 + 1).min(65535) as u16; [v, v, v, u16::MAX] }).collect();
        let src = texture16(&dev, w, h, &px);
        for plan in [
            ColorStackPlan { lut: Some((lut(), 1.0)), ..Default::default() },
            ColorStackPlan { lut: Some((lut(), 0.6)), cdl: CdlParams { lift: 0.03, gamma: 1.15, gain: 1.1, shadow: 0.2, highlight: -0.3 }, ..Default::default() },
            ColorStackPlan { cdl: CdlParams { lift: -0.02, gamma: 0.9, gain: 0.95, shadow: -0.1, highlight: 0.15 }, ..Default::default() },
        ] {
            let shown = plan.shown();
            let out = dev.apply_color_stack_per_eye_16(&src, w, h, &plan).expect("stack").expect("a stage ran");
            let got = read16(&dev, &out, w, h);
            let mut worst = 0f64;
            for (g, c) in got.chunks_exact(4).zip(px.chunks_exact(4)) {
                worst = worst.max((g[1] as f64 / 65535.0 - shown.at(c[1] as f64 / 65535.0)).abs());
            }
            assert!(worst < 6e-5, "the curve is {worst} off what the stages show");
        }
        // No stage: the codes themselves.
        assert_eq!(ColorStackPlan::default().shown(), Shown::identity());
    }

    /// The stage on the GPU against its CPU twin, pixel by pixel.
    #[test]
    fn gpu_tone_matches_the_cpu_one() {
        let dev = Device::new().expect("device");
        let (w, h) = (128u32, 96u32);
        let mut s = 99u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        // Every level, coloured and grey, some at the ends of the range.
        let px: Vec<u16> = (0..w * h).flat_map(|i| {
            let v = match i % 9 { 0 => 65535, 1 => next() % 3000, _ => 5000 + next() % 58000 } as f32;
            let tint = if i % 4 == 0 { [1.0, 1.0, 1.0] } else { [0.6 + 0.4 * (next() % 100) as f32 / 100.0, 1.0, 0.5 + 0.5 * (next() % 100) as f32 / 100.0] };
            [(v * tint[0]) as u16, (v * tint[1]) as u16, (v * tint[2]) as u16, 40000 + (i % 100) as u16]
        }).collect();
        let src = texture16(&dev, w, h, &px);
        let none = ToneSettings::default();
        for (light, lut_on) in [(ToneLight::Scene(ILOG), true), (ToneLight::Shown, true), (ToneLight::Shown, false)] {
            for settings in [
                ToneSettings { exposure: 1.2, ..none },
                ToneSettings { exposure: -1.5, ..none },
                ToneSettings { whites: -70.0, ..none },
                ToneSettings { whites: 60.0, ..none },
                ToneSettings { blacks: 50.0, ..none },
                ToneSettings { blacks: -60.0, exposure: 0.5, ..none },
                ToneSettings { contrast: 70.0, ..none },
                ToneSettings { contrast: -50.0, whites: 30.0, blacks: -20.0, exposure: -0.4, ..none },
            ] {
                let plan = ColorStackPlan {
                    tone: Some(stage(settings, light)),
                    lut: lut_on.then(|| (lut(), 1.0)),
                    ..Default::default()
                };
                let k = dev.linear_tone_constants(&[&src], &plan).expect("an active stage");
                assert!(k.local.is_none());
                // The stage alone: the plan's LUT is what it looks through, not run here.
                let dst = dev.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("tone_test_dst"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let mut encoder = dev.device.create_command_encoder(&Default::default());
                dev.record_linear_tone(&mut encoder, &src, &dst, &k, true);
                dev.queue.submit(Some(encoder.finish()));
                let got = read16(&dev, &dst, w, h);
                let (mut worst, mut sum, mut moved) = (0f32, 0f64, 0usize);
                for (g, c) in got.chunks_exact(4).zip(px.chunks_exact(4)) {
                    let codes = [c[0], c[1], c[2]].map(|v| v as f32 / 65535.0);
                    let want = k.apply(codes, k.base_luminance(codes).log2());
                    for ch in 0..3 {
                        let d = (g[ch] as f32 / 65535.0 - want[ch]).abs();
                        worst = worst.max(d);
                        sum += d as f64;
                    }
                    moved += ((g[1] as f32 / 65535.0 - codes[1]).abs() > 0.004) as usize;
                    assert_eq!(g[3], c[3], "alpha is carried");
                }
                let mean = sum / (3 * w * h) as f64;
                assert!(mean < 2e-5 && worst < 6e-4, "{light:?} lut {lut_on} {settings:?}: mean {mean}, worst {worst}");
                assert!(moved > (w * h / 10) as usize, "{light:?} {settings:?}: only {moved} pixels moved");
            }
        }
    }

    /// On an I-Log clip the stage's exposure is the log exposure stage's:
    /// the same codes come out of either, the ones under the curve's black
    /// included (the recovery of a clip shot bright moves into this stage
    /// once a tone control is in use).
    #[test]
    fn exposure_in_the_scenes_light_is_the_log_exposure_stage() {
        let dev = Device::new().expect("device");
        let (w, h) = (256u32, 256u32);
        // Every 16-bit code once per channel: red up, green down, blue in steps.
        let px: Vec<u16> = (0..w * h).flat_map(|i| [i as u16, (65535 - i) as u16, (i.wrapping_mul(40503) & 0xffff) as u16, u16::MAX]).collect();
        let src = texture16(&dev, w, h, &px);
        for stops in [-3.0f32, -2.0, -1.0, 0.7] {
            let by_tone = ColorStackPlan {
                tone: Some(LinearTone { pre_stops: stops, ..stage(ToneSettings::default(), ToneLight::Scene(ILOG)) }),
                ..Default::default()
            };
            let by_log = ColorStackPlan {
                log_exposure: Some(crate::gpu::LogExposure { curve: ILOG, stops }),
                ..Default::default()
            };
            let run = |plan: &ColorStackPlan| read16(&dev, &dev.apply_color_stack_per_eye_16(&src, w, h, plan).expect("stack").expect("a stage ran"), w, h);
            let (a, b) = (run(&by_tone), run(&by_log));
            let (mut worst, mut sum, mut n) = (0i32, 0i64, 0usize);
            for ((x, y), c) in a.chunks_exact(4).zip(b.chunks_exact(4)).zip(px.chunks_exact(4)) {
                // A colour with a channel pushed past the top of the range
                // is another matter: this stage keeps its channels in order
                // between the largest and the smallest, the other cuts each
                // one off by itself.
                let top = c[..3].iter().cloned().max().unwrap_or(0) as f64 / 65535.0;
                if ILOG.expose(top, stops as f64) >= 0.999 {
                    continue;
                }
                for ch in 0..3 {
                    let d = (x[ch] as i32 - y[ch] as i32).abs();
                    worst = worst.max(d);
                    sum += d as i64;
                }
                n += 3;
            }
            // In 16-bit codes: a 10-bit step is 64 of them.
            let mean = sum as f64 / n as f64;
            assert!(n > 3000 && worst <= 8 && mean < 1.0, "{stops} stops: worst {worst}, mean {mean} of a 16-bit code over {n} values");
        }
    }

    /// Shadows / Highlights on the GPU against the CPU twin: the base read
    /// from the bilateral grid, the gain of that base.
    #[test]
    fn gpu_tone_base_matches_the_cpu_one() {
        use crate::raw_local::{base_layout, reduce_log, BaseGrid};
        let dev = Device::new().expect("device");
        let (w, h) = (192u32, 128u32);
        let px = scene(w, h);
        let src = texture16(&dev, w, h, &px);
        let none = ToneSettings::default();
        for (light, settings) in [
            (ToneLight::Scene(ILOG), ToneSettings { shadows: 100.0, ..none }),
            (ToneLight::Scene(ILOG), ToneSettings { highlights: -100.0, exposure: 0.7, ..none }),
            (ToneLight::Shown, ToneSettings { shadows: -40.0, highlights: 60.0, ..none }),
            (ToneLight::Shown, ToneSettings { shadows: 70.0, highlights: -80.0, blacks: 30.0, whites: -40.0, contrast: 25.0, ..none }),
        ] {
            let plan = ColorStackPlan { tone: Some(stage(settings, light)), lut: Some((lut(), 1.0)), ..Default::default() };
            let mut k = dev.linear_tone_constants(&[&src], &plan).expect("an active stage");
            // A long side that gives more than one pixel per texel of the reduced copy.
            k.long_side = 2048;
            assert!(k.local.as_ref().expect("local stage").iter().any(|g| g.abs() > 0.2), "{settings:?}: no gain");
            let (n, cell) = base_layout(2048);
            let lum: Vec<f32> = px.chunks_exact(4).map(|c| k.base_luminance([c[0], c[1], c[2]].map(|v| v as f32 / 65535.0))).collect();
            let (red, rw, rh) = reduce_log(&lum, w as usize, h as usize, n as usize);
            let grid = BaseGrid::build(&red, rw, rh, cell);
            let dst = dev.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tone_test_dst"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let mut encoder = dev.device.create_command_encoder(&Default::default());
            dev.record_linear_tone(&mut encoder, &src, &dst, &k, true);
            dev.queue.submit(Some(encoder.finish()));
            let got = read16(&dev, &dst, w, h);
            let (mut worst, mut sum, mut moved) = (0f32, 0f64, 0usize);
            for (i, (g, c)) in got.chunks_exact(4).zip(px.chunks_exact(4)).enumerate() {
                let (x, y) = ((i as u32 % w) as f32, (i as u32 / w) as f32);
                let codes = [c[0], c[1], c[2]].map(|v| v as f32 / 65535.0);
                let want = k.apply(codes, grid.base(x, y, n as f32, lum[i].log2()));
                for ch in 0..3 {
                    let d = (g[ch] as f32 / 65535.0 - want[ch]).abs();
                    worst = worst.max(d);
                    sum += d as f64;
                }
                moved += ((g[1] as f32 / 65535.0 - codes[1]).abs() > 0.004) as usize;
            }
            let mean = sum / (3 * w * h) as f64;
            // The grid is 16-bit float on the GPU: a few thousandths at most.
            assert!(mean < 6e-4 && worst < 8e-3, "{light:?} {settings:?}: mean {mean}, worst {worst}");
            assert!(moved > (w * h / 10) as usize, "{light:?} {settings:?}: only {moved} pixels moved");
        }
    }

    /// The frame is measured in the stage's light, both eyes as one frame,
    /// and both are toned with what was found.
    #[test]
    fn both_eyes_of_a_frame_are_toned_alike() {
        let dev = Device::new().expect("device");
        let (w, h) = (252u32, 252u32);
        let flat = |code: u16| -> Vec<u16> { (0..w * h).flat_map(|_| [code, code, code, u16::MAX]).collect() };
        // The left eye dark with one patch of the right eye's grey, the
        // right eye that grey all over.
        let (dark, grey) = ((ILOG.encode(0.02) * 65535.0) as u16, (ILOG.encode(0.4) * 65535.0) as u16);
        let mut left_px = flat(dark);
        for y in 100..140 {
            for x in 100..140 {
                let i = ((y * w + x) * 4) as usize;
                left_px[i..i + 3].copy_from_slice(&[grey; 3]);
            }
        }
        let (left, right) = (texture16(&dev, w, h, &left_px), texture16(&dev, w, h, &flat(grey)));
        let tone = stage(ToneSettings { contrast: 60.0, blacks: -40.0, whites: 30.0, ..Default::default() }, ToneLight::Scene(ILOG));
        let plan = ColorStackPlan { tone: Some(tone), ..Default::default() };
        // The measurement: the light of the codes, relative to the light of code 1.
        let shown = plan.shown();
        let tables = LightTables::of(tone.light, &shown);
        let stats = dev.linear_tone_stats(&[&left, &right], &tables.decode, Measure::Wait);
        let white = ILOG.decode(1.0);
        assert!((stats.keys.max as f64 - 0.4 / white).abs() < 2e-3 / white, "max {} of {}", stats.keys.max, 0.4 / white);
        assert!((stats.keys.dark as f64 - 0.02 / white).abs() < 2e-3 / white, "dark {}", stats.keys.dark);
        assert_eq!(stats.levels.len(), 2 * 21 * 21);
        // Toned as a pair, the grey of the left eye's patch is the right eye's grey.
        let (l, r) = dev.apply_color_stack_pair_16(&left, &right, w, h, &plan, None).expect("stack");
        let (l, r) = (read16(&dev, &l.expect("left"), w, h), read16(&dev, &r.expect("right"), w, h));
        let at = |v: &[u16], x: u32, y: u32| v[((y * w + x) * 4 + 1) as usize];
        assert_eq!(at(&l, 120, 120), at(&r, 120, 120));
        assert_eq!(at(&r, 10, 10), at(&r, 120, 120));
        assert_ne!(at(&r, 120, 120), grey, "the stage did nothing");
        // Each eye on its own is another frame: the right one, all grey,
        // has no spread for Contrast to work on and comes out differently.
        let alone = read16(&dev, &dev.apply_color_stack_per_eye_16(&right, w, h, &plan).expect("stack").expect("a stage ran"), w, h);
        assert_ne!(at(&alone, 120, 120), at(&r, 120, 120));
    }

    /// A stage with nothing moved is not run, and the 8-bit stack tones as
    /// the 16-bit one does.
    #[test]
    fn nothing_moved_is_not_run_and_eight_bits_follow_sixteen() {
        let dev = Device::new().expect("device");
        let (w, h) = (192u32, 128u32);
        let px = scene(w, h);
        let src = texture16(&dev, w, h, &px);
        let idle = ColorStackPlan { tone: Some(stage(ToneSettings::default(), ToneLight::Shown)), ..Default::default() };
        assert!(!idle.any_active() && idle.active_tone().is_none());
        assert!(dev.apply_color_stack_per_eye_16(&src, w, h, &idle).expect("stack").is_none());
        assert!(dev.linear_tone_constants(&[&src], &idle).is_none());

        let plan = ColorStackPlan {
            tone: Some(stage(ToneSettings { exposure: 0.5, highlights: -60.0, shadows: 40.0, contrast: 20.0, ..Default::default() }, ToneLight::Scene(ILOG))),
            lut: Some((lut(), 1.0)),
            ..Default::default()
        };
        let wide = read16(&dev, &dev.apply_color_stack_per_eye_16(&src, w, h, &plan).expect("stack").expect("a stage ran"), w, h);
        // The same picture at 8 bits, through the stack that writes a BGRA frame.
        let px8: Vec<u8> = px.iter().map(|v| (*v as f32 / 257.0).round() as u8).collect();
        let src8 = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone_test_src8"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &src8, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &px8,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
        let frame = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone_test_frame"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        dev.apply_color_stack_to_eye_bgra(&src8, &frame, w, h, &plan).expect("stack");
        // The frame's stored bytes are B, G, R.
        let narrow = dev.read_texture_rgb8(&frame, w, h).expect("readback");
        let (mut worst, mut sum) = (0f32, 0f64);
        for (n, wd) in narrow.chunks_exact(3).zip(wide.chunks_exact(4)) {
            for ch in 0..3 {
                let d = (n[2 - ch] as f32 / 255.0 - wd[ch] as f32 / 65535.0).abs();
                worst = worst.max(d);
                sum += d as f64;
            }
        }
        // The source's own rounding to 8 bits, through the steepest part of
        // the LUT, and the output's.
        let mean = sum / (3 * w * h) as f64;
        assert!(mean < 4e-3 && worst < 0.03, "8 bits against 16: mean {mean}, worst {worst}");
    }

    /// The host picture of the 8-bit fallback: two eyes side by side are
    /// one frame, toned as the GPU stack tones them.
    #[test]
    fn a_host_picture_is_toned_as_the_stack_tones_it() {
        let dev = Device::new().expect("device");
        let (w, h) = (96u32, 64u32);
        let (l16, r16) = (scene(w, h), scene(w, h).iter().map(|v| (*v as u32 * 3 / 5) as u16).collect::<Vec<_>>());
        let bytes = |px: &[u16]| -> Vec<u8> { px.iter().map(|v| (*v as f32 / 257.0).round() as u8).collect() };
        let (l8, r8) = (bytes(&l16), bytes(&r16));
        // The frame as the host holds it: three bytes a pixel, the eyes side by side.
        let mut sbs = Vec::with_capacity((2 * w * h * 3) as usize);
        for y in 0..h as usize {
            for eye in [&l8, &r8] {
                for px in eye[y * w as usize * 4..(y + 1) * w as usize * 4].chunks_exact(4) {
                    sbs.extend_from_slice(&px[..3]);
                }
            }
        }
        let plan = ColorStackPlan {
            tone: Some(stage(ToneSettings { exposure: -0.3, shadows: 50.0, highlights: -40.0, contrast: 30.0, ..Default::default() }, ToneLight::Shown)),
            ..Default::default()
        };
        let toned = dev.apply_linear_tone_rgb8(&sbs, 2 * w, h, w, &plan).expect("host stage");
        assert_eq!(toned.len(), sbs.len());
        // The same through the stack that writes a frame from two eye textures.
        let eye8 = |px: &[u8]| -> wgpu::Texture {
            let t = dev.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tone_test_eye8"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            dev.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                px,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            t
        };
        let frame = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone_test_frame"),
            size: wgpu::Extent3d { width: 2 * w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        dev.apply_color_stack_to_sbs_bgra(&eye8(&l8), &eye8(&r8), &frame, w, h, &plan).expect("stack");
        let stack = dev.read_texture_rgb8(&frame, 2 * w, h).expect("readback");
        // The frame's stored bytes are B, G, R.
        let differing = toned.chunks_exact(3).zip(stack.chunks_exact(3)).filter(|(a, b)| a[0] != b[2] || a[1] != b[1] || a[2] != b[0]).count();
        assert_eq!(differing, 0, "{differing} pixels differ from the stack's");
        assert!(toned.iter().zip(&sbs).filter(|(a, b)| a != b).count() > sbs.len() / 4, "the stage did nothing");
        // Nothing moved: the picture as it came.
        let idle = ColorStackPlan::default();
        assert_eq!(dev.apply_linear_tone_rgb8(&sbs, 2 * w, h, w, &idle).expect("host stage"), sbs);
    }

    /// The live preview does not wait: it gets what the last measurement
    /// found, and is told when one lands.
    #[test]
    fn the_preview_is_told_when_its_measurement_lands() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dev = Device::new().expect("device");
        // 126 cells of 4 px along the frame.
        let (w, h) = (504u32, 252u32);
        let small = scene(w, h);
        let src = texture16(&dev, w, h, &small);
        let told = Arc::new(AtomicUsize::new(0));
        let counter = told.clone();
        dev.set_linear_tone_notifier(Some(Arc::new(move || { counter.fetch_add(1, Ordering::SeqCst); })));
        let tables = LightTables::of(ToneLight::Shown, &Shown::identity());
        let measure = |t: &wgpu::Texture| {
            let stats = dev.linear_tone_stats(&[t], &tables.decode, Measure::Latest);
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            stats
        };
        let first = measure(&src);
        assert!(first.levels.is_empty(), "nothing can be known yet");
        assert_eq!(told.load(Ordering::SeqCst), 1);
        let second = measure(&src);
        assert!(!second.levels.is_empty());
        // The same frame again says the same: nobody is told twice.
        assert_eq!(told.load(Ordering::SeqCst), 1);
        assert!(same_stats(&second, &dev.linear_tone_stats(&[&src], &tables.decode, Measure::Wait)));
        // The same picture drawn larger is measured at other places, and
        // drawn by other arithmetic: the little that differs there (here a
        // hundredth of the codes, a thirtieth of a stop of light) is no
        // reason to draw anything again.
        let (bw, bh) = (2 * w, 2 * h);
        let enlarged = |gain: f32| -> wgpu::Texture {
            let px: Vec<u16> = (0..bw * bh).flat_map(|i| {
                let at = (((i / bw) / 2 * w + (i % bw) / 2) * 4) as usize;
                small[at..at + 4].iter().map(|v| (*v as f32 * gain).min(65535.0) as u16).collect::<Vec<_>>()
            }).collect();
            texture16(&dev, bw, bh, &px)
        };
        measure(&enlarged(1.01));
        assert_eq!(told.load(Ordering::SeqCst), 1, "an enlargement of the frame had it drawn again");
        // Between two frames of one size that much is a change.
        measure(&enlarged(1.01 * 1.01));
        assert_eq!(told.load(Ordering::SeqCst), 2);
        // And another picture is one at any size.
        measure(&texture16(&dev, w, h, &small.iter().map(|v| v / 3).collect::<Vec<_>>()));
        assert_eq!(told.load(Ordering::SeqCst), 3);
        // Another clip: what was measured is forgotten.
        dev.set_linear_tone_notifier(None);
        assert!(dev.linear_tone_stats(&[&src], &tables.decode, Measure::Latest).levels.is_empty());
    }
}
