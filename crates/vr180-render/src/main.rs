//! `vr180-render` — CLI binary.
//!
//! Phase 0.1 — `--help` placeholder.
//! Phase 0.2 — `probe-gyro <file.360>` reads the file via ffmpeg-next,
//!             extracts the GPMF data stream, parses CORI/IORI, prints
//!             counts and Euler-angle ranges. Validates the headless
//!             gyro pipeline end-to-end against a real file.
//!
//! Phase 0.9 — `export --config <json>` will be the wedge: the existing
//! Python GUI on `main` shells out to this binary for the heavy work,
//! no UI port needed.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "vr180-render", version, about)]
struct Cli {
    /// Verbosity (repeat: -v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum HwAccel {
    /// Platform default: VideoToolbox on macOS, software elsewhere
    /// (silently falls back to software if hwaccel init fails).
    Auto,
    /// Force software decode.
    Sw,
    /// Force VideoToolbox (macOS only). Errors out if VT unavailable.
    Vt,
}

impl From<HwAccel> for vr180_pipeline::decode::HwDecode {
    fn from(h: HwAccel) -> Self {
        match h {
            HwAccel::Auto => Self::Auto,
            HwAccel::Sw   => Self::Software,
            HwAccel::Vt   => Self::VideoToolbox,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Probe gyro data from a .360 file: CORI/IORI count + Euler ranges.
    ProbeGyro {
        /// Path to the .360 file (or first segment of a chain).
        path: PathBuf,
        /// Also run the VQF (Versatile Quaternion Filter) 9D fusion
        /// pipeline (GYRO + GRAV×9.81 + MNOR → quat + bias estimate).
        /// Use this to validate the no-firmware-RS path on a clip where
        /// CORI is bias-drifting (xyz_max < ~0.001 at t=0).
        #[arg(long)]
        vqf: bool,
    },

    /// Probe the EAC layout of a .360 file: stream dimensions, derived
    /// tile width, cross size. With `--out <file.png>` also writes the
    /// assembled Lens-A + Lens-B EAC crosses (3936×7872 RGB PNG by
    /// default on a standard Max — smaller on Max 2 variants).
    ProbeEac {
        path: PathBuf,
        /// Output PNG path for the raw assembled cross pair (no projection).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Output PNG path for the GPU-projected half-equirect SBS image
        /// (left eye = Lens A, right eye = Lens B). Triggers the wgpu
        /// compute kernel.
        #[arg(long)]
        equirect: Option<PathBuf>,
        /// Half-equirect output width per eye (default 2048). Final PNG
        /// is `2 * eye_w × eye_w` — a square half-equirect per eye,
        /// stitched side-by-side.
        #[arg(long, default_value_t = 2048)]
        eye_w: u32,
        /// Hardware-accelerated decode: auto (platform default), sw
        /// (force software), or vt (force VideoToolbox, macOS only).
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
        /// Optional .cube 3D LUT to apply after the equirect projection.
        /// On macOS the bundled GoPro GP-Log LUT at
        /// `assets/Recommended Lut GPLOG.cube` is a sensible default;
        /// pass `--lut bundled` to use it.
        #[arg(long)]
        lut: Option<String>,
        /// LUT blend factor [0..1]. 0 = original, 1 = full LUT.
        #[arg(long, default_value_t = 1.0)]
        lut_intensity: f32,
    },

    /// Phase 0.6 decode-throughput benchmark. Decode the first N frames
    /// of one video stream end-to-end (including `av_hwframe_transfer_data`
    /// when VT is in use) and report fps. Compares software vs VideoToolbox
    /// hwaccel under steady-state conditions where the single-frame
    /// cold-start cost has been amortized.
    BenchDecode {
        path: PathBuf,
        /// Number of frames to decode.
        #[arg(short, long, default_value_t = 100)]
        frames: u32,
        /// Hardware decode path: auto, sw, or vt.
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
    },

    /// Phase 0.6.5: decode one VideoToolbox frame, extract its
    /// CVPixelBuffer's IOSurface, wrap the Y and UV planes as
    /// `wgpu::Texture`s via the IOSurface↔Metal↔wgpu-hal bridge, and
    /// read back a few Y-plane bytes to confirm the chain works.
    /// **macOS only.**
    #[cfg(target_os = "macos")]
    ProbeIosurface {
        path: PathBuf,
    },

    /// Phase 0.8 export: decode → assemble → equirect → (LUT) → H.265 mp4.
    /// Single-codec libx265 software encode (Phase 0.8 baseline).
    Export {
        /// Input `.360` file.
        input: PathBuf,
        /// Output `.mov` / `.mp4` file.
        output: PathBuf,
        /// Half-equirect width per eye. Final SBS frame is `2 * eye_w × eye_w`.
        #[arg(long, default_value_t = 2048)]
        eye_w: u32,
        /// Number of frames to export (0 = all).
        #[arg(short = 'n', long, default_value_t = 0)]
        frames: u32,
        /// Output FPS. Defaults to the source clip's FPS.
        #[arg(long)]
        fps: Option<f32>,
        /// libx265 target bitrate in kbps.
        #[arg(long, default_value_t = 12_000)]
        bitrate: u32,
        /// Optional .cube 3D LUT. `bundled` = the GP-Log LUT from
        /// `assets/`.
        #[arg(long)]
        lut: Option<String>,
        /// LUT blend factor [0..1].
        #[arg(long, default_value_t = 1.0)]
        lut_intensity: f32,
        /// Hardware-accelerated decode: auto / sw / vt.
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
        /// Skip host-memory hop entirely: VideoToolbox decoder's IOSurface
        /// goes straight to wgpu via the Metal HAL escape, EAC assembly
        /// happens on the GPU (`nv12_to_eac_cross.wgsl`). **macOS only.**
        /// Forces `--hw-accel vt`.
        #[arg(long, default_value_t = false)]
        zero_copy: bool,
    },
}

fn init_tracing(verbosity: u8) {
    use tracing_subscriber::EnvFilter;
    let default = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match cli.command {
        None => {
            println!(
                "vr180-render {} — Phase 0.2 (GPMF + CORI/IORI)",
                env!("CARGO_PKG_VERSION")
            );
            println!("Try: vr180-render --help");
            Ok(())
        }
        Some(Cmd::ProbeGyro { path, vqf }) => probe_gyro(&path, vqf),
        Some(Cmd::ProbeEac { path, out, equirect, eye_w, hw_accel, lut, lut_intensity }) =>
            probe_eac(&path, out.as_deref(), equirect.as_deref(), eye_w,
                      hw_accel.into(), lut.as_deref(), lut_intensity),
        Some(Cmd::BenchDecode { path, frames, hw_accel }) =>
            bench_decode(&path, frames, hw_accel.into()),
        #[cfg(target_os = "macos")]
        Some(Cmd::ProbeIosurface { path }) => probe_iosurface(&path),
        Some(Cmd::Export { input, output, eye_w, frames, fps, bitrate, lut, lut_intensity, hw_accel, zero_copy }) =>
            export(&input, &output, eye_w, frames, fps, bitrate,
                   lut.as_deref(), lut_intensity, hw_accel.into(), zero_copy),
    }
}

fn export(
    input: &std::path::Path,
    output: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: Option<f32>,
    bitrate_kbps: u32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
    hw: vr180_pipeline::decode::HwDecode,
    zero_copy: bool,
) -> anyhow::Result<()> {
    if zero_copy && !cfg!(target_os = "macos") {
        anyhow::bail!(
            "--zero-copy is macOS-only (requires IOSurface↔Metal interop). \
             Phase 0.6.8 will add the Windows CUDA↔Vulkan equivalent."
        );
    }
    if zero_copy {
        #[cfg(target_os = "macos")]
        return export_zero_copy(input, output, eye_w, n_frames, fps, bitrate_kbps,
                                lut_spec, lut_intensity);
        #[cfg(not(target_os = "macos"))]
        unreachable!()
    } else {
        export_cpu_assemble(input, output, eye_w, n_frames, fps, bitrate_kbps,
                            lut_spec, lut_intensity, hw)
    }
}

/// Phase 0.6 / 0.7 / 0.8 path: hwaccel decode → host-memory NV12 →
/// swscale RGB → CPU EAC assembly → GPU projection + LUT → libx265.
fn export_cpu_assemble(
    input: &std::path::Path,
    output: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: Option<f32>,
    bitrate_kbps: u32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
    hw: vr180_pipeline::decode::HwDecode,
) -> anyhow::Result<()> {
    use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::{iter_stream_pairs, probe_video};
    use vr180_pipeline::encode::H265Encoder;
    use vr180_pipeline::gpu::Device;

    let probe = probe_video(input)?;
    let out_w = eye_w * 2;
    let out_h = eye_w;
    let fps = fps.unwrap_or(probe.fps);

    println!("Export (CPU-assemble path): {} → {}", input.display(), output.display());
    println!("  source : {} × {}  @ {:.3} fps  ({:.1}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("  output : {} × {}  @ {:.3} fps  H.265 {} kbps",
        out_w, out_h, fps, bitrate_kbps);

    let lut = load_lut(lut_spec, lut_intensity)?;

    let device = Device::new()?;
    let mut decoder = iter_stream_pairs(input, hw, n_frames)?;
    let dims = decoder.dims();
    let cw = dims.cross_w() as usize;
    println!("  decode : {} (probed {}×{}, EAC tile_w={})",
        decoder.decode_path(), dims.stream_w, dims.stream_h, dims.tile_w());

    let mut encoder = H265Encoder::create(output, out_w, out_h, fps, bitrate_kbps)?;
    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];

    let t_start = std::time::Instant::now();
    let mut frame_idx: u32 = 0;
    let mut last_print = std::time::Instant::now();
    while let Some(pair) = decoder.next_pair()? {
        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        let mut left  = device.project_cross_to_equirect(&cross_a, dims.cross_w(), eye_w, out_h)?;
        let mut right = device.project_cross_to_equirect(&cross_b, dims.cross_w(), eye_w, out_h)?;
        if let Some((lut, intensity)) = &lut {
            left  = device.apply_lut3d(&left,  eye_w, out_h, lut, *intensity)?;
            right = device.apply_lut3d(&right, eye_w, out_h, lut, *intensity)?;
        }
        encoder.encode_frame(&stitch_sbs(&left, &right, eye_w, out_h))?;
        frame_idx += 1;
        progress_tick(frame_idx, t_start, &mut last_print);
    }
    encoder.finish()?;
    finish_export(output, frame_idx, t_start)?;
    Ok(())
}

/// Phase 0.6.6 path: IOSurface → wgpu zero-copy → GPU EAC assembly +
/// equirect projection + (LUT) → readback → libx265. macOS only.
#[cfg(target_os = "macos")]
fn export_zero_copy(
    input: &std::path::Path,
    output: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: Option<f32>,
    bitrate_kbps: u32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
) -> anyhow::Result<()> {
    use vr180_pipeline::decode::{ZeroCopyStreamPairIter, probe_video};
    use vr180_pipeline::encode::H265Encoder;
    use vr180_pipeline::gpu::{Device, Lens};

    let probe = probe_video(input)?;
    let out_w = eye_w * 2;
    let out_h = eye_w;
    let fps = fps.unwrap_or(probe.fps);

    println!("Export (zero-copy IOSurface path): {} → {}", input.display(), output.display());
    println!("  source : {} × {}  @ {:.3} fps  ({:.1}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("  output : {} × {}  @ {:.3} fps  H.265 {} kbps",
        out_w, out_h, fps, bitrate_kbps);
    println!("  pipeline: VT → IOSurface → wgpu (nv12_to_eac_cross → eac_to_equirect → lut3d) → libx265");

    let lut = load_lut(lut_spec, lut_intensity)?;
    let device = Device::new()?;
    let mut decoder = ZeroCopyStreamPairIter::new(input, n_frames)?;
    let dims = decoder.dims();
    println!("  decode : VideoToolbox (probed {}×{}, EAC tile_w={})",
        dims.stream_w, dims.stream_h, dims.tile_w());

    let mut encoder = H265Encoder::create(output, out_w, out_h, fps, bitrate_kbps)?;
    let t_start = std::time::Instant::now();
    let mut frame_idx: u32 = 0;
    let mut last_print = std::time::Instant::now();
    while let Some(pair) = decoder.next_pair(&device.device)? {
        // GPU EAC assembly: NV12 plane textures → RGB cross texture.
        let cross_a = device.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::A, dims,
        )?;
        let cross_b = device.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::B, dims,
        )?;

        let mut left  = device.project_cross_texture_to_equirect(&cross_a, eye_w, out_h)?;
        let mut right = device.project_cross_texture_to_equirect(&cross_b, eye_w, out_h)?;
        if let Some((lut, intensity)) = &lut {
            left  = device.apply_lut3d(&left,  eye_w, out_h, lut, *intensity)?;
            right = device.apply_lut3d(&right, eye_w, out_h, lut, *intensity)?;
        }
        encoder.encode_frame(&stitch_sbs(&left, &right, eye_w, out_h))?;
        // Drop the zero-copy pair AFTER the kernel dispatch is done —
        // releases the IOSurface retains so the VT decoder can recycle
        // the underlying CVPixelBuffer for the next frame.
        drop(pair);
        frame_idx += 1;
        progress_tick(frame_idx, t_start, &mut last_print);
    }
    encoder.finish()?;
    finish_export(output, frame_idx, t_start)?;
    Ok(())
}

fn load_lut(
    lut_spec: Option<&str>, intensity: f32,
) -> anyhow::Result<Option<(vr180_core::Cube3DLut, f32)>> {
    let path = match lut_spec {
        Some("bundled") => resolve_bundled_lut()
            .ok_or_else(|| anyhow::anyhow!("bundled LUT not found"))?,
        Some(p) => std::path::PathBuf::from(p),
        None => return Ok(None),
    };
    let lut = vr180_core::Cube3DLut::from_file(&path)?;
    println!("  LUT    : {}^3 @ intensity {:.2} ({})",
        lut.size, intensity, path.display());
    Ok(Some((lut, intensity)))
}

fn stitch_sbs(left: &[u8], right: &[u8], eye_w: u32, eye_h: u32) -> Vec<u8> {
    let out_w = eye_w * 2;
    let row_l = (eye_w * 3) as usize;
    let row_sbs = (out_w * 3) as usize;
    let mut sbs = vec![0u8; (out_w * eye_h * 3) as usize];
    for y in 0..eye_h as usize {
        sbs[y * row_sbs..y * row_sbs + row_l]
            .copy_from_slice(&left[y * row_l..y * row_l + row_l]);
        sbs[y * row_sbs + row_l..y * row_sbs + row_l * 2]
            .copy_from_slice(&right[y * row_l..y * row_l + row_l]);
    }
    sbs
}

fn progress_tick(frame_idx: u32, t_start: std::time::Instant, last: &mut std::time::Instant) {
    if last.elapsed().as_secs_f32() > 1.0 {
        let avg_fps = frame_idx as f32 / t_start.elapsed().as_secs_f32();
        print!("\r  frame {frame_idx}  @ {avg_fps:.2} fps  ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        *last = std::time::Instant::now();
    }
}

fn finish_export(output: &std::path::Path, frame_idx: u32, t_start: std::time::Instant)
    -> anyhow::Result<()>
{
    let total = t_start.elapsed();
    let avg_fps = frame_idx as f32 / total.as_secs_f32();
    println!("\nDone: {frame_idx} frames in {total:.2?} ({avg_fps:.2} fps)");
    let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!("Output size: {:.1} MB", size as f64 / 1_048_576.0);
    Ok(())
}

/// Locate `assets/Recommended Lut GPLOG.cube` relative to either the
/// workspace root (dev runs) or the running binary (release builds).
fn resolve_bundled_lut() -> Option<std::path::PathBuf> {
    // Dev: walk up from CWD looking for the assets dir.
    let mut p = std::env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = p.join("assets/Recommended Lut GPLOG.cube");
        if candidate.is_file() {
            return Some(candidate);
        }
        p = p.parent()?.to_path_buf();
    }
    // Release: next to the exe.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("Recommended Lut GPLOG.cube");
            if p.is_file() { return Some(p); }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn probe_iosurface(path: &std::path::Path) -> anyhow::Result<()> {
    use vr180_pipeline::decode::decode_first_vt_frame;
    use vr180_pipeline::interop_macos::{
        extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
        IOSurfaceNv12Descriptor,
    };
    use vr180_pipeline::gpu::Device;
    use std::time::Instant;

    println!("=== Phase 0.6.5: IOSurface ↔ Metal ↔ wgpu zero-copy bridge ===");
    println!("file: {}", path.display());
    println!();

    // 1. Decode one VT frame (no av_hwframe_transfer_data).
    let t = Instant::now();
    let vt_frame = decode_first_vt_frame(path)?;
    println!("[1] decode_first_vt_frame:           {:?}  (format={:?}, {}×{})",
        t.elapsed(), vt_frame.format(), vt_frame.width(), vt_frame.height());

    // 2. Pull the IOSurface backing the CVPixelBuffer.
    let t = Instant::now();
    let surface = extract_iosurface_from_vt_frame(&vt_frame)?;
    println!("[2] CVPixelBufferGetIOSurface:       {:?}  (planes={}, y={}×{}, uv={}×{})",
        t.elapsed(),
        surface.plane_count(),
        surface.plane_width(0), surface.plane_height(0),
        surface.plane_width(1), surface.plane_height(1));

    // 3. Cache geometry (NV12).
    let desc = IOSurfaceNv12Descriptor::new(surface)?;
    println!("[3] IOSurfaceNv12Descriptor:         OK  (y_bpr={}, uv_bpr={})",
        desc.y_bytes_per_row, desc.uv_bytes_per_row);

    // 4. Wrap Y plane as a wgpu::Texture (Metal R8Unorm view).
    let device = Device::new()?;
    let (w, h) = (desc.width, desc.height);
    let t = Instant::now();
    let y_tex = wgpu_texture_from_iosurface_plane(
        &device.device, desc.surface, 0,
        metal::MTLPixelFormat::R8Unorm,
        wgpu::TextureFormat::R8Unorm,
        w, h,
        "iosurface_y",
    )?;
    println!("[4] wgpu_texture_from_iosurface(Y):  {:?}  ({}×{} R8Unorm)",
        t.elapsed(), y_tex.width, y_tex.height);

    // 5. Read back the first row of Y so we can prove the chain is live
    //    (not just a successful-but-blank texture handoff). 256-byte
    //    aligned for the staging buffer requirement.
    let row_bytes_unpadded = w;
    let row_bytes_padded = (row_bytes_unpadded + 255) & !255;
    let buf_size = row_bytes_padded as u64;
    let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("y_readback_enc"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &y_tex.texture, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes_padded),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d { width: w, height: 1, depth_or_array_layers: 1 },
    );
    let t = Instant::now();
    device.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    let mapped = slice.get_mapped_range();
    let first_row = &mapped[..w as usize];
    let min = *first_row.iter().min().unwrap_or(&0);
    let max = *first_row.iter().max().unwrap_or(&0);
    let avg: f32 = first_row.iter().map(|&v| v as f32).sum::<f32>() / w as f32;
    println!("[5] read back Y row 0:               {:?}  (min={min} max={max} avg={avg:.1})",
        t.elapsed());
    drop(mapped);
    staging.unmap();

    println!();
    println!("✓ zero-copy chain works end-to-end.");
    println!("  VT decoder → CVPixelBuffer → IOSurface → MTLTexture → wgpu::Texture");
    println!("  No av_hwframe_transfer_data on this path. The bytes the GPU sees are");
    println!("  the same bytes the VideoToolbox decoder wrote — direct, unified memory.");
    Ok(())
}

fn bench_decode(
    path: &std::path::Path,
    n_frames: u32,
    hw: vr180_pipeline::decode::HwDecode,
) -> anyhow::Result<()> {
    let result = vr180_pipeline::decode::bench_decode_throughput(path, n_frames, hw)?;
    let ms_per_frame = result.total.as_secs_f32() * 1000.0 / result.frames.max(1) as f32;
    println!("file:        {}", path.display());
    println!("frames:      {} (requested {})", result.frames, n_frames);
    println!("decode path: {}", result.decode_path);
    println!("total:       {:.2?}", result.total);
    println!("per frame:   {ms_per_frame:.2} ms");
    println!("throughput:  {:.2} fps", result.fps());
    Ok(())
}

fn probe_eac(
    path: &std::path::Path,
    out: Option<&std::path::Path>,
    equirect: Option<&std::path::Path>,
    eye_w: u32,
    hw: vr180_pipeline::decode::HwDecode,
    lut_spec: Option<&str>,
    lut_intensity: f32,
) -> anyhow::Result<()> {
    use vr180_core::eac::{Dims, assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::{extract_first_stream_pair_with, probe_video};

    let probe = probe_video(path)?;
    let dims = Dims::new(probe.width, probe.height);
    println!("file:        {}", path.display());
    println!("stream:      {} × {}  ({:.3} fps, {:.2}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("EAC tile_w:  {} px  (formula: (w-1920)/4)", dims.tile_w());
    println!("EAC cross:   {0} × {0}  (per lens)", dims.cross_w());
    if !dims.is_valid() {
        anyhow::bail!("stream width {} is not a valid EAC layout (need (w-1920) % 4 == 0)",
            probe.width);
    }

    if out.is_none() && equirect.is_none() {
        println!("(pass --out <png> for the raw cross pair, or --equirect <png> for the GPU projection)");
        return Ok(());
    }

    // Decode + assemble once; reuse for both output paths.
    let t0 = std::time::Instant::now();
    let pair = extract_first_stream_pair_with(path, hw)?;
    let decode_t = t0.elapsed();
    let dims = pair.dims;
    let cw = dims.cross_w() as usize;

    let t1 = std::time::Instant::now();
    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];
    assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
    assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);
    let assemble_t = t1.elapsed();

    println!();
    println!("decoded streams:  {decode_t:.2?}  ({} bytes / stream)  [path: {}]",
        pair.s0.len(), pair.decode_path);
    println!("assembled crosses:{assemble_t:.2?}  (2 × {}×{})", cw, cw);

    if let Some(out) = out {
        let combined_w = cw;
        let combined_h = cw * 2;
        let mut combined = Vec::with_capacity(combined_w * combined_h * 3);
        combined.extend_from_slice(&cross_a);
        combined.extend_from_slice(&cross_b);
        let t = std::time::Instant::now();
        let img = image::RgbImage::from_raw(combined_w as u32, combined_h as u32, combined)
            .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw size mismatch"))?;
        img.save(out)?;
        println!("wrote cross PNG:  {:?}  → {}  ({} × {})",
            t.elapsed(), out.display(), combined_w, combined_h);
    }

    if let Some(eq) = equirect {
        use vr180_pipeline::gpu::Device;
        let eye_h = eye_w; // square half-equirect per eye (±90° × ±90°)
        let t = std::time::Instant::now();
        let device = Device::new()?;
        let device_t = t.elapsed();

        // Resolve LUT path: `bundled` → assets path; anything else → as-is.
        let lut = if let Some(spec) = lut_spec {
            let path = if spec == "bundled" {
                resolve_bundled_lut().ok_or_else(|| anyhow::anyhow!(
                    "--lut bundled but assets/Recommended Lut GPLOG.cube not found"
                ))?
            } else {
                std::path::PathBuf::from(spec)
            };
            let t = std::time::Instant::now();
            let lut = vr180_core::Cube3DLut::from_file(&path)?;
            println!("loaded LUT:       {}^3 from {}  ({:?})",
                lut.size, path.display(), t.elapsed());
            Some(lut)
        } else { None };

        let t = std::time::Instant::now();
        let mut left_eye = device.project_cross_to_equirect(&cross_a, dims.cross_w(), eye_w, eye_h)?;
        let mut right_eye = device.project_cross_to_equirect(&cross_b, dims.cross_w(), eye_w, eye_h)?;
        let gpu_t = t.elapsed();

        if let Some(lut) = &lut {
            let t = std::time::Instant::now();
            left_eye = device.apply_lut3d(&left_eye, eye_w, eye_h, lut, lut_intensity)?;
            right_eye = device.apply_lut3d(&right_eye, eye_w, eye_h, lut, lut_intensity)?;
            println!("applied LUT (2x): {:?}", t.elapsed());
        }

        // Stitch L | R side-by-side.
        let sbs_w = eye_w * 2;
        let sbs_h = eye_h;
        let mut sbs = vec![0u8; (sbs_w * sbs_h * 3) as usize];
        let row_l = (eye_w * 3) as usize;
        let row_sbs = (sbs_w * 3) as usize;
        for y in 0..eye_h as usize {
            sbs[y * row_sbs.. y * row_sbs + row_l]
                .copy_from_slice(&left_eye[y * row_l..y * row_l + row_l]);
            sbs[y * row_sbs + row_l..y * row_sbs + row_l * 2]
                .copy_from_slice(&right_eye[y * row_l..y * row_l + row_l]);
        }

        let t = std::time::Instant::now();
        let img = image::RgbImage::from_raw(sbs_w, sbs_h, sbs)
            .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw size mismatch"))?;
        img.save(eq)?;
        let save_t = t.elapsed();

        println!("wgpu device:      {device_t:.2?}");
        println!("gpu project (2x): {gpu_t:.2?}  (per eye: ~{:.2?})", gpu_t / 2);
        println!("wrote SBS PNG:    {save_t:.2?}  → {}  ({} × {})",
            eq.display(), sbs_w, sbs_h);
    }

    Ok(())
}

fn probe_gyro(path: &std::path::Path, do_vqf: bool) -> anyhow::Result<()> {
    use vr180_core::gyro::{parse_cori, parse_iori, quat_to_euler_zyx};
    use vr180_pipeline::decode::{extract_gpmf_stream, probe_video};

    let t0 = std::time::Instant::now();
    let probe = probe_video(path)?;
    let gpmf = extract_gpmf_stream(path)?;
    let cori = parse_cori(&gpmf);
    let iori = parse_iori(&gpmf);
    let elapsed = t0.elapsed();

    println!("file:          {}", path.display());
    println!("video:         {}×{} @ {:.3} fps, {:.2}s",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("GPMF stream:   {} bytes", gpmf.len());
    println!("CORI samples:  {}", cori.len());
    println!("IORI samples:  {}", iori.len());

    if let Some(&q) = cori.first() {
        println!("CORI[0]:       w={:.6} x={:.6} y={:.6} z={:.6}", q.w, q.x, q.y, q.z);
    }
    if let Some(&q) = iori.first() {
        println!("IORI[0]:       w={:.6} x={:.6} y={:.6} z={:.6}", q.w, q.x, q.y, q.z);
    }

    if !cori.is_empty() {
        let (mut rmin, mut rmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut pmin, mut pmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &q in &cori {
            let (r, p, y) = quat_to_euler_zyx(q);
            rmin = rmin.min(r); rmax = rmax.max(r);
            pmin = pmin.min(p); pmax = pmax.max(p);
            ymin = ymin.min(y); ymax = ymax.max(y);
        }
        println!("CORI Euler ranges (deg):");
        println!("  roll : [{rmin:>8.3}, {rmax:>8.3}]");
        println!("  pitch: [{pmin:>8.3}, {pmax:>8.3}]");
        println!("  yaw  : [{ymin:>8.3}, {ymax:>8.3}]");
    }
    println!("elapsed:       {elapsed:.2?}");

    if do_vqf {
        probe_gyro_vqf(path)?;
    }
    Ok(())
}

fn probe_gyro_vqf(path: &std::path::Path) -> anyhow::Result<()> {
    use vr180_core::gyro::vqf;
    use vr180_pipeline::imu::{prepare_for_vqf, MagSource, AccSource};

    let t0 = std::time::Instant::now();
    let prep = prepare_for_vqf(path)?;
    let prep_elapsed = t0.elapsed();

    let acc_label = match prep.acc_source {
        AccSource::Grav => "GRAV×9.81",
        AccSource::Raw  => "raw ACCL",
        AccSource::None => "none",
    };
    let mag_label = match prep.mag_source {
        MagSource::Mnor => "MNOR",
        MagSource::None => "none",
    };
    let mode = if prep.mag_body.is_some() { "9D" } else { "6D" };

    let t1 = std::time::Instant::now();
    let run = vqf::run(
        &prep.gyro_body,
        &prep.acc_body,
        prep.mag_body.as_deref(),
        prep.gyr_ts,
    );
    let vqf_elapsed = t1.elapsed();

    let bias_deg = run.bias_deg_s();
    let qf = run.quats.first().copied().unwrap_or(vr180_core::gyro::Quat::IDENTITY);
    let ql = run.quats.last().copied().unwrap_or(vr180_core::gyro::Quat::IDENTITY);

    println!();
    println!("VQF {mode} ({acc_label}+{mag_label})");
    println!("  gyro samples:  {}  ({:.2} Hz)", prep.gyro_body.len(), 1.0 / prep.gyr_ts);
    println!("  acc input :    {}  → resampled to {}", prep.n_acc_input, prep.acc_body.len());
    if let Some(ref m) = prep.mag_body {
        println!("  mag input :    {}  → resampled to {}", prep.n_mag_input, m.len());
    }
    println!("  bias deg/s:    [{:.3}, {:.3}, {:.3}]  σ={:.4} rad/s",
        bias_deg[0], bias_deg[1], bias_deg[2], run.bias_sigma);
    println!("  quat[ 0]:      w={:.6} x={:.6} y={:.6} z={:.6}", qf.w, qf.x, qf.y, qf.z);
    println!("  quat[-1]:      w={:.6} x={:.6} y={:.6} z={:.6}", ql.w, ql.x, ql.y, ql.z);
    println!("  prep elapsed:  {prep_elapsed:.2?}");
    println!("  vqf  elapsed:  {vqf_elapsed:.2?}");
    Ok(())
}
