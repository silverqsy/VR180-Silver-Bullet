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
        /// Output PNG path. If supplied, decodes the first frame of
        /// each HEVC stream and writes the assembled cross pair.
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// (placeholder) Render the project described by a JSON config.
    Export {
        /// Path to the export JSON config (schema matches what the
        /// Python GUI writes).
        #[arg(long)]
        config: PathBuf,
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
        Some(Cmd::ProbeEac { path, out }) => probe_eac(&path, out.as_deref()),
        Some(Cmd::Export { config }) => {
            tracing::warn!(?config, "export: not implemented (Phase 0.9)");
            Ok(())
        }
    }
}

fn probe_eac(path: &std::path::Path, out: Option<&std::path::Path>) -> anyhow::Result<()> {
    use vr180_core::eac::{Dims, assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::{extract_first_stream_pair, probe_video};

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

    let Some(out) = out else {
        println!("(pass --out <file.png> to render the assembled cross pair)");
        return Ok(());
    };

    let t0 = std::time::Instant::now();
    let pair = extract_first_stream_pair(path)?;
    let decode_t = t0.elapsed();
    let dims = pair.dims;
    let cw = dims.cross_w() as usize;

    let t1 = std::time::Instant::now();
    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];
    assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
    assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);
    let assemble_t = t1.elapsed();

    // Compose the two crosses vertically into a single PNG so the user
    // sees both lenses in one file. Width = cw, Height = 2*cw.
    let combined_w = cw;
    let combined_h = cw * 2;
    let mut combined = Vec::with_capacity(combined_w * combined_h * 3);
    combined.extend_from_slice(&cross_a);
    combined.extend_from_slice(&cross_b);

    let t2 = std::time::Instant::now();
    let img = image::RgbImage::from_raw(combined_w as u32, combined_h as u32, combined)
        .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw size mismatch"))?;
    img.save(out)?;
    let save_t = t2.elapsed();

    println!();
    println!("decoded streams:  {decode_t:.2?}  ({} bytes / stream)", pair.s0.len());
    println!("assembled crosses:{assemble_t:.2?}  (2 × {}×{})", cw, cw);
    println!("wrote PNG:        {save_t:.2?}  → {}", out.display());
    println!("output dims:      {} × {} (Lens A on top, Lens B on bottom)",
        combined_w, combined_h);
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
