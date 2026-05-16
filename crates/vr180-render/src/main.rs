//! `vr180-render` — CLI binary.
//!
//! Phase 0.1: prints help. Real subcommands land in 0.2+:
//!   - `probe-gyro <file.360>` — CORI/IORI/GRAV/MNOR stats
//!   - `probe-eac  <file.360>` — stream dims + GEOC summary
//!   - `export    --config <json>` — full render
//!
//! The "config" entry point is the wedge that lets the existing
//! Python GUI on `main` shell out to this binary for the heavy
//! work, without us having to port the UI first.

use clap::{Parser, Subcommand};

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
    /// (placeholder) Probe gyro data from a .360 file.
    ProbeGyro {
        /// Path to the .360 file (or first segment of a chain).
        path: std::path::PathBuf,
    },

    /// (placeholder) Render the project described by a JSON config.
    Export {
        /// Path to the export JSON config (schema matches what the
        /// Python GUI writes).
        #[arg(long)]
        config: std::path::PathBuf,
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
            // No subcommand: print a friendly banner so `vr180-render`
            // with no args isn't silent.
            println!(
                "vr180-render {} — Phase 0.1 skeleton",
                env!("CARGO_PKG_VERSION")
            );
            println!("Try: vr180-render --help");
            Ok(())
        }
        Some(Cmd::ProbeGyro { path }) => {
            tracing::warn!(?path, "probe-gyro: not implemented (Phase 0.2)");
            Ok(())
        }
        Some(Cmd::Export { config }) => {
            tracing::warn!(?config, "export: not implemented (Phase 0.9)");
            Ok(())
        }
    }
}
