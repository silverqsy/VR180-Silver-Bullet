//! `vr180-app` — Tauri desktop GUI for VR180 Silver Bullet Neo.
//!
//! This crate hosts the Tauri commands. The `main()` binary
//! (`src/main.rs`) is a thin wrapper that builds a `tauri::Builder`
//! and invokes [`run`].
//!
//! Architecture sketch:
//!
//! ```text
//!  HTML/JS frontend  ──(tauri.invoke)──►  Rust commands (this crate)
//!  (ui/index.html)                             │
//!                                              ▼
//!                                  vr180-pipeline / vr180-core
//!                                  (probe, decode, render, encode)
//! ```
//!
//! Phase 1.0.0 — first cut: open file → probe → display info → export.
//! Phase 1.0.1+ — scrub preview, live playback, full param parity.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;

use tracing_subscriber::EnvFilter;

/// Entry point used by the binary. Spawned from `src/main.rs`.
pub fn run() {
    // Best-effort tracing init. RUST_LOG=info to see backend logs in the
    // terminal you launched `cargo tauri dev` from.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            commands::probe_clip,
            commands::extract_preview_frame,
            commands::detect_segments,
            commands::probe_geoc,
            commands::lookup_srot_ms,
            commands::version_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
