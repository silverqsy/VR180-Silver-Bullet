//! `vr180-gui` — native egui + wgpu GUI for VR180 Silver Bullet Neo.
//!
//! Replaces the Tauri/WebView shell with a single-binary native app.
//! The headline win over the Tauri version: **the decoded + projected
//! `wgpu::Texture` from our pipeline is handed directly to the egui
//! renderer for display**. No CPU readback, no JPEG encode, no IPC,
//! no MP4 proxy file. Roughly the same shape as Gyroflow's preview
//! path (MDK → QRhi compute → QQuickItem), but pure-Rust and built on
//! wgpu instead of Qt RHI.

mod app;
mod audio_player;
mod decoder;

use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,vr180_gui=debug,wgpu_core=warn,wgpu_hal=warn,naga=warn")))
        .with_target(false)
        .try_init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("VR180 Silver Bullet Neo")
            .with_inner_size([1480.0, 920.0])
            .with_min_inner_size([1100.0, 720.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        // 10-bit P010 IOSurfaces need TEXTURE_FORMAT_16BIT_NORM. We
        // request it on the wgpu device eframe creates for us so the
        // pipeline (which borrows that device) can use R16Unorm /
        // Rg16Unorm without re-creating the device.
        wgpu_options: egui_wgpu::WgpuConfiguration {
            device_descriptor: std::sync::Arc::new(|adapter| {
                let want = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
                let have = adapter.features() & want;
                wgpu::DeviceDescriptor {
                    label: Some("vr180-gui device"),
                    required_features: have,
                    required_limits: wgpu::Limits::default(),
                }
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "VR180 Silver Bullet Neo",
        native_options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    ).map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    Ok(())
}
