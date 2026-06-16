//! `vr180-gui` — native egui + wgpu GUI for VR180 Silver Bullet 2.0.
//!
//! Replaces the Tauri/WebView shell with a single-binary native app.
//! The headline win over the Tauri version: **the decoded + projected
//! `wgpu::Texture` from our pipeline is handed directly to the egui
//! renderer for display**. No CPU readback, no JPEG encode, no IPC,
//! no MP4 proxy file. Roughly the same shape as Gyroflow's preview
//! path (MDK → QRhi compute → QQuickItem), but pure-Rust and built on
//! wgpu instead of Qt RHI.

// Release Windows builds run as a GUI-subsystem app — no console window pops up
// on launch. Logs go to a file in the per-OS data dir instead (see `main`).
// Debug builds keep the console so `cargo run` still prints to the terminal.
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod audio_player;
mod decoder;

use tracing_subscriber::EnvFilter;

/// Decode the bundled `.ico` into an egui window icon (its largest frame) for
/// the live window — title bar + taskbar while the app is running. The .exe's
/// *file* icon (Explorer, the installer's shortcut) is a separate build-time
/// Win32 resource (see `build.rs`); winit doesn't reliably adopt that for the
/// running window, so we set it explicitly here too. Returns `None` (the app
/// still launches, just without an icon) rather than panicking if the bundled
/// asset can't be decoded.
#[cfg(not(target_os = "macos"))]
fn load_app_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../../../assets/icon.ico");
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some(egui::IconData { rgba: rgba.into_raw(), width, height })
}

fn main() -> anyhow::Result<()> {
    // Logs go to `<data dir>/vr180-gui.log` (truncated each launch) so the
    // release GUI-subsystem build — which has no console — still leaves a
    // troubleshooting trail. Falls back to stderr (debug `cargo run`) if the
    // file can't be opened. Set RUST_LOG to override the filter.
    let log_filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,vr180_gui=debug,wgpu_core=warn,wgpu_hal=warn,naga=warn")
    });
    let log_path = decoder::Settings::config_path()
        .and_then(|p| p.parent().map(|d| d.join("vr180-gui.log")));
    let log_file = log_path.as_ref().and_then(|p| {
        if let Some(dir) = p.parent() { std::fs::create_dir_all(dir).ok()?; }
        std::fs::File::create(p).ok()
    });
    match log_file {
        Some(f) => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(log_filter())
                .with_target(false)
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(f))
                .try_init();
            if let Some(p) = log_path.as_ref() {
                tracing::info!("log file: {}", p.display());
            }
        }
        None => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(log_filter())
                .with_target(false)
                .try_init();
        }
    }

    #[cfg_attr(target_os = "macos", allow(unused_mut))]
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("VR180 Silver Bullet 2.0")
        .with_inner_size([1480.0, 920.0])
        .with_min_inner_size([1100.0, 720.0])
        .with_drag_and_drop(true);
    // macOS takes the Dock/window icon from the `.app` bundle's `icon.icns`
    // (Info.plist → CFBundleIconFile). Setting a runtime icon there overrides
    // the bundle icon with the raw image — the wrong icon while running — so
    // only set it on Windows/Linux, which have no bundle to carry it.
    #[cfg(not(target_os = "macos"))]
    if let Some(icon) = load_app_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let native_options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        // 10-bit P010 IOSurfaces need TEXTURE_FORMAT_16BIT_NORM. We
        // request it on the wgpu device eframe creates for us so the
        // pipeline (which borrows that device) can use R16Unorm /
        // Rg16Unorm without re-creating the device.
        // egui-wgpu 0.34 moved device-creation knobs into `wgpu_setup`.
        // `WgpuSetupCreateNew` has no `Default`, so start from the default
        // config and override only the device descriptor — we request
        // TEXTURE_FORMAT_16BIT_NORM (needed for the R16/Rg16 16-bit color
        // stack). Backend left default (Vulkan on Windows); forcing DX12
        // caused DXGI_ERROR_DEVICE_REMOVED with D3D11VA decode + wgpu-D3D12.
        wgpu_options: {
            let mut cfg = egui_wgpu::WgpuConfiguration::default();
            if let egui_wgpu::WgpuSetup::CreateNew(create) = &mut cfg.wgpu_setup {
                create.device_descriptor = std::sync::Arc::new(|adapter: &wgpu::Adapter| {
                    // TEXTURE_FORMAT_16BIT_NORM: R16/Rg16 color stack.
                    // TEXTURE_FORMAT_P010 / _NV12: zero-copy import of
                    // NVDEC-decoded frames into wgpu (Windows D3D11→Vulkan).
                    // `& adapter.features()` so we only request what's
                    // supported (graceful on GPUs that lack them).
                    let want = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                        | wgpu::Features::TEXTURE_FORMAT_P010
                        | wgpu::Features::TEXTURE_FORMAT_NV12;
                    let have = adapter.features() & want;
                    wgpu::DeviceDescriptor {
                        label: Some("vr180-gui device"),
                        required_features: have,
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                        trace: wgpu::Trace::Off,
                        experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    }
                });
            }
            cfg
        },
        ..Default::default()
    };

    eframe::run_native(
        "VR180 Silver Bullet 2.0",
        native_options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    ).map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    Ok(())
}
