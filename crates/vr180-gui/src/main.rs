//! `vr180-gui` — native egui + wgpu GUI for VR180 Silver Bullet.
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
mod displays;
mod updater;

use tracing_subscriber::EnvFilter;

/// Decode the bundled `.ico` into an egui window icon (its largest frame) for
/// the live window — title bar + taskbar while the app is running. The .exe's
/// *file* icon (Explorer, the installer's shortcut) is a separate build-time
/// Win32 resource (see `build.rs`); winit doesn't reliably adopt that for the
/// running window, so we set it explicitly here too. Returns `None` (the app
/// still launches, just without an icon) rather than panicking if the bundled
/// asset can't be decoded.
/// The live window/Dock icon while the app runs, decoded from the bundled
/// PNG. egui applies its OWN default icon when none is supplied — which would
/// replace the launch (bundle) icon the moment the window opens — so we always
/// provide ours. The PNG matches the macOS `.app` `icon.icns`, so the Dock
/// tile stays the same from launch through GUI. (The Windows `.exe` *file*
/// icon is a separate build-time Win32 resource — see `build.rs`.) A single
/// PNG decodes unambiguously, unlike picking a frame out of the multi-size
/// `.ico`. Returns `None` (launch icon-less) on decode error rather than panic.
fn load_app_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../../../assets/icon.png");
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some(egui::IconData { rgba: rgba.into_raw(), width, height })
}

/// Windows: the app's GPU device with Vulkan Video decode
/// (`vr180_pipeline::vk_shared::create_ui_device`) — the same features and
/// texture limit the device descriptor below requests, plus FFmpeg's decode
/// queues on the same `VkDevice`, so the live preview decodes through Vulkan
/// Video on the device that presents it. `None` keeps eframe's own device
/// (the preview then decodes through D3D11, as before): no Vulkan adapter,
/// `WGPU_BACKEND` set, `VR180_D3D11_DECODE` set (the full fallback), no
/// MAILBOX present mode, or the device could not be created.
///
/// `Some` must present in MAILBOX mode. Under FIFO the UI thread waits in
/// `get_current_texture` for a free swapchain image, and wgpu holds the
/// device's fence lock through that wait, so a submission from any other
/// thread on the device waits for the display too. The decode thread submits
/// each frame's copy + resolve on this device, and FFmpeg's next decode waits
/// for that copy (NVIDIA decodes into the reference images), so a FIFO
/// display paced decoding: over Remote Desktop, where presenting is slow, the
/// 50 fps preview played at 0.7× speed. MAILBOX never waits for an image.
#[cfg(target_os = "windows")]
fn vulkan_decode_ui_setup() -> Option<egui_wgpu::WgpuSetupExisting> {
    if std::env::var_os("WGPU_BACKEND").is_some() || std::env::var_os("VR180_D3D11_DECODE").is_some() {
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // eframe's own choice: the high-performance adapter unless overridden.
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env()
            .unwrap_or(wgpu::PowerPreference::HighPerformance),
        compatible_surface: None,
        force_fallback_adapter: false,
    })).ok()?;
    if !vulkan_supports_mailbox(&instance, &adapter) {
        tracing::info!("Vulkan present mode MAILBOX unsupported — the preview decodes through D3D11");
        return None;
    }
    match vr180_pipeline::vk_shared::create_ui_device(&adapter) {
        Ok((device, queue)) => {
            tracing::info!("UI device: Vulkan Video decode, MAILBOX present");
            Some(egui_wgpu::WgpuSetupExisting { instance, adapter, device, queue })
        }
        Err(e) => {
            tracing::warn!("Vulkan-decode UI device unavailable ({e}) — the preview decodes through D3D11");
            None
        }
    }
}

/// Whether `adapter` can present a Vulkan surface in MAILBOX mode, asked of a
/// hidden throwaway window: eframe configures its surface with whatever mode
/// it is given, unchecked, and the modes a surface supports are the driver's,
/// not the window's.
#[cfg(target_os = "windows")]
fn vulkan_supports_mailbox(instance: &wgpu::Instance, adapter: &wgpu::Adapter) -> bool {
    use std::num::NonZeroIsize;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, WS_POPUP};
    let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
    unsafe {
        let hinstance = GetModuleHandleW(std::ptr::null());
        let hwnd = CreateWindowExW(
            0, class.as_ptr(), std::ptr::null(), WS_POPUP, 0, 0, 1, 1,
            std::ptr::null_mut(), std::ptr::null_mut(), hinstance, std::ptr::null(),
        );
        if hwnd.is_null() {
            return false;
        }
        // wgpu panics if the driver refuses the surface.
        let modes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (Some(h), Some(i)) = (NonZeroIsize::new(hwnd as isize), NonZeroIsize::new(hinstance as isize))
            else { return Vec::new() };
            let mut window = wgpu::rwh::Win32WindowHandle::new(h);
            window.hinstance = Some(i);
            match instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(wgpu::rwh::RawDisplayHandle::Windows(wgpu::rwh::WindowsDisplayHandle::new())),
                raw_window_handle: wgpu::rwh::RawWindowHandle::Win32(window),
            }) {
                Ok(surface) => surface.get_capabilities(adapter).present_modes,
                Err(_) => Vec::new(),
            }
        })).unwrap_or_default();
        DestroyWindow(hwnd);
        modes.contains(&wgpu::PresentMode::Mailbox)
    }
}

fn main() -> anyhow::Result<()> {
    // Logs go to `<data dir>/vr180-gui.log` (the previous launch's copy is
    // kept as `vr180-gui.prev.log`) so the
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
        // Keep the previous launch's log as `vr180-gui.prev.log`: a report
        // that arrives after a relaunch (or a crash + restart) would otherwise
        // have no trail left to read.
        let _ = std::fs::rename(p, p.with_file_name("vr180-gui.prev.log"));
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

    // Title carries no version — the toolbar's clickable version label is
    // the single source of truth (it also opens the updater popover).
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("VR180 Silver Bullet")
        .with_inner_size([1480.0, 920.0])
        .with_min_inner_size([1100.0, 720.0])
        .with_drag_and_drop(true);
    // Always set the live window icon (all platforms): egui otherwise applies
    // its own default icon when the window opens, which would replace the
    // launch/Dock icon mid-startup. Ours matches the macOS `.app` icon, so the
    // tile is stable from launch through GUI.
    if let Some(icon) = load_app_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    #[cfg(target_os = "windows")]
    let vulkan_ui = vulkan_decode_ui_setup();
    #[cfg(not(target_os = "windows"))]
    let vulkan_ui: Option<egui_wgpu::WgpuSetupExisting> = None;
    let mailbox = vulkan_ui.is_some();

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
        // stack). (Forcing DX12 caused DXGI_ERROR_DEVICE_REMOVED with
        // D3D11VA decode + wgpu-D3D12 — the DX12 fallback below is only for
        // boxes with no Vulkan ICD at all.)
        //
        // Windows: PREFER the Vulkan backend explicitly. Every GPU fast
        // path (D3D11→Vulkan zero-copy import, GPU-resident NVENC, the
        // 4:2:2 ProRes feed) gates on `is_vulkan_backend`; wgpu's default
        // PRIMARY selection usually lands on Vulkan but is not guaranteed,
        // and a silent DX12 pick would disable all of them (CPU-speed
        // exports with no visible reason). Probe synchronously and only
        // restrict the instance when a Vulkan adapter actually exists, so
        // a Vulkan-less machine still starts on DX12.
        wgpu_options: {
            let mut cfg = egui_wgpu::WgpuConfiguration::default();
            // Windows: run the app on a device FFmpeg shares for Vulkan Video
            // decode, so the live preview decodes where it draws (no D3D11),
            // presenting in MAILBOX mode (see `vulkan_decode_ui_setup`). Any
            // failure keeps the device eframe creates below.
            if let Some(existing) = vulkan_ui {
                cfg.wgpu_setup = egui_wgpu::WgpuSetup::Existing(existing);
                cfg.present_mode = wgpu::PresentMode::Mailbox;
            }
            if let egui_wgpu::WgpuSetup::CreateNew(create) = &mut cfg.wgpu_setup {
                #[cfg(target_os = "windows")]
                if std::env::var_os("WGPU_BACKEND").is_none() { // env override wins
                    let probe = wgpu::Instance::new(wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::VULKAN,
                        ..wgpu::InstanceDescriptor::new_without_display_handle()
                    });
                    if pollster::block_on(probe.enumerate_adapters(wgpu::Backends::VULKAN)).is_empty() {
                        tracing::warn!(
                            "no Vulkan adapter — falling back to wgpu's default \
                             backend pick; GPU fast export paths will be disabled"
                        );
                    } else {
                        create.instance_descriptor.backends = wgpu::Backends::VULKAN;
                    }
                }
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
                        // The adapter's 2D texture limit (16384 on every desktop
                        // GPU; the 8192 default would cap a 15520-wide still at
                        // half its resolution). Never above what it reports.
                        required_limits: wgpu::Limits {
                            max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d.clamp(8192, 16384),
                            ..wgpu::Limits::default()
                        },
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
        "VR180 Silver Bullet",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, mailbox)))),
    ).map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    Ok(())
}
