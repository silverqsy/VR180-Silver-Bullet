//! Hardware regression check: cargo run -p vr180-pipeline --example
//! windows_adapter_smoke -- <dual-stream.osv>
//! Checks each Vulkan hardware adapter, both eyes, and segment reopening.
#[cfg(target_os = "windows")]
fn main() -> anyhow::Result<()> {
    use std::{path::PathBuf, sync::Arc};
    use vr180_pipeline::{
        fisheye_decode::SegmentedD3d11SharedDualStreamIter as Iter,
        gpu::Device,
        interop_windows::{vulkan_device_luid, VulkanImportCtx},
    };
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let path = PathBuf::from(std::env::args_os().nth(1).expect("provide a dual-stream OSV"));
    let segments = vec![path.clone(), path];
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let mut checked = 0;
    for adapter in adapters {
        let info = adapter.get_info();
        if !matches!(info.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu) {
            continue;
        }
        println!("TEST adapter: {}", info.name);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            ..Default::default()
        }))?;
        let pipeline = Device::from_existing(
            instance.clone(), Arc::new(adapter), Arc::new(device), Arc::new(queue),
        )?;
        let ctx = VulkanImportCtx::from_wgpu(&pipeline.adapter, &pipeline.device).unwrap();
        let luid = vulkan_device_luid(&ctx);
        anyhow::ensure!(luid.is_some(), "missing Vulkan LUID");
        anyhow::ensure!(Iter::new(&segments, true, 1280, 1280, None).is_err(),
            "unknown adapter must reject zero-copy");
        anyhow::ensure!(Iter::new(&segments, true, 1280, 1280, Some([255; 8])).is_err(),
            "nonexistent adapter must reject zero-copy");
        let mut iter = Iter::new(&segments, true, 1280, 1280, luid)?;
        let seam = iter.total_duration_s() / 2.0;
        // Read both eyes before and after crossing a segment, then seek back.
        for seek in [None, Some(seam + 0.1), Some(0.0)] {
            if let Some(t) = seek { iter.seek(t)?; }
            for _ in 0..3 {
                let pair = iter.next_pair()?.expect("expected decoded frame");
                for eye in [&pair.left, &pair.right] {
                    // Decoder initialization verified this exact LUID; the
                    // iterator fenced D3D11 writes. Keep the pair alive until
                    // Vulkan readback completes.
                    let tex = unsafe { ctx.import_rgba16(&pipeline.device, eye) };
                    let bytes = pipeline.read_texture_rgba64(&tex, eye.width, eye.height)?;
                    anyhow::ensure!(bytes.len() == (eye.width * eye.height * 8) as usize);
                    let rgb: Vec<u16> = bytes.chunks_exact(8).flat_map(|p| {
                        [u16::from_le_bytes([p[0], p[1]]),
                         u16::from_le_bytes([p[2], p[3]]),
                         u16::from_le_bytes([p[4], p[5]])]
                    }).collect();
                    anyhow::ensure!(rgb.iter().min() != rgb.iter().max(),
                        "imported image is uniform");
                }
            }
        }
        println!("PASS {}: 18 eye imports/readbacks, segment seeks, missing/invalid LUID rejection", info.name);
        checked += 1;
    }
    anyhow::ensure!(checked > 0, "no hardware Vulkan adapters tested");
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("This hardware smoke test requires Windows.");
}
