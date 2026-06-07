//! D3D11 (NVDEC / D3D11VA decode output) ↔ Vulkan (wgpu) zero-copy interop.
//!
//! On Windows, ffmpeg's `d3d11va` hwaccel leaves the decoded HEVC frame in a
//! GPU-resident `ID3D11Texture2D` (NV12 / P010). To project it without the
//! GPU→CPU→GPU round-trip the current path pays, we import that texture into
//! wgpu's Vulkan device and feed it straight to the `fisheye_p010_*` shaders
//! — the Windows analogue of the macOS IOSurface path in `interop_macos`.
//!
//! This is feasible on **wgpu 29**: its Vulkan backend auto-enables
//! `VK_KHR_external_memory_win32` when the adapter supports it (verified on
//! the RTX 4090), and exposes `Device::raw_device()` via the `as_hal` escape.
//!
//! ## Technique (standard, well-documented; original MIT implementation —
//! ## NOT derived from any GPL source such as Gyroflow)
//! 1. D3D11 side (ffmpeg's decode device, background thread):
//!    - The decoder hands back `AV_PIX_FMT_D3D11`: `frame.data[0]` is the
//!      `ID3D11Texture2D` (a DPB **array**), `frame.data[1]` the array slice.
//!    - `CopySubresourceRegion` that slice into a single-subresource texture
//!      created with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE | ..._KEYEDMUTEX`.
//!    - `CreateSharedHandle` → an NT `HANDLE`.
//! 2. Vulkan side (wgpu device, main thread):
//!    - `vkGetMemoryWin32HandlePropertiesKHR` → memory type bits.
//!    - `vkCreateImage` with `VkExternalMemoryImageCreateInfo`
//!      (D3D11_TEXTURE handle type), P010 → `G10X6_B10X6R10X6_2PLANE_420`.
//!    - `vkAllocateMemory` with `VkImportMemoryWin32HandleInfoKHR` +
//!      `VkMemoryDedicatedAllocateInfo`, bind, then
//!      `wgpu_hal::vulkan::Device::texture_from_raw(.., TextureMemory::External)`
//!      → `Device::create_texture_from_hal::<Vulkan>()`.
//!    - Per-plane views (`TextureAspect::Plane0/1`) feed the existing
//!      `project_fisheye_p010_*` shaders.
//! 3. Sync: `IDXGIKeyedMutex` acquire/release across the two devices, mirrored
//!    on the Vulkan submit via `VkWin32KeyedMutexAcquireReleaseInfoKHR`.
//!
//! Status: foundation (the wgpu→Vulkan escape) is implemented and verified
//! below; the D3D11-share + Vulkan-import steps are built on top of it.

use ash::vk;
use wgpu::hal::api::Vulkan;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HMODULE, LUID};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4, IDXGIResource1};

/// Borrow the raw `ash::Device` backing a wgpu device, if wgpu is running on
/// the Vulkan backend. The returned `ash::Device` is a cheap clonable handle
/// (it does not own the underlying `VkDevice`; wgpu still owns its lifetime).
///
/// # Safety
/// The caller must not destroy the `VkDevice` or use it after the wgpu device
/// is dropped. We only ever use it to create/import resources that wgpu then
/// takes ownership of via `create_texture_from_hal`.
pub fn vulkan_raw_device(device: &wgpu::Device) -> Option<ash::Device> {
    // SAFETY: we only read the raw device handle and clone the ash wrapper;
    // we uphold wgpu-hal's invariant of not outliving the wgpu device.
    unsafe {
        device
            .as_hal::<Vulkan>()
            .map(|hal_device| hal_device.raw_device().clone())
    }
}

/// True if the given wgpu device is backed by the Vulkan HAL (a precondition
/// for the D3D11→Vulkan import path). On a DX12/GL/Metal wgpu device the
/// zero-copy import is unavailable and callers fall back to the CPU path.
pub fn is_vulkan_backend(device: &wgpu::Device) -> bool {
    // SAFETY: read-only probe of the active HAL backend.
    unsafe { device.as_hal::<Vulkan>().is_some() }
}

/// Raw Vulkan handles + the external-memory device-extension loader, pulled
/// from a wgpu adapter+device (Vulkan backend) — everything the D3D11→Vulkan
/// import path needs. The `ash::Instance`/`ash::Device` are clonable handles
/// that share wgpu's underlying `VkInstance`/`VkDevice`; we never destroy
/// them (wgpu owns their lifetime).
pub struct VulkanImportCtx {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    /// `VK_KHR_external_memory_win32` device fns
    /// (`vkGetMemoryWin32HandlePropertiesKHR`). wgpu-hal 29 enables the
    /// extension automatically when the adapter supports it.
    pub ext_mem_win32: ash::khr::external_memory_win32::Device,
}

impl VulkanImportCtx {
    /// Extract from a wgpu adapter + device. Returns `None` if wgpu isn't on
    /// the Vulkan backend (DX12/GL/Metal), in which case the zero-copy import
    /// is unavailable and the caller falls back to the CPU path.
    pub fn from_wgpu(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<Self> {
        // SAFETY: we only read handles and clone the ash wrappers; the
        // handles must not outlive the wgpu device, which the caller keeps
        // alive for the duration of any import built on this context.
        unsafe {
            let (instance, physical_device) = {
                let hal_adapter = adapter.as_hal::<Vulkan>()?;
                (
                    hal_adapter.shared_instance().raw_instance().clone(),
                    hal_adapter.raw_physical_device(),
                )
            };
            let ash_device = {
                let hal_device = device.as_hal::<Vulkan>()?;
                hal_device.raw_device().clone()
            };
            let ext_mem_win32 =
                ash::khr::external_memory_win32::Device::new(&instance, &ash_device);
            Some(Self {
                instance,
                physical_device,
                device: ash_device,
                ext_mem_win32,
            })
        }
    }
}

impl std::fmt::Debug for VulkanImportCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanImportCtx")
            .field("physical_device", &self.physical_device)
            .finish_non_exhaustive()
    }
}

/// Query the 8-byte adapter LUID of the wgpu Vulkan device. Cross-API memory
/// sharing requires the D3D11 device to sit on the *same* physical adapter,
/// which we match by this LUID.
pub fn vulkan_device_luid(ctx: &VulkanImportCtx) -> Option<[u8; 8]> {
    // SAFETY: standard Vulkan 1.1 property query; `id_props` is filled via the
    // pNext chain before `props2` (which borrows it) is dropped.
    unsafe {
        let mut id_props = vk::PhysicalDeviceIDProperties::default();
        {
            let mut props2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
            ctx.instance
                .get_physical_device_properties2(ctx.physical_device, &mut props2);
        }
        (id_props.device_luid_valid == vk::TRUE).then_some(id_props.device_luid)
    }
}

/// A D3D11 texture shared out as an NT handle, ready to import into Vulkan.
/// Holds the owning device/context alive (the handle and texture stay valid
/// while this struct lives).
pub struct D3d11SharedTexture {
    pub device: ID3D11Device,
    pub context: ID3D11DeviceContext,
    pub texture: ID3D11Texture2D,
    pub handle: HANDLE,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Debug for D3d11SharedTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d11SharedTexture")
            .field("handle", &self.handle.0)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl D3d11SharedTexture {
    /// Block until the GPU has finished producing this texture's contents
    /// (the convert/copy that wrote it). Call this AFTER submitting all the
    /// work for a frame so the two eyes' GPU work overlaps, and BEFORE the
    /// importing thread reads the shared handle. Runs on the decode worker
    /// thread (never wgpu/main), so blocking here is safe.
    ///
    /// # Safety
    /// `device`/`context` must still be live (they are, as long as `self` is).
    pub unsafe fn wait_gpu_idle(&self) {
        d3d11_wait_gpu_idle(&self.device, &self.context);
    }
}

impl Drop for D3d11SharedTexture {
    fn drop(&mut self) {
        // Close the NT handle exactly once. A successful Vulkan import dup'd its
        // own reference to the underlying resource, so the imported memory is
        // unaffected; the D3D11 texture/device/context COM refs are released by
        // their own `Drop`. `import_d3d11_handle_to_wgpu` never closes the handle,
        // so there's no double-close. A null handle (never created) is skipped.
        if !self.handle.is_invalid() {
            // SAFETY: `handle` is a valid NT handle we created via
            // `CreateSharedHandle` and have not closed yet.
            unsafe { let _ = CloseHandle(self.handle); }
        }
    }
}

/// Create an `RGBA8` D3D11 texture on the adapter identified by `luid`,
/// initialised with `rgba` (`w*h*4` bytes), shared as an NT handle. Used by
/// the interop smoke test to validate the D3D11→Vulkan path with known
/// pixels; the real path reuses the same share/handle technique on ffmpeg's
/// d3d11va decode texture.
pub fn create_d3d11_test_texture(
    luid: [u8; 8],
    w: u32,
    h: u32,
    rgba: &[u8],
) -> windows::core::Result<D3d11SharedTexture> {
    // SAFETY: direct D3D11/DXGI COM calls; every fallible call is `?`-checked
    // and out-params are initialised to `None` before use.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1()?;
        let target = LUID {
            LowPart: u32::from_le_bytes([luid[0], luid[1], luid[2], luid[3]]),
            HighPart: i32::from_le_bytes([luid[4], luid[5], luid[6], luid[7]]),
        };
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(target)?;

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
        let device = device.unwrap();
        let context = context.unwrap();

        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32,
        };
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: rgba.as_ptr() as *const core::ffi::c_void,
            SysMemPitch: w * 4,
            SysMemSlicePitch: 0,
        };
        let mut tex: Option<ID3D11Texture2D> = None;
        device.CreateTexture2D(&desc, Some(&init), Some(&mut tex))?;
        let texture = tex.unwrap();
        // Flush so the initial-data upload completes and is visible to the
        // importing (Vulkan) device before it reads the shared texture.
        context.Flush();

        let dxgi_res: IDXGIResource1 = texture.cast()?;
        let handle = dxgi_res.CreateSharedHandle(None, GENERIC_ALL.0, None)?;

        Ok(D3d11SharedTexture {
            device,
            context,
            texture,
            handle,
            width: w,
            height: h,
        })
    }
}

/// Block the calling thread until the GPU has completed the work submitted to
/// `context` so far — used to fence a `CopySubresourceRegion` before its result
/// is read across-API (on the Vulkan device, from another thread). Uses a
/// `D3D11_QUERY_EVENT`: `End` records a marker after the copy, and `GetData`
/// returns `S_OK` once the GPU retires it. Best-effort (returns immediately if
/// the query can't be created).
///
/// # Safety
/// `device`/`context` must be live D3D11 objects on the same device.
unsafe fn d3d11_wait_gpu_idle(device: &ID3D11Device, context: &ID3D11DeviceContext) {
    // Measurement escape hatch (repro only): skip the GPU fence to isolate how
    // much of the worker time is the GPU wait vs CPU-side resource creation.
    if std::env::var_os("VR180_NOFENCE").is_some() {
        return;
    }
    use windows::Win32::Graphics::Direct3D11::{D3D11_QUERY_DESC, D3D11_QUERY_EVENT, ID3D11Query};
    let qdesc = D3D11_QUERY_DESC { Query: D3D11_QUERY_EVENT, MiscFlags: 0 };
    let mut query: Option<ID3D11Query> = None;
    if device.CreateQuery(&qdesc, Some(&mut query)).is_err() {
        return;
    }
    let Some(query) = query else { return };
    context.End(&query);
    // Poll until the event retires (the copy completes in well under a
    // millisecond). `GetData` writes a non-zero BOOL only once the GPU has
    // signaled the event; while pending it returns S_FALSE and leaves the
    // output untouched. (windows-rs maps both S_OK and S_FALSE to `Ok`, so we
    // read completion from the written BOOL, not the HRESULT.) Bounded so a
    // driver hiccup can't hang the worker forever.
    let mut done: i32 = 0;
    for _ in 0..5_000_000u32 {
        let _ = context.GetData(
            &query,
            Some(&mut done as *mut i32 as *mut core::ffi::c_void),
            std::mem::size_of::<i32>() as u32,
            0,
        );
        if done != 0 {
            return;
        }
        // Yield rather than busy-spin: the wait is GPU-bound (multiple ms), so
        // hand the core to the decoder/main threads instead of burning it.
        std::thread::yield_now();
    }
    tracing::warn!("d3d11_wait_gpu_idle: query did not retire within bound");
}

/// Copy one array slice of a D3D11 texture (e.g. a `d3d11va` decoder DPB
/// slice) into a fresh single-slice **shareable** texture of the same format,
/// flush, and export an NT handle — ready to import into Vulkan. This is the
/// per-frame analogue of `create_d3d11_test_texture` but sourcing pixels from
/// an existing GPU texture instead of CPU init data, so the frame never
/// touches system memory.
///
/// # Safety
/// `src`/`device`/`context` must be live D3D11 objects on the same device,
/// and `slice` a valid subresource index of `src`.
pub unsafe fn share_d3d11_texture_slice(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    src: &ID3D11Texture2D,
    slice: u32,
) -> windows::core::Result<D3d11SharedTexture> {
    let mut sdesc = D3D11_TEXTURE2D_DESC::default();
    src.GetDesc(&mut sdesc);
    let (w, h) = (sdesc.Width, sdesc.Height);
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: sdesc.Format, // e.g. DXGI_FORMAT_P010 for 10-bit HEVC
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32,
    };
    let mut tex: Option<ID3D11Texture2D> = None;
    device.CreateTexture2D(&desc, None, Some(&mut tex))?;
    let texture = tex.unwrap();
    // GPU→GPU copy of the decoder's slice into our shareable texture.
    context.CopySubresourceRegion(&texture, 0, 0, 0, 0, src, slice, None);
    context.Flush();
    // ── Cross-API sync ──────────────────────────────────────────────
    // Block until the GPU has actually FINISHED the copy before we hand the
    // shared handle to the (other-thread) Vulkan importer. `Flush` only
    // *submits* the copy; it returns immediately. Without waiting, the
    // importer can sample the texture mid-copy — and because P010's chroma
    // plane is copied after the luma plane, the symptom is correct luma with
    // stale/garbled chroma (the orange/blue cast). A D3D11 EVENT query
    // signals when the GPU reaches this point, i.e. the copy is done and the
    // bytes are live in VRAM; the later Vulkan import/sample then reads
    // completed data with no cross-queue primitive needed. This runs on the
    // decode worker thread (never the wgpu/main thread), so the brief CPU spin
    // is safe — no Lesson-#1 deadlock risk.
    d3d11_wait_gpu_idle(device, context);
    let dxgi_res: IDXGIResource1 = texture.cast()?;
    let handle = dxgi_res.CreateSharedHandle(None, GENERIC_ALL.0, None)?;
    Ok(D3d11SharedTexture {
        device: device.clone(),
        context: context.clone(),
        texture,
        handle,
        width: w,
        height: h,
    })
}

/// HLSL: P010 (Y plane R16 SRV + UV plane Rg16 SRV) → RGBA16 UAV, with BT.709
/// limited-range expansion and a box downscale to the output dims. Done on the
/// D3D11 side because the multi-plane P010 texture imports into Vulkan with a
/// broken chroma-plane offset — but a single-plane RGBA16 imports cleanly.
/// SRVs (unlike `CopySubresourceRegion`) DO work on P010 planes.
const P010_TO_RGBA16_HLSL: &str = r#"
Texture2D<float>  Ytex  : register(t0);
Texture2D<float2> UVtex : register(t1);
SamplerState      smp   : register(s0);
RWTexture2D<unorm float4> Outp : register(u0);
cbuffer CB : register(b0) { uint src_w; uint src_h; uint out_w; uint out_h; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= out_w || id.y >= out_h) return;
    float sx = (float)src_w / (float)out_w;
    float sy = (float)src_h / (float)out_h;
    int kx = max(1, (int)ceil(sx));
    int ky = max(1, (int)ceil(sy));
    float cx = ((float)id.x + 0.5) * sx;
    float cy = ((float)id.y + 0.5) * sy;
    float3 acc = float3(0, 0, 0);
    [loop] for (int j = 0; j < ky; j++) {
        [loop] for (int i = 0; i < kx; i++) {
            float ox = (((float)i + 0.5) / (float)kx - 0.5) * sx;
            float oy = (((float)j + 0.5) / (float)ky - 0.5) * sy;
            float2 uvc = float2(cx + ox, cy + oy) / float2((float)src_w, (float)src_h);
            float y  = Ytex.SampleLevel(smp, uvc, 0);
            float2 c = UVtex.SampleLevel(smp, uvc, 0);
            float yl = y   * (65535.0 / 56064.0) - (64.0 / 876.0);
            float ul = c.x * (65535.0 / 57344.0) - (512.0 / 896.0);
            float vl = c.y * (65535.0 / 57344.0) - (512.0 / 896.0);
            acc += float3(
                saturate(yl + 1.5748 * vl),
                saturate(yl - 0.1873 * ul - 0.4681 * vl),
                saturate(yl + 1.8556 * ul));
        }
    }
    acc /= (float)(kx * ky);
    Outp[id.xy] = float4(acc, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvDims {
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
}

/// A cached D3D11 compute pipeline that converts a P010 texture to a
/// single-plane RGBA16 (+ box downscale). One per D3D11 device (the OSV
/// streams decode on separate `d3d11va` devices, so the iterator keeps one
/// converter per stream). Built once; `convert` runs per frame.
pub struct P010Converter {
    cs: windows::Win32::Graphics::Direct3D11::ID3D11ComputeShader,
    sampler: windows::Win32::Graphics::Direct3D11::ID3D11SamplerState,
    cbuffer: windows::Win32::Graphics::Direct3D11::ID3D11Buffer,
}

impl std::fmt::Debug for P010Converter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("P010Converter").finish_non_exhaustive()
    }
}

impl P010Converter {
    /// Compile the converter for a specific D3D11 device.
    pub fn new(device: &ID3D11Device) -> windows::core::Result<Self> {
        use windows::core::s;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_BIND_CONSTANT_BUFFER, D3D11_BUFFER_DESC, D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            D3D11_SAMPLER_DESC, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_USAGE_DEFAULT,
            ID3D11ComputeShader, ID3D11SamplerState,
        };
        unsafe {
            let mut blob: Option<ID3DBlob> = None;
            let mut errs: Option<ID3DBlob> = None;
            let hr = D3DCompile(
                P010_TO_RGBA16_HLSL.as_ptr() as *const core::ffi::c_void,
                P010_TO_RGBA16_HLSL.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_0"),
                0,
                0,
                &mut blob,
                Some(&mut errs),
            );
            if hr.is_err() {
                if let Some(e) = errs {
                    let msg = std::slice::from_raw_parts(
                        e.GetBufferPointer() as *const u8, e.GetBufferSize());
                    tracing::error!("P010Converter HLSL compile: {}", String::from_utf8_lossy(msg));
                }
                hr?;
            }
            let blob = blob.unwrap();
            let bytecode = std::slice::from_raw_parts(
                blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
            let mut cs: Option<ID3D11ComputeShader> = None;
            device.CreateComputeShader(bytecode, None, Some(&mut cs))?;

            let sdesc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MaxLOD: f32::MAX,
                ..Default::default()
            };
            let mut sampler: Option<ID3D11SamplerState> = None;
            device.CreateSamplerState(&sdesc, Some(&mut sampler))?;

            let bdesc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<ConvDims>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                ..Default::default()
            };
            let mut cbuffer = None;
            device.CreateBuffer(&bdesc, None, Some(&mut cbuffer))?;

            Ok(Self { cs: cs.unwrap(), sampler: sampler.unwrap(), cbuffer: cbuffer.unwrap() })
        }
    }

    /// Convert a single-slice P010 texture (must have `BIND_SHADER_RESOURCE`)
    /// to a fresh shareable RGBA16 texture at `out_w × out_h`, exported as an
    /// NT handle ready to import single-plane into Vulkan.
    ///
    /// # Safety
    /// `device`/`context` must own `src_p010`; `src_p010` must be a live
    /// shader-resource P010 texture of `src_w × src_h`.
    pub unsafe fn convert(
        &self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        src_p010: &ID3D11Texture2D,
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) -> windows::core::Result<D3d11SharedTexture> {
        use windows::Win32::Graphics::Direct3D::D3D11_SRV_DIMENSION_TEXTURE2D;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS,
            D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC_0,
            D3D11_TEX2D_SRV, D3D11_TEX2D_UAV, D3D11_UAV_DIMENSION_TEXTURE2D,
            D3D11_UNORDERED_ACCESS_VIEW_DESC, D3D11_UNORDERED_ACCESS_VIEW_DESC_0,
            D3D11_USAGE_DEFAULT, ID3D11ShaderResourceView, ID3D11UnorderedAccessView,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16_UNORM,
            DXGI_SAMPLE_DESC,
        };

        // Output RGBA16 texture: UAV (compute writes) + SRV (Vulkan import
        // samples) + shareable NT handle.
        let odesc = D3D11_TEXTURE2D_DESC {
            Width: out_w,
            Height: out_h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R16G16B16A16_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32,
        };
        let mut out_tex: Option<ID3D11Texture2D> = None;
        device.CreateTexture2D(&odesc, None, Some(&mut out_tex))?;
        let out_tex = out_tex.unwrap();

        // Plane SRVs on the P010 (format selects the plane).
        let mk_srv = |fmt| -> windows::core::Result<ID3D11ShaderResourceView> {
            let d = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: fmt,
                ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1 },
                },
            };
            let mut v = None;
            device.CreateShaderResourceView(src_p010, Some(&d), Some(&mut v))?;
            Ok(v.unwrap())
        };
        let srv_y = mk_srv(DXGI_FORMAT_R16_UNORM)?;
        let srv_uv = mk_srv(DXGI_FORMAT_R16G16_UNORM)?;

        let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_UNORM,
            ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
            },
        };
        let mut uav: Option<ID3D11UnorderedAccessView> = None;
        device.CreateUnorderedAccessView(&out_tex, Some(&uav_desc), Some(&mut uav))?;
        let _ = D3D11_USAGE_DEFAULT;

        let dims = ConvDims { src_w, src_h, out_w, out_h };
        context.UpdateSubresource(
            &self.cbuffer, 0, None,
            &dims as *const ConvDims as *const core::ffi::c_void, 0, 0,
        );

        context.CSSetShader(&self.cs, None);
        context.CSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
        context.CSSetConstantBuffers(0, Some(&[Some(self.cbuffer.clone())]));
        context.CSSetShaderResources(0, Some(&[Some(srv_y.clone()), Some(srv_uv.clone())]));
        context.CSSetUnorderedAccessViews(0, 1, Some(&Some(uav.clone().unwrap())), None);
        context.Dispatch((out_w + 7) / 8, (out_h + 7) / 8, 1);
        // Unbind the UAV so the texture can be read/shared.
        context.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
        context.Flush();
        // NOTE: no GPU fence here — the caller converts BOTH eyes (on their two
        // separate d3d11va devices) and then fences both via
        // `D3d11SharedTexture::wait_gpu_idle`, so the two streams' decode+convert
        // overlap on the GPU instead of serializing.
        let dxgi_res: IDXGIResource1 = out_tex.cast()?;
        let handle = dxgi_res.CreateSharedHandle(None, GENERIC_ALL.0, None)?;
        Ok(D3d11SharedTexture {
            device: device.clone(),
            context: context.clone(),
            texture: out_tex,
            handle,
            width: out_w,
            height: out_h,
        })
    }
}

impl VulkanImportCtx {
    /// Import a shared **RGBA16** eye (the D3D11-converted single-plane output
    /// of [`P010Converter::convert`]) into wgpu as `Rgba16Unorm`. Single-plane,
    /// so it imports cleanly (unlike the multi-plane P010).
    ///
    /// # Safety
    /// `eye.handle` must be a live shared NT handle for an `R16G16B16A16_UNORM`
    /// texture of `eye.width`×`eye.height` on the same device as `self`.
    pub unsafe fn import_rgba16(
        &self,
        wgpu_device: &wgpu::Device,
        eye: &D3d11SharedTexture,
    ) -> wgpu::Texture {
        import_d3d11_handle_to_wgpu(
            wgpu_device,
            self,
            eye.handle,
            eye.width,
            eye.height,
            wgpu::TextureFormat::Rgba16Unorm,
            vk::Format::R16G16B16A16_UNORM,
        )
    }
}

/// Minimal mirror of libavutil's `AVD3D11VADeviceContext` (ffmpeg-sys-next
/// doesn't generate a binding for it). Field order/layout matches
/// `libavutil/hwcontext_d3d11va.h`; we only read `device`/`device_context`.
#[repr(C)]
struct D3D11VADeviceContextLayout {
    device: *mut core::ffi::c_void,         // ID3D11Device*
    device_context: *mut core::ffi::c_void, // ID3D11DeviceContext*
    video_device: *mut core::ffi::c_void,
    video_context: *mut core::ffi::c_void,
    lock: *mut core::ffi::c_void,
    unlock: *mut core::ffi::c_void,
    lock_ctx: *mut core::ffi::c_void,
}

/// Pull the decoder-output D3D11 texture (and its array-slice index) plus the
/// ffmpeg-owned D3D11 device/context out of a `d3d11va`-decoded frame
/// (`AV_PIX_FMT_D3D11`). The decoded texture is one slice of the decoder's DPB
/// **array**; callers `CopySubresourceRegion` `array_index` into a shareable
/// texture created on the returned `device`.
///
/// Returned COM interfaces are AddRef'd clones (independently valid). Returns
/// `None` if the frame isn't a d3d11va GPU frame.
///
/// # Safety
/// `frame` must be a live `AV_PIX_FMT_D3D11` frame from a d3d11va decoder.
pub unsafe fn extract_d3d11_from_frame(
    frame: &ffmpeg_next::frame::Video,
) -> Option<(ID3D11Texture2D, u32, ID3D11Device, ID3D11DeviceContext)> {
    use ffmpeg_next::ffi::{AVHWDeviceContext, AVHWFramesContext};
    let av = frame.as_ptr();
    if av.is_null() {
        return None;
    }
    let tex_raw = (*av).data[0] as *mut core::ffi::c_void;
    if tex_raw.is_null() {
        return None;
    }
    let array_index = (*av).data[1] as usize as u32;

    let frames_ref = (*av).hw_frames_ctx;
    if frames_ref.is_null() {
        return None;
    }
    let fctx = (*frames_ref).data as *const AVHWFramesContext;
    let dev_ref = (*fctx).device_ref;
    if dev_ref.is_null() {
        return None;
    }
    let dctx = (*dev_ref).data as *const AVHWDeviceContext;
    let d3d11ctx = (*dctx).hwctx as *const D3D11VADeviceContextLayout;
    if d3d11ctx.is_null() {
        return None;
    }
    let device_raw = (*d3d11ctx).device as *mut core::ffi::c_void;
    let context_raw = (*d3d11ctx).device_context as *mut core::ffi::c_void;

    let tex = ID3D11Texture2D::from_raw_borrowed(&tex_raw)?.clone();
    let device = ID3D11Device::from_raw_borrowed(&device_raw)?.clone();
    let context = ID3D11DeviceContext::from_raw_borrowed(&context_raw)?.clone();
    Some((tex, array_index, device, context))
}

/// Copy one array slice of a `d3d11va` P010 DPB texture into a fresh
/// single-slice **shader-resource** P010 (no share). The decoder's DPB array
/// has only `BIND_DECODER`, so it can't be SRV-sampled directly; this gives us
/// an SRV-able copy for [`P010Converter::convert`].
///
/// # Safety
/// `device`/`context` own `src`; `slice` is a valid subresource of `src`.
unsafe fn copy_slice_to_shader_p010(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    src: &ID3D11Texture2D,
    slice: u32,
    w: u32,
    h: u32,
) -> Option<ID3D11Texture2D> {
    let mut sdesc = D3D11_TEXTURE2D_DESC::default();
    src.GetDesc(&mut sdesc);
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: sdesc.Format, // P010
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut tex: Option<ID3D11Texture2D> = None;
    device.CreateTexture2D(&desc, None, Some(&mut tex)).ok()?;
    let tex = tex.unwrap();
    context.CopySubresourceRegion(&tex, 0, 0, 0, 0, src, slice, None);
    // No flush here — `convert` flushes + GPU-fences after its compute pass,
    // which is ordered after this copy on the same context.
    Some(tex)
}

/// Decode-frame → shareable **RGBA16** eye, doing the YCbCr→RGB on the D3D11
/// side (the multi-plane P010 imports into Vulkan with a broken chroma-plane
/// offset; a single-plane RGBA16 imports cleanly). Lazily builds the converter
/// for the frame's device. `work_w`/`work_h` is the (downscaled) preview
/// working resolution. After this returns the source frame can be dropped.
///
/// # Safety
/// `frame` must be a live `AV_PIX_FMT_D3D11` P010 frame from a d3d11va decoder.
pub unsafe fn share_eye_converted(
    frame: &ffmpeg_next::frame::Video,
    converter: &mut Option<P010Converter>,
    work_w: u32,
    work_h: u32,
) -> Option<D3d11SharedTexture> {
    let (tex, slice, dev, dctx) = extract_d3d11_from_frame(frame)?;
    if converter.is_none() {
        match P010Converter::new(&dev) {
            Ok(c) => *converter = Some(c),
            Err(e) => {
                tracing::error!("P010Converter::new failed: {e}");
                return None;
            }
        }
    }
    let conv = converter.as_ref().unwrap();
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    tex.GetDesc(&mut desc);
    let (nw, nh) = (desc.Width, desc.Height);
    let p010 = copy_slice_to_shader_p010(&dev, &dctx, &tex, slice, nw, nh)?;
    match conv.convert(&dev, &dctx, &p010, nw, nh, work_w, work_h) {
        Ok(shared) => Some(shared),
        Err(e) => {
            tracing::warn!("share_eye_converted: convert failed: {e}");
            None
        }
    }
}

/// One-call extract + share for a `d3d11va`-decoded frame: pull the decoder's
/// DPB texture/slice/device, `CopySubresourceRegion` the slice into a fresh
/// shareable single-slice texture, and export it as an NT handle. After this
/// returns, the source `frame` can be dropped/recycled by ffmpeg immediately —
/// the pixels live in our own copy. Keeps all `windows`/COM types inside this
/// crate so the GUI worker can stay dependency-free.
///
/// Returns `None` if the frame isn't a d3d11va GPU frame or the share fails.
///
/// # Safety
/// `frame` must be a live `AV_PIX_FMT_D3D11` frame from a d3d11va decoder.
pub unsafe fn share_eye_from_frame(
    frame: &ffmpeg_next::frame::Video,
) -> Option<D3d11SharedTexture> {
    let (tex, slice, dev, dctx) = extract_d3d11_from_frame(frame)?;
    match share_d3d11_texture_slice(&dev, &dctx, &tex, slice) {
        Ok(shared) => Some(shared),
        Err(e) => {
            tracing::warn!("share_eye_from_frame: share slice failed: {e}");
            None
        }
    }
}

/// Import a shared D3D11 NT handle into wgpu's Vulkan device as a wgpu
/// `Texture` that **aliases the D3D11 texture's GPU memory** (zero-copy).
/// `wgpu_format` / `vk_format` must be a matching pair (e.g. `Rgba8Unorm` /
/// `R8G8B8A8_UNORM`). The wgpu texture does not own the memory (`External`);
/// wgpu won't free it.
///
/// # Safety
/// `handle` must be a valid shared NT handle for a texture of `w`×`h` in
/// `vk_format` on the same physical device as `ctx`.
pub unsafe fn import_d3d11_handle_to_wgpu(
    wgpu_device: &wgpu::Device,
    ctx: &VulkanImportCtx,
    handle: HANDLE,
    w: u32,
    h: u32,
    wgpu_format: wgpu::TextureFormat,
    vk_format: vk::Format,
) -> wgpu::Texture {
    let handle_type = vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;

    // 1. Create a VkImage flagged for external memory.
    let mut ext = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
    let img_ci = vk::ImageCreateInfo::default()
        .push_next(&mut ext)
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = ctx.device.create_image(&img_ci, None).expect("vk create_image");

    // 2. Pick a memory type satisfying both the image and the imported handle.
    let mem_req = ctx.device.get_image_memory_requirements(image);
    let mut hprops = vk::MemoryWin32HandlePropertiesKHR::default();
    ctx.ext_mem_win32
        .get_memory_win32_handle_properties(handle_type, handle.0 as _, &mut hprops)
        .expect("get_memory_win32_handle_properties");
    let mem_props = ctx
        .instance
        .get_physical_device_memory_properties(ctx.physical_device);
    let bits = mem_req.memory_type_bits & hprops.memory_type_bits;
    let mem_type_index = (0..mem_props.memory_type_count)
        .find(|&i| (bits & (1 << i)) != 0)
        .expect("no compatible memory type for imported handle");

    // 3. Import the handle as dedicated device memory and bind it.
    let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(handle_type)
        .handle(handle.0 as _);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let alloc = vk::MemoryAllocateInfo::default()
        .push_next(&mut import)
        .push_next(&mut dedicated)
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type_index);
    let memory = ctx.device.allocate_memory(&alloc, None).expect("import device memory");
    ctx.device.bind_image_memory(image, memory, 0).expect("bind_image_memory");

    // 4. Wrap the VkImage as a wgpu texture (external — wgpu won't free it).
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("d3d11-imported"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    let hal_tex = {
        let hal_device = wgpu_device.as_hal::<Vulkan>().expect("wgpu not on Vulkan");
        // `Dedicated(memory)` (not `External`): hand the imported VkDeviceMemory
        // to wgpu so it frees BOTH the VkImage and the VkDeviceMemory when the
        // texture drops (`destroy_texture` → `free_memory` + `destroy_image`).
        // `External` would leak a whole frame's memory + image every call — fatal
        // for live playback at 30-50 fps. The NT handle is owned separately and
        // closed in `D3d11SharedTexture::drop`; Vulkan dup'd its own reference at
        // import time, so the imported memory stays valid after the handle closes.
        hal_device.texture_from_raw(
            image,
            &hal_desc,
            None,
            wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
        )
    };
    wgpu_device.create_texture_from_hal::<Vulkan>(
        hal_tex,
        &wgpu::TextureDescriptor {
            label: Some("d3d11-imported"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )
}

impl VulkanImportCtx {
    /// Import a shared **P010** eye (the d3d11va NVDEC output) into wgpu as a
    /// zero-copy `wgpu::TextureFormat::P010` texture aliasing the D3D11 memory.
    /// The returned texture's two planes (`Plane0` = Y/R16, `Plane1` = CbCr/Rg16)
    /// feed `Device::project_fisheye_p010_planar_to_equirect_texture_16`.
    ///
    /// Keep `eye` alive until the projection GPU work that reads this texture has
    /// completed (the D3D11 texture backs the imported memory). The convenience
    /// here is purely so callers (the GUI worker) don't have to name `ash`/`vk`
    /// or `windows` types.
    ///
    /// # Safety
    /// `eye.handle` must be a live shared NT handle for a P010 texture of
    /// `eye.width`×`eye.height` on the same physical device as `self`.
    pub unsafe fn import_p010(
        &self,
        wgpu_device: &wgpu::Device,
        eye: &D3d11SharedTexture,
    ) -> wgpu::Texture {
        import_d3d11_handle_to_wgpu(
            wgpu_device,
            self,
            eye.handle,
            eye.width,
            eye.height,
            wgpu::TextureFormat::P010,
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        )
    }
}
