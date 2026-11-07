// braw_zc_win.cpp — the Windows twin of braw_zc.mm: see braw_zc.h.
//
// A pool of SHAREABLE D3D11 textures (NT handles), one or two per slot,
// each an eye on the app's canvas (a SQUARE of the eye's width for a stereo
// pair — a shorter recorded eye is padded black top and bottom, a taller one
// cropped — the recorded frame for a mono clip), RGBA8. The app duplicates
// the handles into its own process and imports them into Vulkan
// (VK_KHR_external_memory_win32, as it does for NVDEC frames), so only the
// slot number crosses the pipe per frame.
//
// The fill is a host round trip inside the helper: the SDK's processed
// image (CPU memory, or CUDA memory read back through the SDK's resource
// manager) goes into the slot's textures with UpdateSubresource, flushed and
// fenced before the frame is reported. A CUDA-interop blit (the SDK's
// CUDA buffer straight into the texture) would save that copy; it is not
// needed at the sizes the preview asks for (eighth / quarter scale).
//
// The D3D11 device sits on the adapter the app named (`--adapter-luid`, the
// LUID of its Vulkan device) — a cross-adapter handle cannot be imported —
// else on the first hardware adapter.
#include "braw_platform.h"
#include "braw_zc.h"

#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct Eye {
    ID3D11Texture2D* texture = nullptr;
    HANDLE handle = nullptr;        // NT handle (GENERIC_ALL), ours until destroy
};

struct Slot { Eye eye[2]; };

uint64_t g_adapterLuid = 0;        // 0 = not specified

// Rows of the source that land in the canvas, and where (the macOS twin's
// row_map, verbatim).
struct RowMap { uint32_t src_y0, dst_y0, rows; };
RowMap row_map(const ZcGeometry& g, uint32_t src_h) {
    RowMap m{0, 0, 0};
    if (g.mono) { m.rows = std::min(src_h, g.eye_h); return m; }
    if (g.pad_rows >= 0) { m.dst_y0 = (uint32_t)g.pad_rows; m.rows = std::min(src_h, g.eye_h - m.dst_y0); }
    else { m.src_y0 = (uint32_t)(-g.pad_rows); m.rows = std::min(src_h - std::min(src_h, m.src_y0), g.eye_h); }
    return m;
}

IDXGIAdapter1* pick_adapter() {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)) || !factory) return nullptr;
    IDXGIAdapter1* chosen = nullptr;
    IDXGIAdapter1* firstHw = nullptr;
    for (UINT i = 0;; i++) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) != S_OK || !a) break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        const bool software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        uint64_t luid = ((uint64_t)(uint32_t)d.AdapterLuid.HighPart << 32) | (uint64_t)d.AdapterLuid.LowPart;
        if (g_adapterLuid != 0 && luid == g_adapterLuid) { chosen = a; break; }
        if (!software && !firstHw) { firstHw = a; continue; }
        a->Release();
    }
    factory->Release();
    if (chosen) { if (firstHw) firstHw->Release(); return chosen; }
    if (g_adapterLuid != 0) fprintf(stderr, "note: --adapter-luid %016llx not found; using the first hardware adapter\n", (unsigned long long)g_adapterLuid);
    return firstHw;
}

} // namespace

struct ZcPool {
    ZcGeometry g;
    std::vector<Slot> slots;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11Query* fence = nullptr;
    std::mutex mu;                  // the immediate context is not thread-safe

    // Block until every command submitted so far has executed.
    void wait_gpu_idle() {
        if (!fence) return;
        context->End(fence);
        context->Flush();
        BOOL done = FALSE;
        while (context->GetData(fence, &done, sizeof done, 0) == S_FALSE) { /* spin: sub-millisecond */ }
    }
};

void zc_set_adapter_luid(uint64_t luid) { g_adapterLuid = luid; }
const char* zc_transport() { return "d3d11"; }
uint32_t zc_owner_pid() { return (uint32_t)GetCurrentProcessId(); }

ZcPool* zc_pool_create(uint32_t slots, const ZcGeometry& g) {
    auto* p = new ZcPool();
    p->g = g;
    IDXGIAdapter1* adapter = pick_adapter();
    if (!adapter) { fprintf(stderr, "ERROR: no DXGI adapter for the zero-copy pool\n"); delete p; return nullptr; }
    {
        DXGI_ADAPTER_DESC1 d{}; adapter->GetDesc1(&d);
        fprintf(stderr, "note: zero-copy pool on adapter %ls (LUID %08x%08x)\n", d.Description, (unsigned)d.AdapterLuid.HighPart, (unsigned)d.AdapterLuid.LowPart);
    }
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    const D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, want, 2, D3D11_SDK_VERSION, &p->device, &fl, &p->context);
    adapter->Release();
    if (FAILED(hr) || !p->device) { fprintf(stderr, "ERROR: D3D11CreateDevice failed (0x%08x)\n", (unsigned)hr); delete p; return nullptr; }
    D3D11_QUERY_DESC qd{ D3D11_QUERY_EVENT, 0 };
    p->device->CreateQuery(&qd, &p->fence);

    // Opaque black for the pad rows (written once, never again).
    std::vector<uint8_t> black((size_t)g.eye_w * g.eye_h * 4, 0);
    for (size_t i = 3; i < black.size(); i += 4) black[i] = 255;

    p->slots.resize(slots);
    const int eyes = g.mono ? 1 : 2;
    for (uint32_t s = 0; s < slots; s++) {
        for (int e = 0; e < eyes; e++) {
            D3D11_TEXTURE2D_DESC td{};
            td.Width = g.eye_w; td.Height = g.eye_h; td.MipLevels = 1; td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
            D3D11_SUBRESOURCE_DATA init{ black.data(), g.eye_w * 4, 0 };
            Eye& eye = p->slots[s].eye[e];
            hr = p->device->CreateTexture2D(&td, &init, &eye.texture);
            if (FAILED(hr) || !eye.texture) {
                fprintf(stderr, "ERROR: CreateTexture2D(%ux%u RGBA8, shared) failed for slot %u eye %d (0x%08x)\n", g.eye_w, g.eye_h, s, e, (unsigned)hr);
                zc_pool_destroy(p);
                return nullptr;
            }
            IDXGIResource1* res = nullptr;
            if (FAILED(eye.texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&res)) || !res
                || FAILED(res->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &eye.handle)) || !eye.handle) {
                if (res) res->Release();
                fprintf(stderr, "ERROR: CreateSharedHandle failed for slot %u eye %d\n", s, e);
                zc_pool_destroy(p);
                return nullptr;
            }
            res->Release();
        }
        if (g.mono) p->slots[s].eye[1] = p->slots[s].eye[0];   // same texture, shown to both eyes
    }
    p->wait_gpu_idle();
    return p;
}

void zc_pool_destroy(ZcPool* p) {
    if (!p) return;
    const int eyes = p->g.mono ? 1 : 2;
    for (auto& s : p->slots) {
        for (int e = 0; e < eyes; e++) {
            if (s.eye[e].handle) CloseHandle(s.eye[e].handle);
            if (s.eye[e].texture) s.eye[e].texture->Release();
        }
    }
    if (p->fence) p->fence->Release();
    if (p->context) p->context->Release();
    if (p->device) p->device->Release();
    delete p;
}

uint32_t zc_pool_surface_id(const ZcPool* p, uint32_t slot, int eye) {
    return (uint32_t)zc_pool_handle(p, slot, eye);
}

uint64_t zc_pool_handle(const ZcPool* p, uint32_t slot, int eye) {
    if (!p || slot >= p->slots.size()) return 0;
    return (uint64_t)(uintptr_t)p->slots[slot].eye[eye ? 1 : 0].handle;
}

bool zc_pool_fill_metal(ZcPool*, uint32_t, void*, void*, uint32_t, uint32_t, uint32_t, int) {
    // No GPU-side blit on Windows: the caller reads a GPU resource back
    // through the SDK and uses zc_pool_fill_cpu.
    return false;
}

bool zc_pool_fill_cpu(ZcPool* p, uint32_t slot, const uint8_t* src,
                      uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track) {
    if (!p || slot >= p->slots.size() || !src) return false;
    const RowMap m = row_map(p->g, src_h);
    if (m.rows == 0) return false;
    std::lock_guard<std::mutex> lk(p->mu);
    auto copy_eye = [&](int e, uint32_t x0) {
        const uint32_t w = std::min(p->g.eye_w, src_w - std::min(src_w, x0));
        if (w == 0) return;
        D3D11_BOX box{ 0, m.dst_y0, 0, w, m.dst_y0 + m.rows, 1 };
        p->context->UpdateSubresource(p->slots[slot].eye[e].texture, 0, &box,
            src + (size_t)m.src_y0 * src_row_bytes + (size_t)x0 * 4, src_row_bytes, 0);
    };
    if (p->g.mono) copy_eye(0, 0);
    else if (track < 0) { copy_eye(0, 0); copy_eye(1, p->g.eye_w); }
    else copy_eye(track ? 1 : 0, 0);
    // The app reads the texture through Vulkan as soon as it sees the "F"
    // line: the upload must be complete, not merely queued.
    p->wait_gpu_idle();
    return true;
}
