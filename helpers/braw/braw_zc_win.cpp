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
// Fills:
//  * GPU (the SDK's CUDA pipeline): the processed image is a CUdeviceptr.
//    CUDA's D3D11 interop refuses SHARED textures, so the pool keeps one
//    non-shared STAGING texture per eye, registered with CUDA
//    (cuGraphicsD3D11RegisterResource); the frame is copied device→array
//    into it (cuMemcpy2DAsync) and D3D11 copies staging→slot
//    (CopyResource) — two GPU copies, no host round trip. The CUDA driver
//    is loaded at runtime (nvcuda.dll), so the helper still runs where there
//    is no NVIDIA driver; the staging/registration is set up on the first
//    GPU fill and needs the SDK's CUDA device to be the pool's adapter.
//  * CPU (the SDK's CPU pipeline, or a GPU frame read back when the blit is
//    unavailable): UpdateSubresource.
// Both flush and fence before returning: the app reads the texture through
// Vulkan the moment it sees the "F" line.
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

#if BRAW_ZC_CUDA
#include <cuda.h>
#include <cudaD3D11.h>
#endif

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

uint64_t luid_u64(const LUID& l) { return ((uint64_t)(uint32_t)l.HighPart << 32) | (uint64_t)l.LowPart; }

IDXGIAdapter1* pick_adapter(uint64_t* chosenLuid) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)) || !factory) return nullptr;
    IDXGIAdapter1* chosen = nullptr;
    IDXGIAdapter1* firstHw = nullptr;
    uint64_t firstLuid = 0;
    for (UINT i = 0;; i++) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) != S_OK || !a) break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        const bool software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        const uint64_t luid = luid_u64(d.AdapterLuid);
        if (g_adapterLuid != 0 && luid == g_adapterLuid) { chosen = a; *chosenLuid = luid; break; }
        if (!software && !firstHw) { firstHw = a; firstLuid = luid; continue; }
        a->Release();
    }
    factory->Release();
    if (chosen) { if (firstHw) firstHw->Release(); return chosen; }
    if (g_adapterLuid != 0) fprintf(stderr, "note: --adapter-luid %016llx not found; using the first hardware adapter\n", (unsigned long long)g_adapterLuid);
    *chosenLuid = firstLuid;
    return firstHw;
}

#if BRAW_ZC_CUDA
// The CUDA driver API, resolved at runtime (no link-time dependency, so the
// helper loads on machines without an NVIDIA driver).
struct CudaFns {
    HMODULE lib = nullptr;
    CUresult (CUDAAPI *CtxSetCurrent)(CUcontext) = nullptr;
    CUresult (CUDAAPI *CtxGetDevice)(CUdevice*) = nullptr;
    CUresult (CUDAAPI *DeviceGetLuid)(char*, unsigned int*, CUdevice) = nullptr;
    CUresult (CUDAAPI *GraphicsD3D11RegisterResource)(CUgraphicsResource*, ID3D11Resource*, unsigned int) = nullptr;
    CUresult (CUDAAPI *GraphicsUnregisterResource)(CUgraphicsResource) = nullptr;
    CUresult (CUDAAPI *GraphicsMapResources)(unsigned int, CUgraphicsResource*, CUstream) = nullptr;
    CUresult (CUDAAPI *GraphicsUnmapResources)(unsigned int, CUgraphicsResource*, CUstream) = nullptr;
    CUresult (CUDAAPI *GraphicsSubResourceGetMappedArray)(CUarray*, CUgraphicsResource, unsigned int, unsigned int) = nullptr;
    CUresult (CUDAAPI *Memcpy2DAsync)(const CUDA_MEMCPY2D*, CUstream) = nullptr;
    CUresult (CUDAAPI *StreamSynchronize)(CUstream) = nullptr;
    CUresult (CUDAAPI *EventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (CUDAAPI *EventRecord)(CUevent, CUstream) = nullptr;
    CUresult (CUDAAPI *EventSynchronize)(CUevent) = nullptr;
    CUresult (CUDAAPI *EventDestroy)(CUevent) = nullptr;
    CUresult (CUDAAPI *GetErrorString)(CUresult, const char**) = nullptr;
    bool ok = false;

    bool load() {
        lib = LoadLibraryW(L"nvcuda.dll");
        if (!lib) return false;
        auto get = [&](const char* name) -> void* { return (void*)GetProcAddress(lib, name); };
        CtxSetCurrent = (decltype(CtxSetCurrent))get("cuCtxSetCurrent");
        CtxGetDevice = (decltype(CtxGetDevice))get("cuCtxGetDevice");
        DeviceGetLuid = (decltype(DeviceGetLuid))get("cuDeviceGetLuid");
        GraphicsD3D11RegisterResource = (decltype(GraphicsD3D11RegisterResource))get("cuGraphicsD3D11RegisterResource");
        GraphicsUnregisterResource = (decltype(GraphicsUnregisterResource))get("cuGraphicsUnregisterResource");
        GraphicsMapResources = (decltype(GraphicsMapResources))get("cuGraphicsMapResources");
        GraphicsUnmapResources = (decltype(GraphicsUnmapResources))get("cuGraphicsUnmapResources");
        GraphicsSubResourceGetMappedArray = (decltype(GraphicsSubResourceGetMappedArray))get("cuGraphicsSubResourceGetMappedArray");
        Memcpy2DAsync = (decltype(Memcpy2DAsync))get("cuMemcpy2DAsync_v2");
        StreamSynchronize = (decltype(StreamSynchronize))get("cuStreamSynchronize");
        EventCreate = (decltype(EventCreate))get("cuEventCreate");
        EventRecord = (decltype(EventRecord))get("cuEventRecord");
        EventSynchronize = (decltype(EventSynchronize))get("cuEventSynchronize");
        EventDestroy = (decltype(EventDestroy))get("cuEventDestroy_v2");
        GetErrorString = (decltype(GetErrorString))get("cuGetErrorString");
        ok = CtxSetCurrent && CtxGetDevice && DeviceGetLuid && GraphicsD3D11RegisterResource && GraphicsUnregisterResource
            && GraphicsMapResources && GraphicsUnmapResources && GraphicsSubResourceGetMappedArray && Memcpy2DAsync && StreamSynchronize
            && EventCreate && EventRecord && EventSynchronize && EventDestroy;
        return ok;
    }
    const char* err(CUresult r) { const char* s = "?"; if (GetErrorString) GetErrorString(r, &s); return s; }
};
#endif

} // namespace

struct ZcPool {
    ZcGeometry g;
    std::vector<Slot> slots;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11Query* fence = nullptr;
    uint64_t adapterLuid = 0;
    std::mutex mu;                  // the immediate context is not thread-safe
#if BRAW_ZC_CUDA
    CudaFns cu;
    // Non-shared staging textures the CUDA copies land in (one per eye,
    // reused for every slot), registered with CUDA on the first GPU fill.
    ID3D11Texture2D* staging[2] = { nullptr, nullptr };
    CUgraphicsResource cuStaging[2] = { nullptr, nullptr };
    // Recorded on the SDK's stream right after our copies: waiting on it
    // covers our work only, not the frames the SDK has queued behind it
    // (cuStreamSynchronize would, and collapsed the SDK's pipeline).
    CUevent copied = nullptr;
    enum { BlitUntried, BlitReady, BlitUnavailable } blit = BlitUntried;
    // BRAW_HELPER_DEBUG: per-stage wall clock of the blit, every 30 fills.
    bool debug = getenv("BRAW_HELPER_DEBUG") != nullptr;
    uint64_t fills = 0; double t_map = 0, t_copy = 0, t_wait = 0, t_d3d = 0;
#endif

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

static ID3D11Texture2D* make_texture(ID3D11Device* dev, uint32_t w, uint32_t h, const std::vector<uint8_t>& black, bool shared) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = shared ? (D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED) : 0;
    D3D11_SUBRESOURCE_DATA init{ black.data(), w * 4, 0 };
    ID3D11Texture2D* t = nullptr;
    HRESULT hr = dev->CreateTexture2D(&td, &init, &t);
    if (FAILED(hr) || !t) {
        fprintf(stderr, "ERROR: CreateTexture2D(%ux%u RGBA8%s) failed (0x%08x)\n", w, h, shared ? ", shared" : "", (unsigned)hr);
        return nullptr;
    }
    return t;
}

ZcPool* zc_pool_create(uint32_t slots, const ZcGeometry& g) {
    auto* p = new ZcPool();
    p->g = g;
    IDXGIAdapter1* adapter = pick_adapter(&p->adapterLuid);
    if (!adapter) { fprintf(stderr, "ERROR: no DXGI adapter for the zero-copy pool\n"); delete p; return nullptr; }
    {
        DXGI_ADAPTER_DESC1 d{}; adapter->GetDesc1(&d);
        fprintf(stderr, "note: zero-copy pool on adapter %ls (LUID %016llx)\n", d.Description, (unsigned long long)p->adapterLuid);
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
            Eye& eye = p->slots[s].eye[e];
            eye.texture = make_texture(p->device, g.eye_w, g.eye_h, black, true);
            if (!eye.texture) { zc_pool_destroy(p); return nullptr; }
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
#if BRAW_ZC_CUDA
    for (int e = 0; e < eyes; e++) {
        p->staging[e] = make_texture(p->device, g.eye_w, g.eye_h, black, false);
        if (!p->staging[e]) { zc_pool_destroy(p); return nullptr; }
    }
#endif
    p->wait_gpu_idle();
    return p;
}

void zc_pool_destroy(ZcPool* p) {
    if (!p) return;
    const int eyes = p->g.mono ? 1 : 2;
#if BRAW_ZC_CUDA
    for (int e = 0; e < eyes; e++) {
        if (p->cuStaging[e] && p->cu.ok) p->cu.GraphicsUnregisterResource(p->cuStaging[e]);
        if (p->staging[e]) p->staging[e]->Release();
    }
    if (p->copied && p->cu.ok) p->cu.EventDestroy(p->copied);
    if (p->cu.lib) FreeLibrary(p->cu.lib);
#endif
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
    return false;   // macOS only; Windows GPU frames go through zc_pool_fill_gpu
}

#if BRAW_ZC_CUDA
// First GPU fill: load the driver, check the SDK's CUDA device is the pool's
// adapter, register the staging textures. Any failure → host fallback.
static bool blit_setup(ZcPool* p, CUcontext ctx) {
    if (p->blit != ZcPool::BlitUntried) return p->blit == ZcPool::BlitReady;
    p->blit = ZcPool::BlitUnavailable;
    if (!p->cu.load()) { fprintf(stderr, "note: CUDA driver (nvcuda.dll) not available; zero-copy frames go through host memory\n"); return false; }
    CUresult r = p->cu.CtxSetCurrent(ctx);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "note: cuCtxSetCurrent failed (%s); zero-copy frames go through host memory\n", p->cu.err(r)); return false; }
    CUdevice dev = 0; char luid[8] = {}; unsigned mask = 0;
    if (p->cu.CtxGetDevice(&dev) != CUDA_SUCCESS || p->cu.DeviceGetLuid(luid, &mask, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "note: CUDA device LUID unavailable; zero-copy frames go through host memory\n"); return false;
    }
    uint64_t cuLuid = 0; memcpy(&cuLuid, luid, 8);
    if (cuLuid != p->adapterLuid) {
        fprintf(stderr, "note: the SDK decodes on CUDA device LUID %016llx but the pool is on %016llx; zero-copy frames go through host memory\n",
                (unsigned long long)cuLuid, (unsigned long long)p->adapterLuid);
        return false;
    }
    const int eyes = p->g.mono ? 1 : 2;
    for (int e = 0; e < eyes; e++) {
        r = p->cu.GraphicsD3D11RegisterResource(&p->cuStaging[e], p->staging[e], 0 /* CU_GRAPHICS_REGISTER_FLAGS_NONE */);
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "note: cuGraphicsD3D11RegisterResource failed (%s); zero-copy frames go through host memory\n", p->cu.err(r));
            for (int k = 0; k < e; k++) { p->cu.GraphicsUnregisterResource(p->cuStaging[k]); p->cuStaging[k] = nullptr; }
            p->cuStaging[e] = nullptr;
            return false;
        }
    }
    r = p->cu.EventCreate(&p->copied, 0x2 /* CU_EVENT_DISABLE_TIMING */);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "note: cuEventCreate failed (%s); zero-copy frames go through host memory\n", p->cu.err(r));
        for (int k = 0; k < eyes; k++) { p->cu.GraphicsUnregisterResource(p->cuStaging[k]); p->cuStaging[k] = nullptr; }
        return false;
    }
    fprintf(stderr, "note: zero-copy GPU blit ready (CUDA → D3D11 staging → shared slot)\n");
    p->blit = ZcPool::BlitReady;
    return true;
}

static double now_ms() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#endif

bool zc_pool_fill_gpu(ZcPool* p, uint32_t slot, void* resource, void* context, void* queue,
                      uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track) {
#if BRAW_ZC_CUDA
    if (!p || slot >= p->slots.size() || !resource || !context) return false;
    std::lock_guard<std::mutex> lk(p->mu);
    if (!blit_setup(p, (CUcontext)context)) return false;
    const RowMap m = row_map(p->g, src_h);
    if (m.rows == 0) return false;
    CUstream stream = (CUstream)queue;
    p->cu.CtxSetCurrent((CUcontext)context);
    const int eyes = p->g.mono ? 1 : 2;
    const double t0 = p->debug ? now_ms() : 0.0;
    CUresult r = p->cu.GraphicsMapResources((unsigned)eyes, p->cuStaging, stream);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "ERROR: cuGraphicsMapResources: %s\n", p->cu.err(r)); return false; }
    const double t1 = p->debug ? now_ms() : 0.0;
    bool ok = true;
    auto copy_eye = [&](int e, uint32_t x0) {
        const uint32_t w = std::min(p->g.eye_w, src_w - std::min(src_w, x0));
        if (w == 0 || !ok) return;
        CUarray arr = nullptr;
        CUresult rr = p->cu.GraphicsSubResourceGetMappedArray(&arr, p->cuStaging[e], 0, 0);
        if (rr != CUDA_SUCCESS || !arr) { fprintf(stderr, "ERROR: cuGraphicsSubResourceGetMappedArray: %s\n", p->cu.err(rr)); ok = false; return; }
        CUDA_MEMCPY2D c{};
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice = (CUdeviceptr)(uintptr_t)resource + (size_t)m.src_y0 * src_row_bytes + (size_t)x0 * 4;
        c.srcPitch = src_row_bytes;
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        c.dstArray = arr;
        c.dstY = m.dst_y0;
        c.WidthInBytes = (size_t)w * 4;
        c.Height = m.rows;
        rr = p->cu.Memcpy2DAsync(&c, stream);
        if (rr != CUDA_SUCCESS) { fprintf(stderr, "ERROR: cuMemcpy2DAsync into the staging texture: %s\n", p->cu.err(rr)); ok = false; }
    };
    if (p->g.mono) copy_eye(0, 0);
    else if (track < 0) { copy_eye(0, 0); copy_eye(1, p->g.eye_w); }
    else copy_eye(track ? 1 : 0, 0);
    r = p->cu.GraphicsUnmapResources((unsigned)eyes, p->cuStaging, stream);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "ERROR: cuGraphicsUnmapResources: %s\n", p->cu.err(r)); ok = false; }
    const double t2 = p->debug ? now_ms() : 0.0;
    // Wait for OUR copies (and the unmap), not for whatever the SDK queued
    // on this stream after them.
    if (ok && (p->cu.EventRecord(p->copied, stream) != CUDA_SUCCESS || (r = p->cu.EventSynchronize(p->copied)) != CUDA_SUCCESS)) {
        fprintf(stderr, "ERROR: waiting for the zero-copy blit: %s\n", p->cu.err(r)); ok = false;
    }
    if (!ok) return false;
    const double t3 = p->debug ? now_ms() : 0.0;
    // Staging → the shared slot (the pad rows stay black in both), fenced.
    if (p->g.mono) p->context->CopyResource(p->slots[slot].eye[0].texture, p->staging[0]);
    else if (track < 0) { p->context->CopyResource(p->slots[slot].eye[0].texture, p->staging[0]); p->context->CopyResource(p->slots[slot].eye[1].texture, p->staging[1]); }
    else { const int e = track ? 1 : 0; p->context->CopyResource(p->slots[slot].eye[e].texture, p->staging[e]); }
    p->wait_gpu_idle();
    if (p->debug) {
        const double t4 = now_ms();
        p->t_map += t1 - t0; p->t_copy += t2 - t1; p->t_wait += t3 - t2; p->t_d3d += t4 - t3;
        if (++p->fills % 30 == 0) {
            fprintf(stderr, "note: zc blit per fill (avg of 30): map %.2f ms, copy-enqueue %.2f, wait %.2f, d3d copy+fence %.2f\n",
                    p->t_map / 30, p->t_copy / 30, p->t_wait / 30, p->t_d3d / 30);
            p->t_map = p->t_copy = p->t_wait = p->t_d3d = 0;
        }
    }
    return true;
#else
    (void)p; (void)slot; (void)resource; (void)context; (void)queue; (void)src_w; (void)src_h; (void)src_row_bytes; (void)track;
    return false;   // built without the CUDA headers: host fallback
#endif
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
