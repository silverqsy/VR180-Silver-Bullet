// braw_zc.h — zero-copy frame hand-off for braw_helper.
//
// macOS (braw_zc.mm): global IOSurfaces. Windows (braw_zc_win.cpp): shareable
// D3D11 textures (NT handles the app duplicates and imports into Vulkan).
// The protocol is the same on both; `zc_transport()` names the kind.
//
// A pool of IOSurfaces, one or two per slot, each already laid out the way
// the app's kernels want an eye: a SQUARE canvas of the eye's width (the
// 2:1 timeline — a shorter recorded eye is padded black top and bottom, a
// taller one cropped), RGBA8. A decoded frame is copied into a slot on the
// GPU (a Metal blit from the SDK's MTLBuffer) or by the CPU (a memcpy when
// the SDK decodes on the CPU), and only the slot number crosses the pipe;
// the app maps the same surfaces (they are global) as textures. Slots are
// handed back by the app ("R <slot>" on stdin) once its GPU is done.
#pragma once
#include <cstdint>

struct ZcPool;

// Geometry of one eye slot. `eye_w × eye_h` is the canvas (a square for a
// stereo pair; the recorded frame for a mono clip); `pad_rows` = rows the
// canvas adds above the recorded eye (negative: rows cropped off the top).
struct ZcGeometry {
    uint32_t eye_w = 0, eye_h = 0;
    int32_t pad_rows = 0;
    bool mono = false;        // one surface per slot, shown to both eyes
};

// Create `slots` slots (2 global IOSurfaces each, 1 for mono), zero-filled
// to opaque black. Returns null (and prints the reason to stderr) when an
// IOSurface or its Metal texture cannot be created.
ZcPool* zc_pool_create(uint32_t slots, const ZcGeometry& g);
void zc_pool_destroy(ZcPool* p);

// The global IOSurface ID of slot `slot`, eye 0 (left) / 1 (right).
uint32_t zc_pool_surface_id(const ZcPool* p, uint32_t slot, int eye);
// The slot/eye's share token as the helper prints it: the IOSurface ID on
// macOS, the NT handle value on Windows (valid in `zc_owner_pid()`).
uint64_t zc_pool_handle(const ZcPool* p, uint32_t slot, int eye);
// "iosurface" or "d3d11".
const char* zc_transport();
// The helper's own process id (the app duplicates handles out of it).
uint32_t zc_owner_pid();
// Windows: the DXGI adapter LUID the pool must live on (the app's GPU);
// ignored on macOS.
void zc_set_adapter_luid(uint64_t luid);

// Fill a slot from a processed frame.
//   `src_w × src_h` is the frame as processed; `src_row_bytes` its pitch
//   (RGBA8). `track` −1: the frame holds both eyes side by side (eye k =
//   columns [k·eye_w, (k+1)·eye_w)); 0 / 1: the frame is ONE track that
//   goes into eye `track` (dual-track clips). Mono pools take the whole
//   frame into their single surface.
// The Metal variant blits with a command buffer on `mtl_queue` (the SDK's
// queue, so the blit is ordered after the SDK's processing) and waits for
// it; the CPU variant copies rows under an IOSurface lock.
bool zc_pool_fill_metal(ZcPool* p, uint32_t slot, void* mtl_buffer, void* mtl_queue,
                        uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track);
bool zc_pool_fill_cpu(ZcPool* p, uint32_t slot, const uint8_t* src,
                      uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track);
