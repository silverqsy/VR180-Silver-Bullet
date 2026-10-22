// braw_zc.mm — see braw_zc.h. Objective-C++ (ARC): Metal + IOSurface.
#include "braw_zc.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Eye {
    IOSurfaceRef surface = nullptr;   // +1, global
    id<MTLTexture> texture = nil;     // aliases the surface (Metal device of the first fill)
    uint32_t id = 0;
};

struct Slot { Eye eye[2]; };

// kCVPixelFormatType_32RGBA = 'RGBA': R,G,B,A bytes in memory — the SDK's
// RGBAU8 layout, and the app's Rgba8Unorm texture order.
constexpr uint32_t kFormatRGBA = 0x52474241;

IOSurfaceRef create_surface(uint32_t w, uint32_t h) {
    NSDictionary* props = @{
        (id)kIOSurfaceWidth: @(w),
        (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @(4),
        (id)kIOSurfacePixelFormat: @(kFormatRGBA),
        // Global: the app looks the surface up by ID (IOSurfaceLookup). The
        // flag is deprecated but is the one cross-process route that needs
        // no mach-port plumbing; both ends are our own, unsandboxed binaries.
        (id)kIOSurfaceIsGlobal: @YES,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

// Opaque black (R,G,B,A = 0,0,0,255): one row written, then copied to the
// rest. The pad rows of a short eye are never written again.
void fill_black(IOSurfaceRef s) {
    IOSurfaceLock(s, 0, nullptr);
    uint8_t* base = (uint8_t*)IOSurfaceGetBaseAddress(s);
    const size_t bpr = IOSurfaceGetBytesPerRow(s);
    const size_t w = IOSurfaceGetWidth(s), h = IOSurfaceGetHeight(s);
    if (base && h > 0) {
        for (size_t x = 0; x < w; x++) { base[x * 4 + 0] = 0; base[x * 4 + 1] = 0; base[x * 4 + 2] = 0; base[x * 4 + 3] = 255; }
        for (size_t y = 1; y < h; y++) memcpy(base + y * bpr, base, w * 4);
    }
    IOSurfaceUnlock(s, 0, nullptr);
}

id<MTLTexture> texture_for(id<MTLDevice> device, IOSurfaceRef s) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:IOSurfaceGetWidth(s)
                                                                                 height:IOSurfaceGetHeight(s)
                                                                              mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [device newTextureWithDescriptor:d iosurface:s plane:0];
}

// Rows of the source that land in the canvas, and where.
struct RowMap { uint32_t src_y0, dst_y0, rows; };
RowMap row_map(const ZcGeometry& g, uint32_t src_h) {
    RowMap m{0, 0, 0};
    if (g.mono) { m.rows = std::min(src_h, g.eye_h); return m; }
    if (g.pad_rows >= 0) { m.dst_y0 = (uint32_t)g.pad_rows; m.rows = std::min(src_h, g.eye_h - m.dst_y0); }
    else { m.src_y0 = (uint32_t)(-g.pad_rows); m.rows = std::min(src_h - std::min(src_h, m.src_y0), g.eye_h); }
    return m;
}

} // namespace

struct ZcPool {
    ZcGeometry g;
    std::vector<Slot> slots;
    id<MTLDevice> device = nil;      // set by the first Metal fill
};

ZcPool* zc_pool_create(uint32_t slots, const ZcGeometry& g) {
    auto* p = new ZcPool();
    p->g = g;
    p->slots.resize(slots);
    const int eyes = g.mono ? 1 : 2;
    for (uint32_t s = 0; s < slots; s++) {
        for (int e = 0; e < eyes; e++) {
            IOSurfaceRef surf = create_surface(g.eye_w, g.eye_h);
            if (!surf) {
                fprintf(stderr, "ERROR: IOSurfaceCreate(%u×%u RGBA) failed for slot %u eye %d\n", g.eye_w, g.eye_h, s, e);
                zc_pool_destroy(p);
                return nullptr;
            }
            fill_black(surf);
            p->slots[s].eye[e].surface = surf;
            p->slots[s].eye[e].id = IOSurfaceGetID(surf);
        }
        if (g.mono) p->slots[s].eye[1] = p->slots[s].eye[0];   // same surface, shown to both eyes
    }
    return p;
}

void zc_pool_destroy(ZcPool* p) {
    if (!p) return;
    for (auto& s : p->slots) {
        const int eyes = p->g.mono ? 1 : 2;
        for (int e = 0; e < eyes; e++) {
            s.eye[e].texture = nil;
            if (s.eye[e].surface) CFRelease(s.eye[e].surface);
        }
    }
    delete p;
}

uint32_t zc_pool_surface_id(const ZcPool* p, uint32_t slot, int eye) {
    if (!p || slot >= p->slots.size()) return 0;
    return p->slots[slot].eye[eye ? 1 : 0].id;
}

bool zc_pool_fill_metal(ZcPool* p, uint32_t slot, void* mtl_buffer, void* mtl_queue,
                        uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track) {
    if (!p || slot >= p->slots.size() || !mtl_buffer || !mtl_queue) return false;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)mtl_buffer;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtl_queue;
        if (p->device == nil) p->device = queue.device;
        const int eyes = p->g.mono ? 1 : 2;
        for (int e = 0; e < eyes; e++) {
            Eye& eye = p->slots[slot].eye[e];
            if (eye.texture == nil) {
                eye.texture = texture_for(p->device, eye.surface);
                if (eye.texture == nil) { fprintf(stderr, "ERROR: Metal texture for IOSurface %u failed\n", eye.id); return false; }
            }
        }
        const RowMap m = row_map(p->g, src_h);
        if (m.rows == 0) return false;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        auto copy_eye = [&](int e, uint32_t x0) {
            const uint32_t w = std::min(p->g.eye_w, src_w - std::min(src_w, x0));
            if (w == 0) return;
            [blit copyFromBuffer:buf
                    sourceOffset:(NSUInteger)m.src_y0 * src_row_bytes + (NSUInteger)x0 * 4
               sourceBytesPerRow:src_row_bytes
             sourceBytesPerImage:(NSUInteger)m.rows * src_row_bytes
                      sourceSize:MTLSizeMake(w, m.rows, 1)
                       toTexture:p->slots[slot].eye[e].texture
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, m.dst_y0, 0)];
        };
        if (p->g.mono) copy_eye(0, 0);
        else if (track < 0) { copy_eye(0, 0); copy_eye(1, p->g.eye_w); }
        else copy_eye(track ? 1 : 0, 0);
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "ERROR: zero-copy blit failed: %s\n", cb.error ? cb.error.localizedDescription.UTF8String : "?");
            return false;
        }
    }
    return true;
}

bool zc_pool_fill_cpu(ZcPool* p, uint32_t slot, const uint8_t* src,
                      uint32_t src_w, uint32_t src_h, uint32_t src_row_bytes, int track) {
    if (!p || slot >= p->slots.size() || !src) return false;
    const RowMap m = row_map(p->g, src_h);
    if (m.rows == 0) return false;
    auto copy_eye = [&](int e, uint32_t x0) {
        const uint32_t w = std::min(p->g.eye_w, src_w - std::min(src_w, x0));
        if (w == 0) return;
        IOSurfaceRef s = p->slots[slot].eye[e].surface;
        IOSurfaceLock(s, 0, nullptr);
        uint8_t* base = (uint8_t*)IOSurfaceGetBaseAddress(s);
        const size_t bpr = IOSurfaceGetBytesPerRow(s);
        for (uint32_t y = 0; y < m.rows; y++)
            memcpy(base + (size_t)(m.dst_y0 + y) * bpr,
                   src + (size_t)(m.src_y0 + y) * src_row_bytes + (size_t)x0 * 4, (size_t)w * 4);
        IOSurfaceUnlock(s, 0, nullptr);
    };
    if (p->g.mono) copy_eye(0, 0);
    else if (track < 0) { copy_eye(0, 0); copy_eye(1, p->g.eye_w); }
    else copy_eye(track ? 1 : 0, 0);
    return true;
}
