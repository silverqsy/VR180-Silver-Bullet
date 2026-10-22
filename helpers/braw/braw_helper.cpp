/*
 * braw_helper.cpp — Blackmagic RAW decode helper for vr180_fisheye_converter
 *
 * Modes:
 *   --info  <file.braw>                                          → JSON clip metadata to stdout
 *   --decode <file.braw> [--start N] [--count M] [--track T]     → raw RGBA frames to stdout (older builds: BGRA)
 *            [--16bit] [--scale full|half|quarter|eighth]          (JSON header on stderr first)
 *            [--look film|video|extended|rec709] [--gamma S] [--gamut S] [--wb K] ...
 *            [--pipeline auto|metal|cpu]  (auto: the first Metal device, else the CPU;
 *             the header reports "pipeline" + "gpu_device"; frames are pipelined 2–3 deep)
 *            [--step N]  (every Nth frame; "frames" in the header = the emitted count, "format": "rgba")
 *   --gyro  <file.braw>                                          → binary gyro+accel to stdout, JSON header to stderr
 *   --audio <file.braw>                                          → WAV to stdout, JSON header to stderr
 *   --meta  <file.braw> [--frame N]                              → every clip + frame metadata key, one per line
 *   --shutters <file.braw> [--step N]                            → "<frame> <seconds> \"<shutter_value>\"" per frame
 *   --lut   <file.braw> <out.cube>                               → the embedded post-3D LUT as a .cube file
 *
 * Multi-video (e.g. URSA Cine Immersive stereo):
 *   --info reports video_track_count > 1 and per-track dimensions
 *   --decode with --track 0 or --track 1 decodes a single track
 *   --decode without --track on a multi-video clip decodes all tracks,
 *     writing them side-by-side (left=track0, right=track1) per frame
 *
 * --scale asks the SDK for a reduced decode (the BRAW bitstream is
 * partially decodable, so half/quarter/eighth are much cheaper than a
 * full decode + downscale). The decode header always reports the
 * processed image size, i.e. the size of the frames actually written.
 *
 * Build:
 *   clang++ -std=c++17 -O2 braw_helper.cpp \
 *     -I"/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Include" \
 *     -F"/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Libraries" \
 *     -rpath "/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Libraries" \
 *     -framework BlackmagicRawAPI -framework CoreFoundation -framework CoreServices \
 *     -include "/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Include/BlackmagicRawAPIDispatch.cpp" \
 *     -o braw_helper
 */

#include "BlackmagicRawAPI.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <atomic>
#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include <condition_variable>
#include <thread>
#include "braw_zc.h"

#include <CoreServices/CoreServices.h>

#ifdef DEBUG
    #include <cassert>
    #define VERIFY(condition) assert(SUCCEEDED(condition))
#else
    #define VERIFY(condition) condition
#endif

static const char* SDK_LIB_PATH = "/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Libraries";

// ── Utility ──────────────────────────────────────────────────────────────────

static std::string cfstr_to_string(CFStringRef cfStr) {
    if (!cfStr) return "";
    const char* cStr = CFStringGetCStringPtr(cfStr, kCFStringEncodingUTF8);
    if (cStr) return cStr;
    char buf[2048];
    if (CFStringGetCString(cfStr, buf, sizeof(buf), kCFStringEncodingUTF8))
        return buf;
    return "";
}

// Minimal JSON string escaping for the values we print (titles, names).
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", (unsigned)(unsigned char)c); out += b; }
        else out += c;
    }
    return out;
}

static std::string get_metadata_string(IBlackmagicRawClip* clip, const char* key) {
    Variant v;
    VariantInit(&v);
    CFStringRef cfKey = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
    HRESULT result = clip->GetMetadata(cfKey, &v);
    CFRelease(cfKey);
    if (result != S_OK) return "";
    std::string s;
    if (v.vt == blackmagicRawVariantTypeString)
        s = cfstr_to_string(v.bstrVal);
    VariantClear(&v);
    return s;
}

static double get_metadata_float(IBlackmagicRawClip* clip, const char* key, double def = 0.0) {
    Variant v;
    VariantInit(&v);
    CFStringRef cfKey = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
    HRESULT result = clip->GetMetadata(cfKey, &v);
    CFRelease(cfKey);
    if (result != S_OK) return def;
    double val = def;
    if (v.vt == blackmagicRawVariantTypeFloat32) val = v.fltVal;
    else if (v.vt == blackmagicRawVariantTypeFloat64) val = v.dblVal;
    else if (v.vt == blackmagicRawVariantTypeU8) val = (double)(v.uiVal & 0xff);
    else if (v.vt == blackmagicRawVariantTypeS16) val = (double)v.iVal;
    else if (v.vt == blackmagicRawVariantTypeU16) val = (double)v.uiVal;
    else if (v.vt == blackmagicRawVariantTypeS32) val = (double)v.intVal;
    else if (v.vt == blackmagicRawVariantTypeU32) val = (double)v.uintVal;
    VariantClear(&v);
    return val;
}

// A 2-element numeric array ([w, h] / [x, y]) → both values, 0 if absent.
static bool get_metadata_pair(IBlackmagicRawClip* clip, const char* key, double* a, double* b) {
    Variant v; VariantInit(&v);
    CFStringRef cfKey = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
    HRESULT result = clip->GetMetadata(cfKey, &v); CFRelease(cfKey);
    bool ok = false;
    if (result == S_OK && v.vt == blackmagicRawVariantTypeSafeArray && v.parray) {
        BlackmagicRawVariantType et; long lo = 0, hi = -1; void* data = nullptr;
        SafeArrayGetVartype(v.parray, &et); SafeArrayGetLBound(v.parray, 1, &lo); SafeArrayGetUBound(v.parray, 1, &hi); SafeArrayAccessData(v.parray, &data);
        if (data && hi - lo >= 1) {
            auto at = [&](long k) -> double {
                switch (et) {
                    case blackmagicRawVariantTypeFloat32: return ((float*)data)[k];
                    case blackmagicRawVariantTypeFloat64: return ((double*)data)[k];
                    case blackmagicRawVariantTypeU16: return ((uint16_t*)data)[k];
                    case blackmagicRawVariantTypeS16: return ((int16_t*)data)[k];
                    case blackmagicRawVariantTypeU32: return ((uint32_t*)data)[k];
                    case blackmagicRawVariantTypeS32: return ((int32_t*)data)[k];
                    default: return 0.0;
                }
            };
            *a = at(0); *b = at(1); ok = true;
        }
    }
    VariantClear(&v);
    return ok;
}

// Per-frame shutter: the camera writes `shutter_value` as "7.2°" (angle) or
// "1/50" (speed, when shutter_type is "Speed"); returned in seconds, 0 if absent.
struct FrameShutter { double seconds = 0; std::string text; };

static FrameShutter parse_shutter_text(const std::string& t, double fps) {
    FrameShutter fs; fs.text = t;
    if (!t.empty() && t[0] == '1' && t.find('/') != std::string::npos) {
        double d = atof(t.c_str() + t.find('/') + 1); if (d > 0) fs.seconds = 1.0 / d;
    } else {
        double deg = atof(t.c_str()); if (deg > 0 && fps > 0) fs.seconds = deg / 360.0 / fps;
    }
    return fs;
}

// Read-job callback that only looks at frame metadata (no decode). Works
// with one job at a time (`done` + `out`, used by --info) and with many
// read jobs in flight at once (`results` keyed by frame index, used by
// --shutters; ReadComplete may run on SDK threads concurrently).
class ShutterCallback : public IBlackmagicRawCallback {
public:
    std::atomic<bool> done{false};
    FrameShutter out;                          // the most recent completed frame
    std::map<uint64_t, FrameShutter> results;  // every completed frame, by index
    std::mutex mutex;
    double fps = 0;
    void ReadComplete(IBlackmagicRawJob* readJob, HRESULT result, IBlackmagicRawFrame* frame) override {
        FrameShutter fs; uint64_t idx = 0; bool haveIdx = false;
        if (result == S_OK && frame) {
            haveIdx = frame->GetFrameIndex(&idx) == S_OK;
            Variant v; VariantInit(&v);
            CFStringRef k = CFStringCreateWithCString(NULL, "shutter_value", kCFStringEncodingUTF8);
            if (frame->GetMetadata(k, &v) == S_OK && v.vt == blackmagicRawVariantTypeString)
                fs = parse_shutter_text(cfstr_to_string(v.bstrVal), fps);
            VariantClear(&v); CFRelease(k);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            out = fs;
            if (haveIdx) results[idx] = fs;
        }
        done.store(true);
        readJob->Release();
    }
    void ReadAudioComplete(IBlackmagicRawJob* job, HRESULT, IBlackmagicRawAudioBuffer*) override { if (job) job->Release(); }
    void DecodeComplete(IBlackmagicRawJob*, HRESULT) override {}
    void ProcessComplete(IBlackmagicRawJob*, HRESULT, IBlackmagicRawProcessedImage*) override {}
    void TrimProgress(IBlackmagicRawJob*, float) override {}
    void TrimComplete(IBlackmagicRawJob*, HRESULT) override {}
    void SidecarMetadataParseWarning(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void SidecarMetadataParseError(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void PreparePipelineComplete(void*, HRESULT) override {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID*) override { return E_NOTIMPL; }
    ULONG STDMETHODCALLTYPE AddRef() override { return 0; }
    ULONG STDMETHODCALLTYPE Release() override { return 0; }
};

// ── BRAW processing overrides ────────────────────────────────────────────────

struct BrawProcessingParams {
    // Clip-level
    std::string gamma;      // e.g. "Rec.709", "Blackmagic Design Film", ""=default
    std::string gamut;      // e.g. "Rec.709", "Blackmagic Design", ""=default
    std::string postLutMode; // "Disabled" / "Preview" / "Export", ""=as shot (set by --look)
    float toneCurveContrast = -1;   // -1 = don't override
    float toneCurveSaturation = -1;
    float toneCurveMidpoint = -1;
    float toneCurveHighlights = -1;
    float toneCurveShadows = -1;
    float toneCurveBlackLevel = -1;
    float toneCurveWhiteLevel = -1;
    int highlightRecovery = -1;     // 0 or 1, -1 = don't override
    float analogGain = -1;          // -1 = don't override
    int gamutCompression = -1;      // 0 or 1, -1 = don't override
    // Frame-level
    int whiteBalanceKelvin = -1;    // -1 = don't override
    int whiteBalanceTint = -9999;   // -9999 = don't override
    float exposure = -9999;         // -9999 = don't override
    int iso = -1;                   // -1 = don't override
};

static BrawProcessingParams g_procParams;

// --look presets: the gamma/gamut pairs the camera's own viewing modes use.
// "film" leaves gamma/gamut as shot. Every look disables the post-3D LUT
// so the result is the bare colour-science transform.
static bool apply_look_preset(const std::string& look) {
    if (look == "film")          { g_procParams.gamma = ""; g_procParams.gamut = ""; }
    else if (look == "video")    { g_procParams.gamma = "Blackmagic Design Video"; g_procParams.gamut = "Blackmagic Design"; }
    else if (look == "extended") { g_procParams.gamma = "Blackmagic Design Extended Video"; g_procParams.gamut = "Blackmagic Design"; }
    else if (look == "rec709")   { g_procParams.gamma = "Rec.709"; g_procParams.gamut = "Rec.709"; }
    else return false;
    g_procParams.postLutMode = "Disabled";
    return true;
}

static std::string variant_repr(const Variant& v);  // defined with --meta below

static const char* attr_name(BlackmagicRawClipProcessingAttribute attr) {
    switch (attr) {
        case blackmagicRawClipProcessingAttributeGamma: return "gamma";
        case blackmagicRawClipProcessingAttributeGamut: return "gamut";
        case blackmagicRawClipProcessingAttributePost3DLUTMode: return "post_3dlut_mode";
        default: return "attribute";
    }
}

// Print the SDK's accepted values for a list-type clip attribute to stderr
// (used when a string override is rejected, so the caller sees what the
// SDK actually wants).
static void print_attribute_list(IBlackmagicRawClipProcessingAttributes* attrs, BlackmagicRawClipProcessingAttribute attr) {
    uint32_t count = 0; bool readOnly = false;
    if (attrs->GetClipAttributeList(attr, nullptr, &count, &readOnly) != S_OK || count == 0) {
        fprintf(stderr, "  %s: no value list available\n", attr_name(attr));
        return;
    }
    std::vector<Variant> items(count);
    for (auto& v : items) VariantInit(&v);
    if (attrs->GetClipAttributeList(attr, items.data(), &count, &readOnly) != S_OK) {
        fprintf(stderr, "  %s: value list unreadable\n", attr_name(attr));
        return;
    }
    fprintf(stderr, "  %s accepts (%u%s):", attr_name(attr), count, readOnly ? ", read-only" : "");
    for (uint32_t i = 0; i < count; i++) {
        if (items[i].vt == blackmagicRawVariantTypeString) fprintf(stderr, " \"%s\"", cfstr_to_string(items[i].bstrVal).c_str());
        else fprintf(stderr, " <%s>", variant_repr(items[i]).c_str());
        VariantClear(&items[i]);
    }
    fprintf(stderr, "\n");
}

// Set a string clip attribute; on rejection report the accepted list.
static bool set_clip_string_attr(IBlackmagicRawClipProcessingAttributes* attrs, BlackmagicRawClipProcessingAttribute attr, const std::string& val) {
    if (val.empty()) return true;
    CFStringRef cf = CFStringCreateWithCString(NULL, val.c_str(), kCFStringEncodingUTF8);
    Variant v; VariantInit(&v); v.vt = blackmagicRawVariantTypeString; v.bstrVal = cf;
    HRESULT hr = attrs->SetClipAttribute(attr, &v);
    CFRelease(cf);
    if (hr != S_OK) {
        // The same override is applied to the clip and to the cloned
        // per-job attributes; report each rejection once.
        static std::vector<uint32_t> reported;
        if (std::find(reported.begin(), reported.end(), (uint32_t)attr) == reported.end()) {
            reported.push_back((uint32_t)attr);
            fprintf(stderr, "WARNING: SDK rejected %s \"%s\" (0x%08lx)\n", attr_name(attr), val.c_str(), (unsigned long)hr);
            print_attribute_list(attrs, attr);
        }
        return false;
    }
    return true;
}

// Apply clip-level overrides to a cloned clip processing attributes object.
// Returns the cloned+modified object (caller must Release), or nullptr if no overrides.
static IBlackmagicRawClipProcessingAttributes* create_clip_overrides(IBlackmagicRawClip* clip) {
    // Check if we have any clip-level overrides
    bool hasOverrides = !g_procParams.gamma.empty() || !g_procParams.gamut.empty() ||
        !g_procParams.postLutMode.empty() ||
        g_procParams.toneCurveContrast >= 0 || g_procParams.toneCurveSaturation >= 0 ||
        g_procParams.toneCurveMidpoint >= 0 || g_procParams.toneCurveHighlights >= 0 ||
        g_procParams.toneCurveShadows >= 0 || g_procParams.toneCurveBlackLevel >= 0 ||
        g_procParams.toneCurveWhiteLevel >= 0 || g_procParams.highlightRecovery >= 0 ||
        g_procParams.analogGain >= 0 || g_procParams.gamutCompression >= 0;
    if (!hasOverrides) return nullptr;

    // Clone the clip's current processing attributes
    IBlackmagicRawClipProcessingAttributes* attrs = nullptr;
    if (clip->CloneClipProcessingAttributes(&attrs) != S_OK || !attrs)
        return nullptr;

    set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributeGamma, g_procParams.gamma);
    set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributeGamut, g_procParams.gamut);
    set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributePost3DLUTMode, g_procParams.postLutMode);
    if (g_procParams.toneCurveContrast >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveContrast;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveContrast, &v);
    }
    if (g_procParams.toneCurveSaturation >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveSaturation;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveSaturation, &v);
    }
    if (g_procParams.toneCurveMidpoint >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveMidpoint;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveMidpoint, &v);
    }
    if (g_procParams.toneCurveHighlights >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveHighlights;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveHighlights, &v);
    }
    if (g_procParams.toneCurveShadows >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveShadows;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveShadows, &v);
    }
    if (g_procParams.toneCurveBlackLevel >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveBlackLevel;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveBlackLevel, &v);
    }
    if (g_procParams.toneCurveWhiteLevel >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.toneCurveWhiteLevel;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeToneCurveWhiteLevel, &v);
    }
    if (g_procParams.highlightRecovery >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeU16; v.uiVal = (uint16_t)g_procParams.highlightRecovery;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeHighlightRecovery, &v);
    }
    if (g_procParams.analogGain >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.analogGain;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeAnalogGain, &v);
    }
    if (g_procParams.gamutCompression >= 0) {
        Variant v; v.vt = blackmagicRawVariantTypeU16; v.uiVal = (uint16_t)g_procParams.gamutCompression;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeGamutCompressionEnable, &v);
    }
    return attrs;  // caller must Release()
}

static void apply_frame_overrides(IBlackmagicRawFrame* frame) {
    IBlackmagicRawFrameProcessingAttributes* attrs = nullptr;
    if (frame->QueryInterface(IID_IBlackmagicRawFrameProcessingAttributes, (void**)&attrs) != S_OK || !attrs)
        return;

    if (g_procParams.whiteBalanceKelvin > 0) {
        Variant v; v.vt = blackmagicRawVariantTypeU32; v.uintVal = (uint32_t)g_procParams.whiteBalanceKelvin;
        attrs->SetFrameAttribute(blackmagicRawFrameProcessingAttributeWhiteBalanceKelvin, &v);
    }
    if (g_procParams.whiteBalanceTint > -9000) {
        Variant v; v.vt = blackmagicRawVariantTypeS16; v.iVal = (int16_t)g_procParams.whiteBalanceTint;
        attrs->SetFrameAttribute(blackmagicRawFrameProcessingAttributeWhiteBalanceTint, &v);
    }
    if (g_procParams.exposure > -9000) {
        Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = g_procParams.exposure;
        attrs->SetFrameAttribute(blackmagicRawFrameProcessingAttributeExposure, &v);
    }
    if (g_procParams.iso > 0) {
        Variant v; v.vt = blackmagicRawVariantTypeU32; v.uintVal = (uint32_t)g_procParams.iso;
        attrs->SetFrameAttribute(blackmagicRawFrameProcessingAttributeISO, &v);
    }
    attrs->Release();
}

static bool g_use16bit = false;  // --16bit flag: output BGRA U16 instead of U8

// --scale: the SDK's partial-decode resolution. Applied twice per frame —
// as a read-job hint (so the reader skips the bitstream levels it won't
// need) and on the frame before the decode job is created.
static BlackmagicRawResolutionScale g_scale = blackmagicRawResolutionScaleFull;

// --pipeline auto|metal|cpu: where the SDK decodes and processes. `auto`
// (the default) takes the first Metal device the SDK offers and falls back
// to the CPU when there is none (or the device lacks the BGRA formats);
// `metal` fails instead of falling back; `cpu` is the old path. With a GPU
// pipeline the processed image is a MTLBuffer: the pixels are pulled back
// through the SDK's resource manager inside ProcessComplete — the only
// time the SDK guarantees the resource is still ours — by its host pointer
// when the buffer is CPU-visible (unified memory), else by a copy into CPU
// memory. `BRAW_HELPER_READBACK=copy` forces the copy (A/B).
static std::string g_pipelineArg = "auto";
// --step N: emit every Nth frame from --start (BRAW is intra-only, so the
// skipped frames cost nothing). The header's "frames" is the emitted count.
static uint64_t g_step = 1;
// --zerocopy N: hand frames over as N slots of global IOSurfaces (see
// braw_zc.h) — only "F <frame> <slot>" lines cross the pipe, the app maps
// the surfaces as textures and returns slots with "R <slot>" on stdin.
// --mono: the frame is ONE fisheye (shown to both eyes), not a side-by-side
// pair — one surface per slot.
static uint32_t g_zcSlots = 0;
static bool g_mono = false;
static IBlackmagicRawPipelineDevice* g_gpuDevice = nullptr;     // alive while the codec is
static IBlackmagicRawResourceManager* g_resourceManager = nullptr;
static std::string g_activePipeline = "cpu";
static std::string g_gpuDeviceName;

// Tell a read job how much of the frame will be decoded. Measured on a
// PYXIS 12K clip over SMB: a metadata-only read job (no decode) pulls the
// whole ~12–15 MB frame regardless of this hint, so it is only applied to
// frames that are actually decoded (--decode), where the scale is what
// determines the work.
static void set_read_hint(IBlackmagicRawJob* readJob, BlackmagicRawResolutionScale scale) {
    IBlackmagicRawReadJobHints* hints = nullptr;
    if (readJob->QueryInterface(IID_IBlackmagicRawReadJobHints, (void**)&hints) == S_OK && hints) {
        hints->SetReaderResolutionScale(scale);
        hints->Release();
    }
}

static bool parse_scale(const std::string& s, BlackmagicRawResolutionScale* out) {
    if (s == "full" || s == "1")         *out = blackmagicRawResolutionScaleFull;
    else if (s == "half" || s == "2")    *out = blackmagicRawResolutionScaleHalf;
    else if (s == "quarter" || s == "4") *out = blackmagicRawResolutionScaleQuarter;
    else if (s == "eighth" || s == "8")  *out = blackmagicRawResolutionScaleEighth;
    else return false;
    return true;
}

// ── GPU pipeline ─────────────────────────────────────────────────────────────

static std::string cfstr(CFStringRef str) {
    if (!str) return "";
    char buf[512];
    if (CFStringGetCString(str, buf, sizeof buf, kCFStringEncodingUTF8)) return buf;
    return "";
}

// The first Metal pipeline device the SDK offers (caller owns the
// reference; `name` receives its name), or null when the SDK has no Metal
// pipeline on this machine.
static IBlackmagicRawPipelineDevice* first_metal_device(IBlackmagicRawFactory* factory, IBlackmagicRaw* codec, std::string* name) {
    IBlackmagicRawConfiguration* cfg = nullptr;
    if (codec->QueryInterface(IID_IBlackmagicRawConfiguration, (void**)&cfg) != S_OK || !cfg) return nullptr;
    bool supported = false;
    HRESULT r = cfg->IsPipelineSupported(blackmagicRawPipelineMetal, &supported);
    cfg->Release();
    if (r != S_OK || !supported) return nullptr;
    IBlackmagicRawPipelineDeviceIterator* it = nullptr;
    if (factory->CreatePipelineDeviceIterator(blackmagicRawPipelineMetal, blackmagicRawInteropNone, &it) != S_OK || !it) {
        it = nullptr;
        if (factory->CreatePipelineDeviceIterator(blackmagicRawPipelineMetal, blackmagicRawInteropOpenGL, &it) != S_OK || !it) return nullptr;
    }
    IBlackmagicRawPipelineDevice* dev = nullptr;
    r = it->CreateDevice(&dev);
    it->Release();
    if (r != S_OK || !dev) return nullptr;
    CFStringRef n = nullptr;
    if (dev->GetName(&n) == S_OK && n) { *name = cfstr(n); CFRelease(n); }
    return dev;
}

static bool device_supports_format(IBlackmagicRawPipelineDevice* dev, BlackmagicRawResourceFormat fmt) {
    uint32_t n = 0;
    if (dev->GetSupportedResourceFormats(nullptr, &n) != S_OK || n == 0) return true;  // unknown: assume
    std::vector<BlackmagicRawResourceFormat> f(n);
    if (dev->GetSupportedResourceFormats(f.data(), &n) != S_OK) return true;
    for (uint32_t i = 0; i < n; i++) if (f[i] == fmt) return true;
    return false;
}

// Switch a fresh codec to the Metal pipeline per `--pipeline` (before
// OpenClip). Returns false only when `metal` was demanded and is not
// available; `auto` falls back to the CPU with a note on stderr.
static bool setup_gpu_pipeline(IBlackmagicRawFactory* factory, IBlackmagicRaw* codec) {
    g_activePipeline = "cpu";
    if (g_pipelineArg == "cpu") return true;
    std::string name;
    IBlackmagicRawPipelineDevice* dev = first_metal_device(factory, codec, &name);
    const BlackmagicRawResourceFormat fmt = g_use16bit ? blackmagicRawResourceFormatRGBAU16 : blackmagicRawResourceFormatRGBAU8;
    if (dev && !device_supports_format(dev, fmt)) {
        fprintf(stderr, "note: Metal device '%s' does not offer the RGBA %d-bit format; decoding on the CPU\n", name.c_str(), g_use16bit ? 16 : 8);
        dev->Release(); dev = nullptr;
    }
    if (!dev) {
        if (g_pipelineArg == "metal") { fprintf(stderr, "ERROR: no Metal pipeline device available\n"); return false; }
        fprintf(stderr, "note: no Metal pipeline device; decoding on the CPU\n");
        return true;
    }
    IBlackmagicRawConfiguration* cfg = nullptr;
    if (codec->QueryInterface(IID_IBlackmagicRawConfiguration, (void**)&cfg) != S_OK || !cfg) { dev->Release(); return g_pipelineArg != "metal"; }
    HRESULT r = cfg->SetFromDevice(dev);
    cfg->Release();
    if (r != S_OK) {
        fprintf(stderr, "%s: SetFromDevice('%s') failed (0x%08x)%s\n", g_pipelineArg == "metal" ? "ERROR" : "note",
                name.c_str(), (unsigned)r, g_pipelineArg == "metal" ? "" : "; decoding on the CPU");
        dev->Release();
        return g_pipelineArg != "metal";
    }
    IBlackmagicRawConfigurationEx* ex = nullptr;
    if (codec->QueryInterface(IID_IBlackmagicRawConfigurationEx, (void**)&ex) == S_OK && ex) {
        ex->GetResourceManager(&g_resourceManager);
        ex->Release();
    }
    g_gpuDevice = dev;
    g_gpuDeviceName = name;
    g_activePipeline = "metal";
    return true;
}

// Copy a processed image's pixels into `dst`: exactly `want` bytes, the
// packed w·h·bpp (the resource may be larger — the URSA Cine Immersive's
// reduced-scale tracks carry trailing padding). A CPU pipeline hands a
// host buffer; a GPU pipeline a MTLBuffer, read through the resource
// manager. Only valid inside ProcessComplete.
static HRESULT fetch_processed_bytes(IBlackmagicRawProcessedImage* img, size_t want, std::vector<uint8_t>& dst) {
    void* res = nullptr; uint32_t sz = 0;
    BlackmagicRawResourceType type = blackmagicRawResourceTypeBufferCPU;
    if (img->GetResource(&res) != S_OK || !res) return E_FAIL;
    img->GetResourceSizeBytes(&sz);
    img->GetResourceType(&type);
    if (sz < want) {
        fprintf(stderr, "ERROR: processed image resource is %u bytes; expected at least %zu\n", sz, want);
        return E_FAIL;
    }
    static bool warnedPad = false;
    if (sz != want && !warnedPad) {
        fprintf(stderr, "note: processed image resource is %u bytes (%zu packed) — writing the packed size\n", sz, want);
        warnedPad = true;
    }
    dst.resize(want);
    if (type == blackmagicRawResourceTypeBufferCPU) { memcpy(dst.data(), res, want); return S_OK; }
    if (!g_resourceManager) { fprintf(stderr, "ERROR: GPU resource without a resource manager\n"); return E_FAIL; }
    void* context = nullptr; void* queue = nullptr;
    img->GetResourceContextAndCommandQueue(&context, &queue);
    static const bool forceCopy = [] { const char* e = getenv("BRAW_HELPER_READBACK"); return e && std::string(e) == "copy"; }();
    HRESULT hostResult = E_FAIL;
    if (!forceCopy) {
        void* host = nullptr;
        hostResult = g_resourceManager->GetResourceHostPointer(context, queue, res, type, &host);
        if (hostResult == S_OK && host) {
            memcpy(dst.data(), host, want);
            return S_OK;
        }
    }
    // Not CPU-visible: copy the WHOLE resource (a partial copy is rejected)
    // into CPU memory, then keep the packed part.
    HRESULT copyResult;
    if (sz == want) {
        copyResult = g_resourceManager->CopyResource(context, queue, res, type, dst.data(), blackmagicRawResourceTypeBufferCPU, sz, false);
    } else {
        std::vector<uint8_t> whole(sz);
        copyResult = g_resourceManager->CopyResource(context, queue, res, type, whole.data(), blackmagicRawResourceTypeBufferCPU, sz, false);
        if (copyResult == S_OK) memcpy(dst.data(), whole.data(), want);
    }
    if (copyResult != S_OK)
        fprintf(stderr, "ERROR: GPU readback failed (host pointer 0x%08x, copy 0x%08x, resource %u bytes)\n",
                (unsigned)hostResult, (unsigned)copyResult, sz);
    return copyResult;
}

// ── Frame decode callback ────────────────────────────────────────────────────
//
// Jobs carry a ticket (the frame index, or index·2 + track for a two-track
// clip) as their user data. Several frames are in flight at once — the SDK
// decodes the next ones while the main thread writes the current — and
// they complete in any order, so results are posted by ticket and the
// writer waits for the one it needs. Frame buffers are recycled.

struct FrameResult {
    HRESULT status = S_OK;
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> bytes;
};

class DecodeCallback : public IBlackmagicRawCallback {
public:
    IBlackmagicRawClipProcessingAttributes* clipOverrides = nullptr;  // set before decode
    size_t bytesPerPixel = 4;
    // Zero-copy: frames go into `zc` slots instead of CPU buffers. The main
    // loop registers each ticket's slot before submitting its read job;
    // with a two-track clip both of a frame's tickets share the slot.
    ZcPool* zc = nullptr;
    bool dual = false;
    void set_slot(uint64_t ticket, uint32_t slot) { std::lock_guard<std::mutex> lk(mu_); slots_[ticket] = slot; }

    void ReadComplete(IBlackmagicRawJob* readJob, HRESULT result, IBlackmagicRawFrame* frame) override {
        void* ticket = nullptr;
        readJob->GetUserData(&ticket);
        IBlackmagicRawJob* decodeJob = nullptr;
        if (result == S_OK)
            // RGBA straight from the SDK: the app's texture order, no swizzle
            // downstream (the header says "format": "rgba").
            result = frame->SetResourceFormat(g_use16bit ? blackmagicRawResourceFormatRGBAU16 : blackmagicRawResourceFormatRGBAU8);
        if (result == S_OK)
            result = frame->SetResolutionScale(g_scale);
        if (result == S_OK)
            apply_frame_overrides(frame);
        if (result == S_OK)
            result = frame->CreateJobDecodeAndProcessFrame(clipOverrides, nullptr, &decodeJob);
        if (result == S_OK)
            result = decodeJob->SetUserData(ticket);
        if (result == S_OK)
            result = decodeJob->Submit();
        if (result != S_OK) {
            if (decodeJob) decodeJob->Release();
            FrameResult r; r.status = result;
            post((uint64_t)(uintptr_t)ticket, std::move(r));
        }
        readJob->Release();
    }

    void ProcessComplete(IBlackmagicRawJob* job, HRESULT result, IBlackmagicRawProcessedImage* img) override {
        void* ticket = nullptr;
        job->GetUserData(&ticket);
        FrameResult r;
        r.status = (result == S_OK && !img) ? E_FAIL : result;
        if (r.status == S_OK && zc) {
            img->GetWidth(&r.w);
            img->GetHeight(&r.h);
            r.status = fill_slot((uint64_t)(uintptr_t)ticket, img, r.w, r.h) ? S_OK : E_FAIL;
        } else if (r.status == S_OK) {
            img->GetWidth(&r.w);
            img->GetHeight(&r.h);
            r.bytes = take_buffer();
            r.status = fetch_processed_bytes(img, (size_t)r.w * r.h * bytesPerPixel, r.bytes);
        }
        post((uint64_t)(uintptr_t)ticket, std::move(r));
        job->Release();
    }

    void DecodeComplete(IBlackmagicRawJob*, HRESULT) override {}
    void TrimProgress(IBlackmagicRawJob*, float) override {}
    void TrimComplete(IBlackmagicRawJob*, HRESULT) override {}
    // SDK ≥ 4.x adds audio read jobs to the callback interface; we read audio
    // synchronously, so nothing to do here.
    void ReadAudioComplete(IBlackmagicRawJob* job, HRESULT, IBlackmagicRawAudioBuffer*) override { if (job) job->Release(); }
    void SidecarMetadataParseWarning(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void SidecarMetadataParseError(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void PreparePipelineComplete(void*, HRESULT) override {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID*) override { return E_NOTIMPL; }
    ULONG STDMETHODCALLTYPE AddRef() override { return 0; }
    ULONG STDMETHODCALLTYPE Release() override { return 0; }

    // Block until the ticket's frame has been posted, and take it.
    FrameResult wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return done_.count(ticket) > 0; });
        FrameResult r = std::move(done_[ticket]);
        done_.erase(ticket);
        return r;
    }
    // Hand a written frame's buffer back for the next decode.
    void recycle(std::vector<uint8_t>&& buf) {
        std::lock_guard<std::mutex> lk(mu_);
        if (pool_.size() < 8) pool_.push_back(std::move(buf));
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<uint64_t, FrameResult> done_;
    std::vector<std::vector<uint8_t>> pool_;
    std::map<uint64_t, uint32_t> slots_;

    // Copy the processed image into its ticket's slot: a GPU blit from the
    // SDK's MTLBuffer, or a memcpy from a CPU resource.
    bool fill_slot(uint64_t ticket, IBlackmagicRawProcessedImage* img, uint32_t w, uint32_t h) {
        uint32_t slot;
        { std::lock_guard<std::mutex> lk(mu_); auto it = slots_.find(ticket); if (it == slots_.end()) return false; slot = it->second; slots_.erase(it); }
        void* res = nullptr; img->GetResource(&res);
        BlackmagicRawResourceType type = blackmagicRawResourceTypeBufferCPU;
        img->GetResourceType(&type);
        if (!res) return false;
        const int track = dual ? (int)(ticket & 1) : -1;
        if (type == blackmagicRawResourceTypeBufferCPU)
            return zc_pool_fill_cpu(zc, slot, (const uint8_t*)res, w, h, w * 4, track);
        void* context = nullptr; void* queue = nullptr;
        img->GetResourceContextAndCommandQueue(&context, &queue);
        return zc_pool_fill_metal(zc, slot, res, queue, w, h, w * 4, track);
    }

    std::vector<uint8_t> take_buffer() {
        std::lock_guard<std::mutex> lk(mu_);
        if (pool_.empty()) return {};
        std::vector<uint8_t> b = std::move(pool_.back());
        pool_.pop_back();
        return b;
    }
    void post(uint64_t ticket, FrameResult&& r) {
        { std::lock_guard<std::mutex> lk(mu_); done_[ticket] = std::move(r); }
        cv_.notify_all();
    }
};

// ── Open clip helper ─────────────────────────────────────────────────────────

struct ClipContext {
    IBlackmagicRawFactory* factory = nullptr;
    IBlackmagicRaw* codec = nullptr;
    IBlackmagicRawClip* clip = nullptr;
    IBlackmagicRawClipMultiVideo* multiVideo = nullptr;  // non-null if multi-video
    uint32_t videoTrackCount = 1;

    // `wantGpu`: configure the decode pipeline per `--pipeline` (only the
    // modes that decode frames pay for the Metal device).
    bool open(const char* path, bool wantGpu = false) {
        factory = CreateBlackmagicRawFactoryInstanceFromPath(
            CFStringCreateWithCString(NULL, SDK_LIB_PATH, kCFStringEncodingUTF8));
        if (!factory) { std::cerr << "ERROR: Cannot create BRAW factory. SDK not found." << std::endl; return false; }

        if (factory->CreateCodec(&codec) != S_OK) { std::cerr << "ERROR: Cannot create codec." << std::endl; return false; }
        if (wantGpu && !setup_gpu_pipeline(factory, codec)) return false;

        CFStringRef cfPath = CFStringCreateWithCString(NULL, path, kCFStringEncodingUTF8);
        HRESULT result = codec->OpenClip(cfPath, &clip);
        CFRelease(cfPath);
        if (result != S_OK) { std::cerr << "ERROR: Cannot open clip." << std::endl; return false; }

        // Check for multi-video tracks
        if (clip->QueryInterface(IID_IBlackmagicRawClipMultiVideo, (void**)&multiVideo) == S_OK && multiVideo) {
            multiVideo->GetVideoTrackCount(&videoTrackCount);
        }

        return true;
    }

    void close() {
        if (multiVideo) { multiVideo->Release(); multiVideo = nullptr; }
        if (clip) { clip->Release(); clip = nullptr; }
        if (g_resourceManager) { g_resourceManager->Release(); g_resourceManager = nullptr; }
        if (codec) { codec->Release(); codec = nullptr; }
        if (g_gpuDevice) { g_gpuDevice->Release(); g_gpuDevice = nullptr; }
        if (factory) { factory->Release(); factory = nullptr; }
    }

    ~ClipContext() { close(); }
};

// ── Mode: --info ─────────────────────────────────────────────────────────────

static int do_info(const char* path) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;

    uint32_t width = 0, height = 0;
    uint64_t frameCount = 0;
    float frameRate = 0;
    ctx.clip->GetWidth(&width);
    ctx.clip->GetHeight(&height);
    ctx.clip->GetFrameCount(&frameCount);
    ctx.clip->GetFrameRate(&frameRate);

    std::string cameraModel = get_metadata_string(ctx.clip, "camera_type");
    std::string firmware = get_metadata_string(ctx.clip, "firmware_version");

    // Gyro info
    uint32_t gyroCount = 0, accelCount = 0;
    float gyroRate = 0, accelRate = 0;

    IBlackmagicRawClipGyroscopeMotion* gyro = nullptr;
    if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipGyroscopeMotion, (void**)&gyro) == S_OK && gyro) {
        gyro->GetSampleCount(&gyroCount);
        gyro->GetSampleRate(&gyroRate);
        gyro->Release();
    }

    IBlackmagicRawClipAccelerometerMotion* accel = nullptr;
    if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipAccelerometerMotion, (void**)&accel) == S_OK && accel) {
        accel->GetSampleCount(&accelCount);
        accel->GetSampleRate(&accelRate);
        accel->Release();
    }

    // Audio info
    uint64_t audioSampleCount = 0;
    uint32_t audioSampleRate = 0;
    uint32_t audioBitDepth = 0;
    uint32_t audioChannels = 0;
    IBlackmagicRawClipAudio* audio = nullptr;
    if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipAudio, (void**)&audio) == S_OK && audio) {
        audio->GetAudioSampleCount(&audioSampleCount);
        audio->GetAudioSampleRate(&audioSampleRate);
        audio->GetAudioBitDepth(&audioBitDepth);
        audio->GetAudioChannelCount(&audioChannels);
        audio->Release();
    }

    // Sensor readout info from clip-level metadata. `sensor_line_time` is
    // in microseconds per line (an integer on PYXIS 12K fw 10.0: 3);
    // `sensor_area_captured` = [w, h] of the window read out (the frame).
    double sensorAreaHeight = get_metadata_float(ctx.clip, "sensor_area_captured_height", 0);
    double areaW = 0, areaH = 0, cropX = 0, cropY = 0;
    if (get_metadata_pair(ctx.clip, "sensor_area_captured", &areaW, &areaH) && sensorAreaHeight <= 0) sensorAreaHeight = areaH;
    get_metadata_pair(ctx.clip, "crop_origin", &cropX, &cropY);
    double sensorLineTime = get_metadata_float(ctx.clip, "sensor_line_time", 0);
    std::string viewingGamma = get_metadata_string(ctx.clip, "viewing_gamma");
    std::string viewingGamut = get_metadata_string(ctx.clip, "viewing_gamut");
    double viewingGen = get_metadata_float(ctx.clip, "viewing_bmdgen", 0);
    std::string lutName = get_metadata_string(ctx.clip, "post_3dlut_embedded_title");
    std::string shutterType = get_metadata_string(ctx.clip, "shutter_type");
    // Shutter, sampled at five frames (cinema cameras hold it within a take;
    // the spread is reported so a varying one shows).
    double shutterMin = 0, shutterMax = 0; std::string shutterText;
    {
        ShutterCallback scb; scb.fps = frameRate;
        ctx.codec->SetCallback(&scb);
        uint64_t idx[5] = {0, frameCount / 4, frameCount / 2, (frameCount * 3) / 4, frameCount ? frameCount - 1 : 0};
        for (int k = 0; k < 5; k++) {
            scb.done.store(false); scb.out = FrameShutter();
            IBlackmagicRawJob* job = nullptr;
            if (ctx.clip->CreateJobReadFrame(idx[k], &job) != S_OK || !job) continue;
            if (job->Submit() != S_OK) { job->Release(); continue; }
            ctx.codec->FlushJobs();
            if (scb.out.seconds > 0) {
                if (shutterMin == 0 || scb.out.seconds < shutterMin) shutterMin = scb.out.seconds;
                if (scb.out.seconds > shutterMax) shutterMax = scb.out.seconds;
                if (k == 0) shutterText = scb.out.text;
            }
        }
        ctx.codec->SetCallback(nullptr);
    }

    double readoutMs = 0;
    if (sensorAreaHeight > 0 && sensorLineTime > 0)
        readoutMs = (sensorAreaHeight * sensorLineTime) / 1000.0;

    // Firmware v7.9 bug: readout time doubled
    if (firmware == "7.9" && frameRate > 0 && readoutMs > (1000.0 / frameRate))
        readoutMs /= 2.0;

    // The pipeline `--decode` takes by default: the first Metal device when
    // the SDK offers one, else the CPU.
    std::string pipeline = "cpu", gpuDevice;
    if (IBlackmagicRawPipelineDevice* dev = first_metal_device(ctx.factory, ctx.codec, &gpuDevice)) {
        pipeline = "metal";
        dev->Release();
    }

    // Processed image size at each --scale, as the SDK reports it.
    bool haveScales = false;
    uint32_t scaleW[4] = {0, 0, 0, 0}, scaleH[4] = {0, 0, 0, 0};
    {
        static const BlackmagicRawResolutionScale scales[4] = {
            blackmagicRawResolutionScaleFull, blackmagicRawResolutionScaleHalf,
            blackmagicRawResolutionScaleQuarter, blackmagicRawResolutionScaleEighth };
        IBlackmagicRawClipResolutions* res = nullptr;
        if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipResolutions, (void**)&res) == S_OK && res) {
            haveScales = true;
            for (int k = 0; k < 4; k++)
                if (res->GetClosestResolutionForScale(scales[k], &scaleW[k], &scaleH[k]) != S_OK) haveScales = false;
            res->Release();
        }
    }

    // Output JSON
    printf("{\n");
    printf("  \"width\": %u,\n", width);
    printf("  \"height\": %u,\n", height);
    printf("  \"frame_count\": %llu,\n", frameCount);
    printf("  \"frame_rate\": %.6f,\n", frameRate);
    printf("  \"duration\": %.6f,\n", frameCount > 0 && frameRate > 0 ? (double)frameCount / frameRate : 0.0);
    printf("  \"camera_model\": \"%s\",\n", cameraModel.c_str());
    printf("  \"firmware_version\": \"%s\",\n", firmware.c_str());
    printf("  \"video_track_count\": %u,\n", ctx.videoTrackCount);
    printf("  \"gyro_sample_count\": %u,\n", gyroCount);
    printf("  \"gyro_sample_rate\": %.2f,\n", gyroRate);
    printf("  \"accel_sample_count\": %u,\n", accelCount);
    printf("  \"accel_sample_rate\": %.2f,\n", accelRate);
    printf("  \"audio_sample_count\": %llu,\n", (unsigned long long)audioSampleCount);
    printf("  \"audio_sample_rate\": %u,\n", audioSampleRate);
    printf("  \"audio_bit_depth\": %u,\n", audioBitDepth);
    printf("  \"audio_channels\": %u,\n", audioChannels);
    printf("  \"readout_ms\": %.6f,\n", readoutMs);
    printf("  \"sensor_line_time_us\": %.6f,\n", sensorLineTime);
    printf("  \"sensor_area_captured\": [%.0f, %.0f],\n", areaW, areaH);
    printf("  \"crop_origin\": [%.0f, %.0f],\n", cropX, cropY);
    printf("  \"shutter_type\": \"%s\",\n", shutterType.c_str());
    printf("  \"shutter_text\": \"%s\",\n", shutterText.c_str());
    printf("  \"shutter_s_min\": %.9f,\n", shutterMin);
    printf("  \"shutter_s_max\": %.9f,\n", shutterMax);
    printf("  \"viewing_gamma\": \"%s\",\n", viewingGamma.c_str());
    printf("  \"viewing_gamut\": \"%s\",\n", viewingGamut.c_str());
    printf("  \"viewing_bmdgen\": %.0f,\n", viewingGen);
    printf("  \"embedded_lut\": \"%s\",\n", json_escape(lutName).c_str());
    printf("  \"post_3dlut_mode\": \"%s\",\n", json_escape(get_metadata_string(ctx.clip, "post_3dlut_mode")).c_str());
    printf("  \"pipeline\": \"%s\",\n", pipeline.c_str());
    printf("  \"zerocopy\": true,\n");
    if (pipeline == "metal") printf("  \"gpu_device\": \"%s\",\n", gpuDevice.c_str());
    else printf("  \"gpu_device\": null,\n");
    if (haveScales)
        printf("  \"resolution_scales\": {\"full\": [%u, %u], \"half\": [%u, %u], \"quarter\": [%u, %u], \"eighth\": [%u, %u]}\n",
               scaleW[0], scaleH[0], scaleW[1], scaleH[1], scaleW[2], scaleH[2], scaleW[3], scaleH[3]);
    else
        printf("  \"resolution_scales\": null\n");
    printf("}\n");

    return 0;
}

// ── Mode: --shutters ─────────────────────────────────────────────────────────
// Per-frame shutter for frames 0, step, 2·step, …: one line per frame on
// stdout, `<frame_index> <shutter_seconds> "<shutter_value text>"`. Read
// jobs only (no decode); up to kInFlight are submitted before each
// FlushJobs so the reader's latency overlaps. Cost is pure I/O — the SDK
// reads the whole frame for its metadata (≈ the clip's size on disk per
// full pass, ~27 frames/s from a NAS), so sample with --step when the
// clip is not local.

static int do_shutters(const char* path, uint64_t step) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;
    if (step == 0) step = 1;

    uint64_t frameCount = 0; float fps = 0;
    ctx.clip->GetFrameCount(&frameCount);
    ctx.clip->GetFrameRate(&fps);

    ShutterCallback cb; cb.fps = fps;
    ctx.codec->SetCallback(&cb);

    const int kInFlight = 32;
    int inFlight = 0;
    uint64_t batchStart = 0;
    auto flush_batch = [&](uint64_t batchEnd) {
        ctx.codec->FlushJobs();
        std::lock_guard<std::mutex> lock(cb.mutex);
        for (uint64_t f = batchStart; f < batchEnd; f += step) {
            auto it = cb.results.find(f);
            if (it == cb.results.end()) printf("%llu 0 \"\"\n", (unsigned long long)f);
            else printf("%llu %.9g \"%s\"\n", (unsigned long long)f, it->second.seconds, it->second.text.c_str());
        }
        cb.results.clear();
        inFlight = 0;
        batchStart = batchEnd;
    };

    for (uint64_t f = 0; f < frameCount; f += step) {
        IBlackmagicRawJob* job = nullptr;
        if (ctx.clip->CreateJobReadFrame(f, &job) == S_OK && job) {
            if (job->Submit() == S_OK) inFlight++;
            else job->Release();
        }
        if (inFlight >= kInFlight) flush_batch(f + step);
    }
    flush_batch(frameCount ? ((frameCount - 1) / step + 1) * step : 0);
    fflush(stdout);
    ctx.codec->SetCallback(nullptr);
    return 0;
}

// ── Mode: --lut ──────────────────────────────────────────────────────────────
// Write the clip's embedded post-3D LUT (the camera's "Film → Extended
// Video" style display LUT) as a standard .cube file.
//
// Source, in order of preference:
//   1. IBlackmagicRawClipProcessingAttributes::GetEmbeddedPost3DLUT —
//      name/title/size + a CPU float buffer of size³×3.
//   2. clip metadata `post_3dlut_embedded_{size,title,name,data}` — the
//      data arrives as a U8 SafeArray of the same floats' little-endian
//      bytes (4·size³·3 of them) on PYXIS 12K fw 10.0, or as Float32.
//
// Element order, measured on the PYXIS 12K fw 10.0 Gen 5 LUT against
// Resolve's shipped "Blackmagic Gen 5 Film to Extended Video.cube": the
// SDK buffer is NOT in .cube order. Entry 1 (the first step of the
// fastest axis) is (0, 0, x) and entry `size` (first step of the middle
// axis) is (0, x, 0), and reversing each entry's three components
// reproduces Resolve's cube to 5e-7 — i.e. .cube index order (red
// fastest) with the components stored B, G, R, matching the SDK's
// BGRA-native pixel convention. (Transposing the outer/inner axes with
// RGB components matches that symmetric LUT equally well; the two are
// only distinguishable on an R/B-asymmetric LUT. The component-swap
// reading is the one used here.) do_lut detects the layout from entries
// 1 and `size` and falls back to writing the buffer as stored when it
// sees neither pattern.

static bool read_embedded_lut(IBlackmagicRawClip* clip, uint32_t* size, std::string* name,
                              std::string* title, std::vector<float>* data, const char** source) {
    // 1. the clip processing attributes object
    IBlackmagicRawClipProcessingAttributes* attrs = nullptr;
    if (clip->QueryInterface(IID_IBlackmagicRawClipProcessingAttributes, (void**)&attrs) == S_OK && attrs) {
        IBlackmagicRawPost3DLUT* lut = nullptr;
        if (attrs->GetEmbeddedPost3DLUT(&lut) == S_OK && lut) {
            uint32_t n = 0, bytes = 0; void* buf = nullptr; CFStringRef s = nullptr;
            lut->GetSize(&n);
            lut->GetResourceSizeBytes(&bytes);
            lut->GetResourceCPU(&buf);
            if (lut->GetName(&s) == S_OK && s) { *name = cfstr_to_string(s); CFRelease(s); }
            if (lut->GetTitle(&s) == S_OK && s) { *title = cfstr_to_string(s); CFRelease(s); }
            const size_t want = (size_t)n * n * n * 3;
            if (n > 0 && buf && bytes >= want * sizeof(float)) {
                data->assign((const float*)buf, (const float*)buf + want);
                *size = n; *source = "GetEmbeddedPost3DLUT";
                lut->Release(); attrs->Release();
                return true;
            }
            lut->Release();
        }
        attrs->Release();
    }

    // 2. the metadata keys
    uint32_t n = (uint32_t)get_metadata_float(clip, "post_3dlut_embedded_size", 0);
    *title = get_metadata_string(clip, "post_3dlut_embedded_title");
    *name = get_metadata_string(clip, "post_3dlut_embedded_name");
    if (n == 0) return false;
    const size_t want = (size_t)n * n * n * 3;
    Variant v; VariantInit(&v);
    CFStringRef k = CFStringCreateWithCString(NULL, "post_3dlut_embedded_data", kCFStringEncodingUTF8);
    HRESULT hr = clip->GetMetadata(k, &v); CFRelease(k);
    bool ok = false;
    if (hr == S_OK && v.vt == blackmagicRawVariantTypeSafeArray && v.parray) {
        BlackmagicRawVariantType et; long lo = 0, hi = -1; void* raw = nullptr;
        SafeArrayGetVartype(v.parray, &et); SafeArrayGetLBound(v.parray, 1, &lo); SafeArrayGetUBound(v.parray, 1, &hi); SafeArrayAccessData(v.parray, &raw);
        const size_t count = (hi >= lo) ? (size_t)(hi - lo + 1) : 0;
        if (raw && et == blackmagicRawVariantTypeFloat32 && count >= want) {
            data->assign((const float*)raw, (const float*)raw + want); ok = true;
        } else if (raw && et == blackmagicRawVariantTypeU8 && count >= want * sizeof(float)) {
            data->resize(want);
            memcpy(data->data(), raw, want * sizeof(float));  // little-endian float32 bytes
            ok = true;
        }
    }
    VariantClear(&v);
    if (ok) { *size = n; *source = "post_3dlut_embedded_data"; }
    return ok;
}

static int do_lut(const char* path, const char* outPath) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;

    uint32_t size = 0; std::string name, title; std::vector<float> data; const char* source = "";
    if (!read_embedded_lut(ctx.clip, &size, &name, &title, &data, &source)) {
        fprintf(stderr, "ERROR: clip has no embedded post-3D LUT\n");
        return 1;
    }

    // Detect the component order from the first red step (entry 1) and
    // the first green step (entry `size`): a display curve maps a
    // red-only input to a red-dominant output, so the component that
    // entry 1 lights up is where R is stored.
    const float* e1 = &data[3];
    const float* eS = &data[(size_t)size * 3];
    const bool bgr = e1[2] > 0 && e1[0] == 0 && eS[1] > 0;   // (B, G, R) per entry — what the SDK gives
    const bool rgb = e1[0] > 0 && e1[2] == 0 && eS[1] > 0;   // already .cube order
    fprintf(stderr, "lut: \"%s\" (%s) size %u from %s; entry[1] = (%g, %g, %g), entry[%u] = (%g, %g, %g) → %s\n",
            title.c_str(), name.c_str(), size, source, e1[0], e1[1], e1[2], size, eS[0], eS[1], eS[2],
            bgr ? "red fastest, components stored B,G,R (swapped to R,G,B)"
                : rgb ? "red fastest, components R,G,B (cube order)"
                      : "unrecognised layout, written as stored");

    FILE* f = fopen(outPath, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", outPath); return 1; }
    fprintf(f, "# Embedded post-3D LUT extracted from %s\n", path);
    if (!name.empty()) fprintf(f, "# %s\n", name.c_str());
    fprintf(f, "TITLE \"%s\"\n", title.empty() ? "Embedded LUT" : title.c_str());
    fprintf(f, "LUT_3D_SIZE %u\n", size);
    fprintf(f, "DOMAIN_MIN 0.0 0.0 0.0\n");
    fprintf(f, "DOMAIN_MAX 1.0 1.0 1.0\n");
    const size_t entries = (size_t)size * size * size;
    for (size_t i = 0; i < entries; i++) {
        const float* e = &data[i * 3];
        if (bgr) fprintf(f, "%.6f %.6f %.6f\n", e[2], e[1], e[0]);
        else     fprintf(f, "%.6f %.6f %.6f\n", e[0], e[1], e[2]);
    }
    fclose(f);
    fprintf(stderr, "lut: wrote %zu entries to %s\n", entries, outPath);
    return 0;
}

// ── Mode: --decode ───────────────────────────────────────────────────────────

// Submit a read job for one frame (of one track) under `ticket`.
static bool submit_read(ClipContext& ctx, uint64_t ticket, uint64_t frameIndex, int track) {
    IBlackmagicRawJob* readJob = nullptr;
    HRESULT result = (track >= 0 && ctx.multiVideo)
        ? ctx.multiVideo->CreateJobReadFrame((uint32_t)track, frameIndex, &readJob)
        : ctx.clip->CreateJobReadFrame(frameIndex, &readJob);
    if (result != S_OK || !readJob) {
        fprintf(stderr, "ERROR: CreateJobReadFrame failed at frame %llu track %d\n", (unsigned long long)frameIndex, track);
        return false;
    }
    // Reader hint: at reduced scale the SDK can skip reading the detail
    // levels it will not decode (the frame-level SetResolutionScale in
    // ReadComplete is what actually sizes the output).
    set_read_hint(readJob, g_scale);
    readJob->SetUserData((void*)(uintptr_t)ticket);
    result = readJob->Submit();
    if (result != S_OK) {
        readJob->Release();
        fprintf(stderr, "ERROR: Submit failed at frame %llu\n", (unsigned long long)frameIndex);
        return false;
    }
    return true;
}

// Frames kept in flight beyond the one being written: bounded by memory
// (about 1.5 GB of decoded frames), at most 3.
static int frames_in_flight_for(size_t frameBytes) {
    size_t k = frameBytes ? (size_t)(1.5e9 / (double)frameBytes) : 3;
    return (int)std::max<size_t>(1, std::min<size_t>(3, k));
}

// Zero-copy hand-off (`--zerocopy N`): see braw_zc.h. The app's releases
// ("R <slot>" on stdin) feed the free list; the main loop never submits a
// read job without a slot for it. Output order is frame order.
static int do_decode_zc(ClipContext& ctx, DecodeCallback& cb, IBlackmagicRawClipProcessingAttributes* clipOverrides,
                        uint64_t start, uint64_t end, uint64_t step, int decode_track, bool is_dual) {
    auto fail = [&]() -> int { ctx.codec->FlushJobs(); if (clipOverrides) clipOverrides->Release(); return 1; };
    if (g_use16bit) { fprintf(stderr, "ERROR: --zerocopy is 8-bit only\n"); return fail(); }
    // The processed size at this scale, from the SDK (one track's for a
    // two-track clip), sizes the pool before anything is decoded.
    uint32_t pw = 0, ph = 0;
    {
        IBlackmagicRawClipResolutions* res = nullptr;
        if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipResolutions, (void**)&res) != S_OK || !res
            || res->GetClosestResolutionForScale(g_scale, &pw, &ph) != S_OK || pw == 0 || ph == 0) {
            if (res) res->Release();
            fprintf(stderr, "ERROR: the SDK did not report the processed size at this scale\n");
            return fail();
        }
        res->Release();
    }
    ZcGeometry g;
    g.mono = g_mono;
    if (g_mono) { g.eye_w = pw; g.eye_h = ph; g.pad_rows = 0; }
    else if (is_dual) { g.eye_w = pw; g.eye_h = pw; g.pad_rows = ((int32_t)pw - (int32_t)ph) / 2; }
    else { g.eye_w = pw / 2; g.eye_h = pw / 2; g.pad_rows = ((int32_t)(pw / 2) - (int32_t)ph) / 2; }
    const uint32_t slots = std::max<uint32_t>(2, g_zcSlots);
    ZcPool* pool = zc_pool_create(slots, g);
    if (!pool) return fail();
    cb.zc = pool;
    cb.dual = is_dual;

    const uint64_t emitted = (end - start + step - 1) / step;
    fprintf(stderr, "{\"width\": %u, \"height\": %u, \"frames\": %llu, \"dual_stream\": %s, \"bit_depth\": 8, \"format\": \"rgba\", \"step\": %llu",
            is_dual ? pw * 2 : pw, is_dual ? ph : ph, (unsigned long long)emitted, is_dual ? "true" : "false", (unsigned long long)step);
    fprintf(stderr, ", \"pipeline\": \"%s\"", g_activePipeline.c_str());
    if (g_activePipeline == "metal") fprintf(stderr, ", \"gpu_device\": \"%s\"", g_gpuDeviceName.c_str());
    fprintf(stderr, ", \"transport\": \"iosurface\", \"slots\": %u, \"eye_width\": %u, \"eye_height\": %u, \"pad_rows\": %d, \"mono\": %s, \"surfaces\": [",
            slots, g.eye_w, g.eye_h, g.pad_rows, g_mono ? "true" : "false");
    for (uint32_t s = 0; s < slots; s++)
        fprintf(stderr, "%s[%u, %u]", s ? ", " : "", zc_pool_surface_id(pool, s, 0), zc_pool_surface_id(pool, s, 1));
    fprintf(stderr, "]}\n");
    fflush(stderr);

    // Free slots: all at first; the app returns them on stdin. stdin EOF =
    // the app is gone.
    std::mutex fm; std::condition_variable fcv;
    std::vector<uint32_t> free_slots;
    for (uint32_t s = 0; s < slots; s++) free_slots.push_back(s);
    bool app_gone = false;
    std::thread reader([&]() {
        char line[128];
        while (fgets(line, sizeof line, stdin)) {
            unsigned s = 0;
            if (sscanf(line, "R %u", &s) == 1 && s < slots) {
                std::lock_guard<std::mutex> lk(fm); free_slots.push_back(s);
            }
            fcv.notify_all();
        }
        { std::lock_guard<std::mutex> lk(fm); app_gone = true; }
        fcv.notify_all();
    });
    auto take_slot = [&](uint32_t* slot) -> bool {
        std::unique_lock<std::mutex> lk(fm);
        fcv.wait(lk, [&] { return app_gone || !free_slots.empty(); });
        if (free_slots.empty()) return false;
        *slot = free_slots.back(); free_slots.pop_back();
        return true;
    };

    // Keep up to `ahead` frames decoding while the app consumes, leaving the
    // app two slots (the frame on screen and the one whose GPU work may
    // still be in flight) before the helper would have to wait.
    const int ahead = (int)std::max<uint32_t>(1, std::min<uint32_t>(3, slots - 2));
    std::vector<std::pair<uint64_t, uint32_t>> inflight;   // (frame, slot), oldest first
    uint64_t next = start;
    int rc = 0;
    auto top_up = [&]() -> bool {
        while ((int)inflight.size() < ahead && next < end) {
            uint32_t slot;
            if (!take_slot(&slot)) return false;     // app gone
            if (is_dual) {
                cb.set_slot(next * 2 + 0, slot); cb.set_slot(next * 2 + 1, slot);
                if (!submit_read(ctx, next * 2 + 0, next, 0) || !submit_read(ctx, next * 2 + 1, next, 1)) { rc = 1; return false; }
            } else {
                cb.set_slot(next, slot);
                if (!submit_read(ctx, next, next, decode_track)) { rc = 1; return false; }
            }
            inflight.emplace_back(next, slot);
            next += step;
        }
        return true;
    };
    bool going = top_up();
    while (going && !inflight.empty()) {
        const auto [frame, slot] = inflight.front();
        inflight.erase(inflight.begin());
        bool ok;
        if (is_dual) {
            FrameResult a = cb.wait(frame * 2 + 0), b = cb.wait(frame * 2 + 1);
            ok = a.status == S_OK && b.status == S_OK;
        } else {
            FrameResult r = cb.wait(frame);
            ok = r.status == S_OK;
        }
        if (!ok) { fprintf(stderr, "ERROR: Decode failed at frame %llu\n", (unsigned long long)frame); rc = 1; break; }
        // Report BEFORE topping up: topping up may have to wait for the app
        // to return a slot, and the app returns slots only as it moves on
        // to frames it has been told about — reporting after waiting
        // deadlocked at the very first frame.
        printf("F %llu %u\n", (unsigned long long)frame, slot);
        fflush(stdout);
        going = top_up();
    }
    ctx.codec->FlushJobs();
    // Let the reader end: closing stdin from our side is not possible, so
    // leave it to the app (it closes its end when it drops the decoder).
    fclose(stdin);
    reader.join();
    cb.zc = nullptr;
    zc_pool_destroy(pool);
    if (clipOverrides) clipOverrides->Release();
    return rc;
}

static int do_decode(const char* path, uint64_t start, uint64_t count, int track) {
    ClipContext ctx;
    if (!ctx.open(path, /*wantGpu=*/true)) return 1;
    // The app applies the clip's LUT itself (it extracts the embedded cube):
    // never let the SDK bake a post-3D LUT into the decode, whatever the
    // clip or a sidecar says — unless a --look already chose.
    if (g_procParams.postLutMode.empty()) g_procParams.postLutMode = "Disabled";

    // Apply clip-level processing overrides directly on the clip
    // (SetClipAttribute on the clip modifies defaults for all subsequent decodes)
    {
        IBlackmagicRawClipProcessingAttributes* attrs = nullptr;
        if (ctx.clip->QueryInterface(IID_IBlackmagicRawClipProcessingAttributes, (void**)&attrs) == S_OK && attrs) {
            auto setFloat = [&](uint32_t attr, float val) {
                if (val >= 0) { Variant v; v.vt = blackmagicRawVariantTypeFloat32; v.fltVal = val; attrs->SetClipAttribute((BlackmagicRawClipProcessingAttribute)attr, &v); }
            };
            auto setU16 = [&](uint32_t attr, int val) {
                if (val >= 0) { Variant v; v.vt = blackmagicRawVariantTypeU16; v.uiVal = (uint16_t)val; attrs->SetClipAttribute((BlackmagicRawClipProcessingAttribute)attr, &v); }
            };
            set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributeGamma, g_procParams.gamma);
            set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributeGamut, g_procParams.gamut);
            set_clip_string_attr(attrs, blackmagicRawClipProcessingAttributePost3DLUTMode, g_procParams.postLutMode);
            setFloat(0x74636F6E, g_procParams.toneCurveContrast);
            setFloat(0x74736174, g_procParams.toneCurveSaturation);
            setFloat(0x746D6964, g_procParams.toneCurveMidpoint);
            setFloat(0x74686968, g_procParams.toneCurveHighlights);
            setFloat(0x74736861, g_procParams.toneCurveShadows);
            setFloat(0x74626C6B, g_procParams.toneCurveBlackLevel);
            setFloat(0x74776974, g_procParams.toneCurveWhiteLevel);
            setU16(0x686C7279, g_procParams.highlightRecovery);
            setFloat(0x6761696E, g_procParams.analogGain);
            setU16(0x67616365, g_procParams.gamutCompression);
            attrs->Release();
        }
    }
    // Also create cloned overrides for passing to CreateJobDecodeAndProcessFrame
    IBlackmagicRawClipProcessingAttributes* clipOverrides = create_clip_overrides(ctx.clip);

    uint64_t frameCount = 0;
    ctx.clip->GetFrameCount(&frameCount);

    bool is_dual = ctx.videoTrackCount >= 2 && track < 0;

    if (start >= frameCount) {
        std::cerr << "ERROR: start frame " << start << " >= frame count " << frameCount << std::endl;
        if (clipOverrides) clipOverrides->Release();
        return 1;
    }

    uint64_t end = (count == 0) ? frameCount : std::min(start + count, frameCount);

    const size_t bytesPerPixel = g_use16bit ? 8 : 4;  // BGRA U16 = 8, U8 = 4
    const int bitDepth = g_use16bit ? 16 : 8;

    DecodeCallback cb;
    cb.clipOverrides = clipOverrides;  // may be nullptr if no overrides
    cb.bytesPerPixel = bytesPerPixel;
    ctx.codec->SetCallback(&cb);

    // Every exit drains the SDK's jobs first: callbacks must not outlive `cb`.
    auto fail = [&]() -> int {
        ctx.codec->FlushJobs();
        if (clipOverrides) clipOverrides->Release();
        return 1;
    };
    // The frame the SDK hands back must be exactly the tightly packed
    // w × h × bpp announced in the header (the consumer reads frames by that
    // byte count): a mid-clip size change is a hard error, not a garbled
    // stream.
    auto check = [&](const FrameResult& r, uint64_t i, uint32_t w, uint32_t h) -> bool {
        if (r.status != S_OK) {
            fprintf(stderr, "ERROR: Decode failed at frame %llu (0x%08x)\n", (unsigned long long)i, (unsigned)r.status);
            return false;
        }
        if (r.w != w || r.h != h || r.bytes.size() != (size_t)w * h * bytesPerPixel) {
            fprintf(stderr, "ERROR: frame %llu is %ux%u (%zu bytes); expected %ux%u\n",
                    (unsigned long long)i, r.w, r.h, r.bytes.size(), w, h);
            return false;
        }
        return true;
    };
    const uint64_t step = g_step;
    const uint64_t emitted = (end - start + step - 1) / step;
    auto header = [&](uint32_t outW, uint32_t outH, bool dual, uint32_t w0, uint32_t h0, uint32_t w1, uint32_t h1) {
        fprintf(stderr, "{\"width\": %u, \"height\": %u, \"frames\": %llu, \"dual_stream\": %s, \"bit_depth\": %d, \"format\": \"rgba\", \"step\": %llu",
                outW, outH, (unsigned long long)emitted, dual ? "true" : "false", bitDepth, (unsigned long long)step);
        if (dual)
            fprintf(stderr, ", \"track0_width\": %u, \"track0_height\": %u, \"track1_width\": %u, \"track1_height\": %u", w0, h0, w1, h1);
        fprintf(stderr, ", \"pipeline\": \"%s\"", g_activePipeline.c_str());
        if (g_activePipeline == "metal") fprintf(stderr, ", \"gpu_device\": \"%s\"", g_gpuDeviceName.c_str());
        fprintf(stderr, "}\n");
        fflush(stderr);
    };

    if (g_zcSlots > 0) {
        int decode_track = track;
        if (decode_track < 0 && ctx.videoTrackCount == 1 && ctx.multiVideo) decode_track = 0;
        return do_decode_zc(ctx, cb, clipOverrides, start, end, step, decode_track, is_dual);
    }

    if (is_dual) {
        // Two tracks per frame, written side by side (track 0 left). Tickets
        // are index·2 + track; both tracks of a frame are in flight together.
        if (!submit_read(ctx, start * 2 + 0, start, 0) || !submit_read(ctx, start * 2 + 1, start, 1)) return fail();
        FrameResult r0 = cb.wait(start * 2 + 0);
        FrameResult r1 = cb.wait(start * 2 + 1);
        if (r0.status != S_OK || r1.status != S_OK) {
            fprintf(stderr, "ERROR: Decode failed at frame %llu\n", (unsigned long long)start);
            return fail();
        }
        const uint32_t w0 = r0.w, h0 = r0.h, w1 = r1.w, h1 = r1.h;
        const uint32_t outH = std::max(h0, h1);
        const uint32_t outW = w0 + w1;
        header(outW, outH, true, w0, h0, w1, h1);

        const size_t rowBytes0 = (size_t)w0 * bytesPerPixel;
        const size_t rowBytes1 = (size_t)w1 * bytesPerPixel;
        const size_t outRowBytes = (size_t)outW * bytesPerPixel;
        std::vector<uint8_t> outRow(outRowBytes);
        const int ahead = frames_in_flight_for(rowBytes0 * h0 + rowBytes1 * h1);
        uint64_t next = start + step;
        int inflight = 0;
        auto top_up = [&]() -> bool {
            while (inflight < ahead && next < end) {
                if (!submit_read(ctx, next * 2 + 0, next, 0) || !submit_read(ctx, next * 2 + 1, next, 1)) return false;
                next += step; inflight++;
            }
            return true;
        };
        auto emit = [&](FrameResult& a, FrameResult& b, uint64_t i) -> bool {
            if (!check(a, i, w0, h0) || !check(b, i, w1, h1)) return false;
            const uint8_t* src0 = a.bytes.data();
            const uint8_t* src1 = b.bytes.data();
            for (uint32_t y = 0; y < outH; y++) {
                if (y < h0) memcpy(outRow.data(), src0 + y * rowBytes0, rowBytes0);
                else memset(outRow.data(), 0, rowBytes0);
                if (y < h1) memcpy(outRow.data() + rowBytes0, src1 + y * rowBytes1, rowBytes1);
                else memset(outRow.data() + rowBytes0, 0, rowBytes1);
                fwrite(outRow.data(), 1, outRowBytes, stdout);
            }
            fflush(stdout);
            cb.recycle(std::move(a.bytes));
            cb.recycle(std::move(b.bytes));
            return true;
        };
        if (!top_up()) return fail();
        if (!emit(r0, r1, start)) return fail();
        for (uint64_t i = start + step; i < end; i += step) {
            FrameResult a = cb.wait(i * 2 + 0);
            FrameResult b = cb.wait(i * 2 + 1);
            inflight--;
            if (!top_up()) return fail();
            if (!emit(a, b, i)) return fail();
        }
    } else {
        // Single-track (or explicit --track N)
        int decode_track = track;  // -1 means use default clip API
        // For single-track multi-video, still use the track API
        if (decode_track < 0 && ctx.videoTrackCount == 1 && ctx.multiVideo) decode_track = 0;

        // The first frame sizes the header.
        if (!submit_read(ctx, start, start, decode_track)) return fail();
        FrameResult first = cb.wait(start);
        if (first.status != S_OK) {
            fprintf(stderr, "ERROR: Decode failed at frame %llu (0x%08x)\n", (unsigned long long)start, (unsigned)first.status);
            return fail();
        }
        const uint32_t outW = first.w, outH = first.h;
        header(outW, outH, false, 0, 0, 0, 0);

        const size_t frameBytes = (size_t)outW * outH * bytesPerPixel;
        const int ahead = frames_in_flight_for(frameBytes);
        uint64_t next = start + step;
        int inflight = 0;
        auto top_up = [&]() -> bool {
            while (inflight < ahead && next < end) {
                if (!submit_read(ctx, next, next, decode_track)) return false;
                next += step; inflight++;
            }
            return true;
        };
        auto emit = [&](FrameResult& r, uint64_t i) -> bool {
            if (!check(r, i, outW, outH)) return false;
            fwrite(r.bytes.data(), 1, frameBytes, stdout);
            fflush(stdout);
            cb.recycle(std::move(r.bytes));
            return true;
        };
        // Submit ahead before writing, so the SDK works while we write.
        if (!top_up()) return fail();
        if (!emit(first, start)) return fail();
        for (uint64_t i = start + step; i < end; i += step) {
            FrameResult r = cb.wait(i);
            inflight--;
            if (!top_up()) return fail();
            if (!emit(r, i)) return fail();
        }
    }

    ctx.codec->FlushJobs();
    if (clipOverrides) clipOverrides->Release();
    return 0;
}

// ── Mode: --meta ─────────────────────────────────────────────────────────────
// Dump EVERY clip-level metadata entry, then every frame-level entry of
// frame N (default 0), as `key = value` lines on stdout. Arrays print as
// comma-separated lists. Used to learn what a camera/firmware writes
// (sensor line time, exposure, lens, ...) — nothing is interpreted here.

static std::string variant_repr(const Variant& v) {
    char buf[64];
    switch (v.vt) {
        case blackmagicRawVariantTypeEmpty: return "<empty>";
        case blackmagicRawVariantTypeU8: snprintf(buf, sizeof buf, "%u", (unsigned)(v.uiVal & 0xff)); return buf;
        case blackmagicRawVariantTypeS16: snprintf(buf, sizeof buf, "%d", (int)v.iVal); return buf;
        case blackmagicRawVariantTypeU16: snprintf(buf, sizeof buf, "%u", (unsigned)v.uiVal); return buf;
        case blackmagicRawVariantTypeS32: snprintf(buf, sizeof buf, "%d", (int)v.intVal); return buf;
        case blackmagicRawVariantTypeU32: snprintf(buf, sizeof buf, "%u", (unsigned)v.uintVal); return buf;
        case blackmagicRawVariantTypeFloat32: snprintf(buf, sizeof buf, "%.9g", (double)v.fltVal); return buf;
        case blackmagicRawVariantTypeFloat64: snprintf(buf, sizeof buf, "%.12g", v.dblVal); return buf;
        case blackmagicRawVariantTypeString: return "\"" + cfstr_to_string(v.bstrVal) + "\"";
        case blackmagicRawVariantTypeSafeArray: {
            SafeArray* a = v.parray; if (!a) return "<null array>";
            BlackmagicRawVariantType et; long lo = 0, hi = -1; void* data = nullptr;
            SafeArrayGetVartype(a, &et); SafeArrayGetLBound(a, 1, &lo); SafeArrayGetUBound(a, 1, &hi); SafeArrayAccessData(a, &data);
            std::string out = "[";
            for (long i = lo; i <= hi && data; i++) {
                long k = i - lo; if (k) out += ", ";
                switch (et) {
                    case blackmagicRawVariantTypeU8: snprintf(buf, sizeof buf, "%u", (unsigned)((uint8_t*)data)[k]); break;
                    case blackmagicRawVariantTypeS16: snprintf(buf, sizeof buf, "%d", (int)((int16_t*)data)[k]); break;
                    case blackmagicRawVariantTypeU16: snprintf(buf, sizeof buf, "%u", (unsigned)((uint16_t*)data)[k]); break;
                    case blackmagicRawVariantTypeS32: snprintf(buf, sizeof buf, "%d", (int)((int32_t*)data)[k]); break;
                    case blackmagicRawVariantTypeU32: snprintf(buf, sizeof buf, "%u", (unsigned)((uint32_t*)data)[k]); break;
                    case blackmagicRawVariantTypeFloat32: snprintf(buf, sizeof buf, "%.9g", (double)((float*)data)[k]); break;
                    case blackmagicRawVariantTypeFloat64: snprintf(buf, sizeof buf, "%.12g", ((double*)data)[k]); break;
                    default: snprintf(buf, sizeof buf, "?"); break;
                }
                out += buf;
                if (k > 4096) { out += ", ..."; break; }
            }
            snprintf(buf, sizeof buf, "] (type %d, %ld)", (int)et, hi - lo + 1);
            return out + buf;
        }
    }
    return "<?>";
}

static void dump_iterator(IBlackmagicRawMetadataIterator* it, const char* prefix) {
    if (!it) return;
    while (true) {
        CFStringRef key = nullptr; Variant v; VariantInit(&v);
        if (it->GetKey(&key) != S_OK || !key) break;
        std::string ks = cfstr_to_string(key);
        if (it->GetData(&v) == S_OK) printf("%s%s = %s\n", prefix, ks.c_str(), variant_repr(v).c_str());
        else printf("%s%s = <no data>\n", prefix, ks.c_str());
        VariantClear(&v);
        if (it->Next() != S_OK) break;
    }
}

class MetaCallback : public IBlackmagicRawCallback {
public:
    std::atomic<bool> done{false};
    void ReadComplete(IBlackmagicRawJob* readJob, HRESULT result, IBlackmagicRawFrame* frame) override {
        if (result == S_OK && frame) {
            IBlackmagicRawMetadataIterator* it = nullptr;
            if (frame->GetMetadataIterator(&it) == S_OK && it) { dump_iterator(it, "frame."); it->Release(); }
            uint64_t ts = 0; CFStringRef tc = nullptr;
            if (frame->GetTimecode(&tc) == S_OK && tc) { printf("frame.timecode = \"%s\"\n", cfstr_to_string(tc).c_str()); CFRelease(tc); }
            (void)ts;
        } else {
            fprintf(stderr, "ERROR: read frame failed (%ld)\n", (long)result);
        }
        done.store(true);
        readJob->Release();
    }
    void ReadAudioComplete(IBlackmagicRawJob* job, HRESULT, IBlackmagicRawAudioBuffer*) override { if (job) job->Release(); }
    void DecodeComplete(IBlackmagicRawJob*, HRESULT) override {}
    void ProcessComplete(IBlackmagicRawJob*, HRESULT, IBlackmagicRawProcessedImage*) override {}
    void TrimProgress(IBlackmagicRawJob*, float) override {}
    void TrimComplete(IBlackmagicRawJob*, HRESULT) override {}
    void SidecarMetadataParseWarning(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void SidecarMetadataParseError(IBlackmagicRawClip*, CFStringRef, uint32_t, CFStringRef) override {}
    void PreparePipelineComplete(void*, HRESULT) override {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, LPVOID*) override { return E_NOTIMPL; }
    ULONG STDMETHODCALLTYPE AddRef() override { return 0; }
    ULONG STDMETHODCALLTYPE Release() override { return 0; }
};

static int do_meta(const char* path, uint64_t frameIndex) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;
    uint32_t w = 0, h = 0; uint64_t n = 0; float fps = 0;
    ctx.clip->GetWidth(&w); ctx.clip->GetHeight(&h); ctx.clip->GetFrameCount(&n); ctx.clip->GetFrameRate(&fps);
    printf("clip.width = %u\nclip.height = %u\nclip.frame_count = %llu\nclip.frame_rate = %.6f\nclip.video_tracks = %u\n",
           w, h, (unsigned long long)n, fps, ctx.videoTrackCount);
    CFStringRef tc = nullptr;
    if (ctx.clip->GetTimecodeForFrame(0, &tc) == S_OK && tc) { printf("clip.timecode_frame0 = \"%s\"\n", cfstr_to_string(tc).c_str()); CFRelease(tc); }
    IBlackmagicRawMetadataIterator* it = nullptr;
    if (ctx.clip->GetMetadataIterator(&it) == S_OK && it) { dump_iterator(it, "clip."); it->Release(); }
    MetaCallback cb;
    ctx.codec->SetCallback(&cb);
    IBlackmagicRawJob* job = nullptr;
    if (ctx.clip->CreateJobReadFrame(frameIndex, &job) == S_OK && job) {
        printf("frame.index = %llu\n", (unsigned long long)frameIndex);
        if (job->Submit() == S_OK) ctx.codec->FlushJobs(); else job->Release();
    }
    fflush(stdout);
    return 0;
}

// ── Mode: --gyro ─────────────────────────────────────────────────────────────

static int do_gyro(const char* path) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;

    float frameRate = 0;
    ctx.clip->GetFrameRate(&frameRate);

    IBlackmagicRawClipGyroscopeMotion* gyro = nullptr;
    IBlackmagicRawClipAccelerometerMotion* accel = nullptr;

    HRESULT result = ctx.clip->QueryInterface(IID_IBlackmagicRawClipGyroscopeMotion, (void**)&gyro);
    if (result != S_OK || !gyro) {
        std::cerr << "ERROR: No gyroscope data in clip." << std::endl;
        return 1;
    }

    result = ctx.clip->QueryInterface(IID_IBlackmagicRawClipAccelerometerMotion, (void**)&accel);
    if (result != S_OK || !accel) {
        std::cerr << "ERROR: No accelerometer data in clip." << std::endl;
        gyro->Release();
        return 1;
    }

    uint32_t gyroCount = 0, accelCount = 0;
    float gyroRate = 0, accelRate = 0;
    uint32_t gyroSampleSize = 0, accelSampleSize = 0;

    gyro->GetSampleCount(&gyroCount);
    gyro->GetSampleRate(&gyroRate);
    gyro->GetSampleSize(&gyroSampleSize);
    accel->GetSampleCount(&accelCount);
    accel->GetSampleRate(&accelRate);
    accel->GetSampleSize(&accelSampleSize);

    uint32_t sampleCount = std::min(gyroCount, accelCount);
    if (gyroSampleSize != 3 || accelSampleSize != 3) {
        fprintf(stderr, "ERROR: motion sample size %u / %u floats (expected 3)\n", gyroSampleSize, accelSampleSize);
        gyro->Release(); accel->Release();
        return 1;
    }

    // Write header to stderr
    fprintf(stderr, "{\"gyro_sample_count\": %u, \"gyro_sample_rate\": %.2f, "
            "\"accel_sample_count\": %u, \"accel_sample_rate\": %.2f, "
            "\"sample_count\": %u, \"frame_rate\": %.6f}\n",
            gyroCount, gyroRate, accelCount, accelRate, sampleCount, frameRate);
    fflush(stderr);

    // Read and output in chunks
    const uint32_t CHUNK = 2000;
    float* gyroBuf = new float[CHUNK * 3];
    float* accelBuf = new float[CHUNK * 3];
    float outBuf[6]; // interleaved: gx, gy, gz, ax, ay, az

    uint64_t offset = 0;
    uint32_t remaining = sampleCount;

    while (remaining > 0) {
        uint32_t toRead = std::min(CHUNK, remaining);
        uint32_t gyroRead = 0, accelRead = 0;

        HRESULT rg = gyro->GetSampleRange(offset, toRead, gyroBuf, &gyroRead);
        HRESULT ra = accel->GetSampleRange(offset, toRead, accelBuf, &accelRead);

        uint32_t n = std::min(gyroRead, accelRead);
        if (rg != S_OK || ra != S_OK || n == 0) {
            // A short or failed read must not spin forever with stdout open.
            fprintf(stderr, "ERROR: motion read at sample %llu: gyro %ld (%u), accel %ld (%u) — %u of %u samples written\n",
                    (unsigned long long)offset, (long)rg, gyroRead, (long)ra, accelRead, (unsigned)(sampleCount - remaining), sampleCount);
            fflush(stdout);
            delete[] gyroBuf; delete[] accelBuf; gyro->Release(); accel->Release();
            return 1;
        }
        for (uint32_t i = 0; i < n; i++) {
            outBuf[0] = gyroBuf[i * 3 + 0];
            outBuf[1] = gyroBuf[i * 3 + 1];
            outBuf[2] = gyroBuf[i * 3 + 2];
            outBuf[3] = accelBuf[i * 3 + 0];
            outBuf[4] = accelBuf[i * 3 + 1];
            outBuf[5] = accelBuf[i * 3 + 2];
            fwrite(outBuf, sizeof(float), 6, stdout);
        }

        offset += n;
        remaining -= n;
    }

    fflush(stdout);
    delete[] gyroBuf;
    delete[] accelBuf;
    gyro->Release();
    accel->Release();

    return 0;
}

// ── Mode: --audio ────────────────────────────────────────────────────────────
// Extract embedded audio from a BRAW clip and write a RIFF/WAV stream to
// stdout. JSON header (sampleRate, bitDepth, channelCount, sampleCount) is
// written to stderr.

#pragma pack(push, 1)
struct WavHeader {
    char     riff[4] = { 'R','I','F','F' };
    uint32_t riffSize = 0;     // 4 + (8 + fmtChunkSize) + (8 + dataBytes)
    char     wave[4] = { 'W','A','V','E' };
    char     fmt[4]  = { 'f','m','t',' ' };
    uint32_t fmtSize = 16;     // PCM
    uint16_t audioFormat = 1;  // PCM
    uint16_t channelCount = 0;
    uint32_t sampleRate = 0;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = 0;
    char     data[4] = { 'd','a','t','a' };
    uint32_t dataSize = 0;
};
#pragma pack(pop)

static int do_audio(const char* path) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;

    IBlackmagicRawClipAudio* audio = nullptr;
    HRESULT result = ctx.clip->QueryInterface(IID_IBlackmagicRawClipAudio, (void**)&audio);
    if (result != S_OK || !audio) {
        fprintf(stderr, "{\"error\": \"no audio track\"}\n");
        return 1;
    }

    uint64_t sampleCount = 0;
    uint32_t bitDepth = 0;
    uint32_t channelCount = 0;
    uint32_t sampleRate = 0;

    if (audio->GetAudioSampleCount(&sampleCount) != S_OK ||
        audio->GetAudioBitDepth(&bitDepth) != S_OK ||
        audio->GetAudioChannelCount(&channelCount) != S_OK ||
        audio->GetAudioSampleRate(&sampleRate) != S_OK) {
        fprintf(stderr, "{\"error\": \"failed to read audio metadata\"}\n");
        audio->Release();
        return 1;
    }

    if (sampleCount == 0 || channelCount == 0 || sampleRate == 0 || bitDepth == 0) {
        fprintf(stderr, "{\"error\": \"empty audio track\"}\n");
        audio->Release();
        return 1;
    }

    uint64_t dataBytes64 = (sampleCount * channelCount * bitDepth) / 8;
    // RIFF/WAV dataSize is a 32-bit field — clamp for the header, though the
    // samples are still streamed in full. For any sane BRAW clip this fits.
    uint32_t dataBytes = (dataBytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : (uint32_t)dataBytes64;

    WavHeader hdr;
    hdr.channelCount  = (uint16_t)channelCount;
    hdr.sampleRate    = sampleRate;
    hdr.byteRate      = sampleRate * channelCount * bitDepth / 8;
    hdr.blockAlign    = (uint16_t)((channelCount * bitDepth) / 8);
    hdr.bitsPerSample = (uint16_t)bitDepth;
    hdr.dataSize      = dataBytes;
    hdr.riffSize      = 36 + dataBytes;

    // Metadata to stderr first, then raw WAV to stdout
    fprintf(stderr,
            "{\"sample_rate\": %u, \"bit_depth\": %u, \"channel_count\": %u, "
            "\"sample_count\": %llu, \"data_bytes\": %llu}\n",
            sampleRate, bitDepth, channelCount,
            (unsigned long long)sampleCount, (unsigned long long)dataBytes64);
    fflush(stderr);

    fwrite(&hdr, sizeof(hdr), 1, stdout);

    // Stream samples in chunks
    const uint32_t maxPerCall = 48000;  // 1 second @ 48 kHz
    uint32_t bufBytes = (maxPerCall * channelCount * bitDepth) / 8;
    uint8_t* buf = new uint8_t[bufBytes];

    int64_t sampleIdx = 0;
    while ((uint64_t)sampleIdx < sampleCount) {
        uint32_t samplesRead = 0;
        uint32_t bytesRead = 0;
        result = audio->GetAudioSamples(sampleIdx, buf, bufBytes, maxPerCall,
                                         &samplesRead, &bytesRead);
        if (result != S_OK || samplesRead == 0) break;
        fwrite(buf, 1, bytesRead, stdout);
        sampleIdx += samplesRead;
    }

    fflush(stdout);
    delete[] buf;
    audio->Release();
    return 0;
}


// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "  braw_helper --info  <file.braw>" << std::endl;
        std::cerr << "  braw_helper --decode <file.braw> [--start N] [--count M] [--track T] [--16bit]" << std::endl;
        std::cerr << "              [--scale full|half|quarter|eighth] [--look film|video|extended|rec709]" << std::endl;
        std::cerr << "              [--pipeline auto|metal|cpu]  (auto = the first Metal device, else CPU)" << std::endl;
        std::cerr << "              [--step N]  (every Nth frame from --start; the header's frames = emitted count)" << std::endl;
        std::cerr << "              [--zerocopy N] [--mono]  (N slots of global IOSurfaces: \"F <frame> <slot>\" per frame on stdout," << std::endl;
        std::cerr << "               \"R <slot>\" on stdin returns one; surface IDs + eye size in the header; 8-bit only)" << std::endl;
        std::cerr << "              (JSON header on stderr, then raw RGBA frames on stdout)" << std::endl;
        std::cerr << "              [--gamma S] [--gamut S] [--wb K] [--tint T] [--exposure EV] [--iso N]" << std::endl;
        std::cerr << "              [--tc-contrast|--tc-saturation|--tc-midpoint|--tc-highlights|--tc-shadows|" << std::endl;
        std::cerr << "               --tc-black-level|--tc-white-level F] [--highlight-recovery 0|1]" << std::endl;
        std::cerr << "              [--analog-gain F] [--gamut-compression 0|1]" << std::endl;
        std::cerr << "  braw_helper --gyro  <file.braw>  (JSON header on stderr, float32 samples on stdout)" << std::endl;
        std::cerr << "  braw_helper --audio <file.braw>  (JSON header on stderr, WAV on stdout)" << std::endl;
        std::cerr << "  braw_helper --meta  <file.braw> [--frame N]   (all clip + frame metadata)" << std::endl;
        std::cerr << "  braw_helper --shutters <file.braw> [--step N]  (\"<frame> <seconds> \\\"<shutter_value>\\\"\" per frame)" << std::endl;
        std::cerr << "  braw_helper --lut   <file.braw> <out.cube>    (embedded post-3D LUT as .cube)" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    const char* filePath = argv[2];

    // Parse flags (no value) and key-value args. Later arguments win, so
    // `--look rec709 --gamma X` ends up with gamma X on the rec709 gamut.
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        // Flags without values
        if (arg == "--16bit") { g_use16bit = true; continue; }
        if (arg == "--mono") { g_mono = true; continue; }
        // Key-value pairs (need next arg as value)
        if (i + 1 >= argc) break;
        std::string val = argv[i + 1];
        if (arg == "--gamma") g_procParams.gamma = val;
        else if (arg == "--gamut") g_procParams.gamut = val;
        else if (arg == "--look") {
            if (!apply_look_preset(val)) { std::cerr << "ERROR: unknown --look " << val << " (film|video|extended|rec709)" << std::endl; return 1; }
        }
        else if (arg == "--scale") {
            if (!parse_scale(val, &g_scale)) { std::cerr << "ERROR: unknown --scale " << val << " (full|half|quarter|eighth)" << std::endl; return 1; }
        }
        else if (arg == "--pipeline") {
            if (val != "auto" && val != "metal" && val != "cpu") { std::cerr << "ERROR: unknown --pipeline " << val << " (auto|metal|cpu)" << std::endl; return 1; }
            g_pipelineArg = val;
        }
        else if (arg == "--step") { g_step = std::max<uint64_t>(1, std::stoull(val)); }
        else if (arg == "--zerocopy") { g_zcSlots = (uint32_t)std::stoul(val); }
        else if (arg == "--wb") g_procParams.whiteBalanceKelvin = std::stoi(val);
        else if (arg == "--tint") g_procParams.whiteBalanceTint = std::stoi(val);
        else if (arg == "--exposure") g_procParams.exposure = std::stof(val);
        else if (arg == "--iso") g_procParams.iso = std::stoi(val);
        else if (arg == "--tc-contrast") g_procParams.toneCurveContrast = std::stof(val);
        else if (arg == "--tc-saturation") g_procParams.toneCurveSaturation = std::stof(val);
        else if (arg == "--tc-midpoint") g_procParams.toneCurveMidpoint = std::stof(val);
        else if (arg == "--tc-highlights") g_procParams.toneCurveHighlights = std::stof(val);
        else if (arg == "--tc-shadows") g_procParams.toneCurveShadows = std::stof(val);
        else if (arg == "--tc-black-level") g_procParams.toneCurveBlackLevel = std::stof(val);
        else if (arg == "--tc-white-level") g_procParams.toneCurveWhiteLevel = std::stof(val);
        else if (arg == "--highlight-recovery") g_procParams.highlightRecovery = std::stoi(val);
        else if (arg == "--analog-gain") g_procParams.analogGain = std::stof(val);
        else if (arg == "--gamut-compression") g_procParams.gamutCompression = std::stoi(val);
        else continue;  // unknown key-value pair, don't skip value
        i++;  // skip the value arg
    }

    if (mode == "--info") {
        return do_info(filePath);
    } else if (mode == "--decode") {
        uint64_t start = 0, count = 0;
        int track = -1;  // -1 = auto (dual SBS if multi-video, single otherwise)
        for (int i = 3; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--start") start = std::stoull(argv[i + 1]);
            else if (std::string(argv[i]) == "--count") count = std::stoull(argv[i + 1]);
            else if (std::string(argv[i]) == "--track") track = std::stoi(argv[i + 1]);
        }
        return do_decode(filePath, start, count, track);
    } else if (mode == "--meta") {
        uint64_t fi = 0;
        for (int i = 3; i < argc - 1; i++) if (std::string(argv[i]) == "--frame") fi = std::stoull(argv[i + 1]);
        return do_meta(filePath, fi);
    } else if (mode == "--gyro") {
        return do_gyro(filePath);
    } else if (mode == "--audio") {
        return do_audio(filePath);
    } else if (mode == "--shutters") {
        uint64_t step = 1;
        for (int i = 3; i < argc - 1; i++) if (std::string(argv[i]) == "--step") step = std::stoull(argv[i + 1]);
        return do_shutters(filePath, step);
    } else if (mode == "--lut") {
        if (argc < 4) { std::cerr << "ERROR: --lut needs an output path: --lut <file.braw> <out.cube>" << std::endl; return 1; }
        return do_lut(filePath, argv[3]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}
