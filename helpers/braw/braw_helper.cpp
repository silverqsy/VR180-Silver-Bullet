/*
 * braw_helper.cpp — Blackmagic RAW decode helper for vr180_fisheye_converter
 *
 * Modes:
 *   --info  <file.braw>                                          → JSON clip metadata to stdout
 *   --decode <file.braw> [--start N] [--count M] [--track T]     → raw BGRA frames to stdout
 *            [--16bit] [--scale full|half|quarter|eighth]          (JSON header on stderr first)
 *            [--look film|video|extended|rec709] [--gamma S] [--gamut S] [--wb K] ...
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

// ── Frame decode callback ────────────────────────────────────────────────────

class DecodeCallback : public IBlackmagicRawCallback {
public:
    std::atomic<bool> done{false};
    std::atomic<HRESULT> status{S_OK};
    IBlackmagicRawProcessedImage* processedImage = nullptr;
    IBlackmagicRawClipProcessingAttributes* clipOverrides = nullptr;  // set before decode

    void ReadComplete(IBlackmagicRawJob* readJob, HRESULT result, IBlackmagicRawFrame* frame) override {
        IBlackmagicRawJob* decodeJob = nullptr;
        if (result == S_OK)
            VERIFY(frame->SetResourceFormat(g_use16bit ? blackmagicRawResourceFormatBGRAU16 : blackmagicRawResourceFormatBGRAU8));
        if (result == S_OK)
            result = frame->SetResolutionScale(g_scale);
        if (result == S_OK)
            apply_frame_overrides(frame);
        if (result == S_OK)
            result = frame->CreateJobDecodeAndProcessFrame(clipOverrides, nullptr, &decodeJob);
        if (result == S_OK)
            result = decodeJob->Submit();
        if (result != S_OK) {
            if (decodeJob) decodeJob->Release();
            status.store(result);
            done.store(true);
        }
        readJob->Release();
    }

    void ProcessComplete(IBlackmagicRawJob* job, HRESULT result, IBlackmagicRawProcessedImage* img) override {
        if (result == S_OK && img) {
            processedImage = img;
            processedImage->AddRef();
        }
        status.store(result);
        done.store(true);
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

    void reset() {
        if (processedImage) { processedImage->Release(); processedImage = nullptr; }
        done.store(false);
        status.store(S_OK);
    }
};

// ── Open clip helper ─────────────────────────────────────────────────────────

struct ClipContext {
    IBlackmagicRawFactory* factory = nullptr;
    IBlackmagicRaw* codec = nullptr;
    IBlackmagicRawClip* clip = nullptr;
    IBlackmagicRawClipMultiVideo* multiVideo = nullptr;  // non-null if multi-video
    uint32_t videoTrackCount = 1;

    bool open(const char* path) {
        factory = CreateBlackmagicRawFactoryInstanceFromPath(
            CFStringCreateWithCString(NULL, SDK_LIB_PATH, kCFStringEncodingUTF8));
        if (!factory) { std::cerr << "ERROR: Cannot create BRAW factory. SDK not found." << std::endl; return false; }

        if (factory->CreateCodec(&codec) != S_OK) { std::cerr << "ERROR: Cannot create codec." << std::endl; return false; }

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
        if (codec) { codec->Release(); codec = nullptr; }
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

    // The codec's decode pipeline (whatever the SDK picked for this process;
    // we never call SetPipeline).
    std::string pipeline = "unknown";
    {
        IBlackmagicRawConfiguration* cfg = nullptr;
        if (ctx.codec->QueryInterface(IID_IBlackmagicRawConfiguration, (void**)&cfg) == S_OK && cfg) {
            BlackmagicRawPipeline p = (BlackmagicRawPipeline)0;
            if (cfg->GetPipeline(&p, nullptr, nullptr) == S_OK) {
                switch (p) {
                    case blackmagicRawPipelineCPU:    pipeline = "cpu"; break;
                    case blackmagicRawPipelineMetal:  pipeline = "metal"; break;
                    case blackmagicRawPipelineCUDA:   pipeline = "cuda"; break;
                    case blackmagicRawPipelineOpenCL: pipeline = "opencl"; break;
                    default: {
                        char fourcc[5] = { (char)(p >> 24), (char)(p >> 16), (char)(p >> 8), (char)p, 0 };
                        pipeline = fourcc;
                    }
                }
            }
            cfg->Release();
        }
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
    printf("  \"pipeline\": \"%s\",\n", pipeline.c_str());
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

// Decode a single frame from a specific track (or default track)
// Returns true on success, fills processedImage in cb
static bool decode_one_frame(ClipContext& ctx, DecodeCallback& cb,
                              uint64_t frameIndex, int track = -1) {
    cb.reset();

    IBlackmagicRawJob* readJob = nullptr;
    HRESULT result;

    if (track >= 0 && ctx.multiVideo) {
        result = ctx.multiVideo->CreateJobReadFrame((uint32_t)track, frameIndex, &readJob);
    } else {
        result = ctx.clip->CreateJobReadFrame(frameIndex, &readJob);
    }

    if (result != S_OK) {
        std::cerr << "ERROR: CreateJobReadFrame failed at frame " << frameIndex
                  << " track " << track << std::endl;
        return false;
    }

    // Reader hint: at reduced scale the SDK can skip reading the detail
    // levels it will not decode (the frame-level SetResolutionScale in
    // ReadComplete is what actually sizes the output).
    set_read_hint(readJob, g_scale);

    result = readJob->Submit();
    if (result != S_OK) {
        readJob->Release();
        std::cerr << "ERROR: Submit failed at frame " << frameIndex << std::endl;
        return false;
    }

    ctx.codec->FlushJobs();

    if (!cb.done.load() || cb.status.load() != S_OK || !cb.processedImage) {
        std::cerr << "ERROR: Decode failed at frame " << frameIndex << std::endl;
        return false;
    }

    return true;
}

static int do_decode(const char* path, uint64_t start, uint64_t count, int track) {
    ClipContext ctx;
    if (!ctx.open(path)) return 1;

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

    uint32_t width = 0, height = 0;
    uint64_t frameCount = 0;
    ctx.clip->GetWidth(&width);
    ctx.clip->GetHeight(&height);
    ctx.clip->GetFrameCount(&frameCount);

    bool is_dual = ctx.videoTrackCount >= 2 && track < 0;

    if (start >= frameCount) {
        std::cerr << "ERROR: start frame " << start << " >= frame count " << frameCount << std::endl;
        return 1;
    }

    uint64_t end = (count == 0) ? frameCount : std::min(start + count, frameCount);

    DecodeCallback cb;
    cb.clipOverrides = clipOverrides;  // may be nullptr if no overrides
    ctx.codec->SetCallback(&cb);

    const size_t bytesPerPixel = g_use16bit ? 8 : 4;  // BGRA U16 = 8, U8 = 4
    const int bitDepth = g_use16bit ? 16 : 8;

    // The frame the SDK hands back must be exactly the tightly packed
    // w × h × bpp we announced in the header (the consumer reads frames by
    // that byte count). A frame that differs in size (a mid-clip scale
    // change) or in buffer size (row padding) is a hard error rather than
    // a silently garbled stream.
    auto check_frame = [&](uint64_t i, uint32_t w, uint32_t h, void** data, size_t* bytes) -> bool {
        uint32_t fw = 0, fh = 0, sz = 0; void* p = nullptr;
        cb.processedImage->GetWidth(&fw);
        cb.processedImage->GetHeight(&fh);
        cb.processedImage->GetResourceSizeBytes(&sz);
        cb.processedImage->GetResource(&p);
        const size_t want = (size_t)w * h * bytesPerPixel;
        if (!p || fw != w || fh != h || sz != want) {
            fprintf(stderr, "ERROR: frame %llu is %ux%u, %u bytes; expected %ux%u, %zu bytes\n",
                    (unsigned long long)i, fw, fh, sz, w, h, want);
            return false;
        }
        *data = p; *bytes = want;
        return true;
    };

    if (is_dual) {
        // Dual-track: decode both tracks per frame, write side-by-side BGRA
        // Each track has its own dimensions — decode track 0 and track 1

        // Decode frame 0 of each track to discover per-track dimensions
        // (main clip width/height may be the combined or single-track size)
        if (!decode_one_frame(ctx, cb, start, 0)) return 1;
        uint32_t w0 = 0, h0 = 0;
        cb.processedImage->GetWidth(&w0);
        cb.processedImage->GetHeight(&h0);
        cb.reset();

        if (!decode_one_frame(ctx, cb, start, 1)) return 1;
        uint32_t w1 = 0, h1 = 0;
        cb.processedImage->GetWidth(&w1);
        cb.processedImage->GetHeight(&h1);
        cb.reset();

        uint32_t outH = std::max(h0, h1);
        uint32_t outW = w0 + w1;

        fprintf(stderr, "{\"width\": %u, \"height\": %u, \"frames\": %llu, "
                "\"dual_stream\": true, \"bit_depth\": %d, \"track0_width\": %u, \"track0_height\": %u, "
                "\"track1_width\": %u, \"track1_height\": %u}\n",
                outW, outH, (unsigned long long)(end - start), bitDepth, w0, h0, w1, h1);
        fflush(stderr);

        const size_t rowBytes0 = (size_t)w0 * bytesPerPixel;
        const size_t rowBytes1 = (size_t)w1 * bytesPerPixel;
        const size_t outRowBytes = (size_t)outW * bytesPerPixel;
        std::vector<uint8_t> outRow(outRowBytes);
        std::vector<uint8_t> buf0;

        for (uint64_t i = start; i < end; i++) {
            // Decode track 0
            if (!decode_one_frame(ctx, cb, i, 0)) return 1;
            void* data0 = nullptr; size_t sz0 = 0;
            if (!check_frame(i, w0, h0, &data0, &sz0)) return 1;

            // Copy track 0 data before reset (SDK may reuse buffer)
            buf0.assign((uint8_t*)data0, (uint8_t*)data0 + sz0);
            cb.reset();

            // Decode track 1
            if (!decode_one_frame(ctx, cb, i, 1)) return 1;
            void* data1 = nullptr; size_t sz1 = 0;
            if (!check_frame(i, w1, h1, &data1, &sz1)) return 1;

            // Write side-by-side: track0 left, track1 right, row by row
            const uint8_t* src0 = buf0.data();
            const uint8_t* src1 = (const uint8_t*)data1;
            for (uint32_t y = 0; y < outH; y++) {
                // Track 0 pixels (left half)
                if (y < h0) {
                    memcpy(outRow.data(), src0 + y * rowBytes0, rowBytes0);
                } else {
                    memset(outRow.data(), 0, rowBytes0);
                }
                // Track 1 pixels (right half)
                if (y < h1) {
                    memcpy(outRow.data() + rowBytes0, src1 + y * rowBytes1, rowBytes1);
                } else {
                    memset(outRow.data() + rowBytes0, 0, rowBytes1);
                }
                fwrite(outRow.data(), 1, outRowBytes, stdout);
            }
            fflush(stdout);
        }
    } else {
        // Single-track (or explicit --track N)
        int decode_track = track;  // -1 means use default clip API

        // For single-track multi-video, still use the track API
        if (decode_track < 0 && ctx.videoTrackCount == 1 && ctx.multiVideo) {
            decode_track = 0;
        }

        // Get actual output dimensions by decoding first frame
        if (!decode_one_frame(ctx, cb, start, decode_track)) return 1;
        uint32_t outW = 0, outH = 0;
        cb.processedImage->GetWidth(&outW);
        cb.processedImage->GetHeight(&outH);

        fprintf(stderr, "{\"width\": %u, \"height\": %u, \"frames\": %llu, \"dual_stream\": false, \"bit_depth\": %d}\n",
                outW, outH, (unsigned long long)(end - start), bitDepth);
        fflush(stderr);

        // Write the first (already decoded) frame, then the rest
        for (uint64_t i = start; i < end; i++) {
            if (i != start && !decode_one_frame(ctx, cb, i, decode_track)) return 1;
            void* data = nullptr; size_t bytes = 0;
            if (!check_frame(i, outW, outH, &data, &bytes)) return 1;
            fwrite(data, 1, bytes, stdout);
            fflush(stdout);
        }
    }

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

        gyro->GetSampleRange(offset, toRead, gyroBuf, &gyroRead);
        accel->GetSampleRange(offset, toRead, accelBuf, &accelRead);

        uint32_t n = std::min(gyroRead, accelRead);
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
        std::cerr << "              [--gamma S] [--gamut S] [--wb K] [--tint T] [--exposure EV] [--iso N]" << std::endl;
        std::cerr << "              [--tc-contrast|--tc-saturation|--tc-midpoint|--tc-highlights|--tc-shadows|" << std::endl;
        std::cerr << "               --tc-black-level|--tc-white-level F] [--highlight-recovery 0|1]" << std::endl;
        std::cerr << "              [--analog-gain F] [--gamut-compression 0|1]" << std::endl;
        std::cerr << "              (JSON header on stderr, then raw BGRA frames on stdout)" << std::endl;
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
