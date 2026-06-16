/*
 * braw_helper.cpp — Blackmagic RAW decode helper for vr180_fisheye_converter
 *
 * Modes:
 *   --info  <file.braw>                                          → JSON clip metadata to stdout
 *   --decode <file.braw> [--start N] [--count M] [--track T]     → raw BGRA frames to stdout
 *   --gyro  <file.braw>                                          → binary gyro+accel to stdout, JSON header to stderr
 *
 * Multi-video (e.g. URSA Cine Immersive stereo):
 *   --info reports video_track_count > 1 and per-track dimensions
 *   --decode with --track 0 or --track 1 decodes a single track
 *   --decode without --track on a multi-video clip decodes all tracks,
 *     writing them side-by-side (left=track0, right=track1) per frame
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

static std::string variant_to_string(const Variant& v) {
    switch (v.vt) {
        case blackmagicRawVariantTypeU8:    return std::to_string((uint8_t)v.iVal);
        case blackmagicRawVariantTypeS16:   return std::to_string(v.iVal);
        case blackmagicRawVariantTypeU16:   return std::to_string(v.uiVal);
        case blackmagicRawVariantTypeS32:   return std::to_string(v.intVal);
        case blackmagicRawVariantTypeU32:   return std::to_string(v.uintVal);
        case blackmagicRawVariantTypeFloat32: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.8g", (double)v.fltVal);
            return buf;
        }
        case blackmagicRawVariantTypeFloat64: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.12g", v.dblVal);
            return buf;
        }
        case blackmagicRawVariantTypeString:
            return "\"" + cfstr_to_string(v.bstrVal) + "\"";
        default: return "null";
    }
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
    VariantClear(&v);
    return val;
}

// ── BRAW processing overrides ────────────────────────────────────────────────

struct BrawProcessingParams {
    // Clip-level
    std::string gamma;      // e.g. "Rec.709", "Blackmagic Design Film", ""=default
    std::string gamut;      // e.g. "Rec.709", "Blackmagic Design", ""=default
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

// Apply clip-level overrides to a cloned clip processing attributes object.
// Returns the cloned+modified object (caller must Release), or nullptr if no overrides.
static IBlackmagicRawClipProcessingAttributes* create_clip_overrides(IBlackmagicRawClip* clip) {
    // Check if we have any clip-level overrides
    bool hasOverrides = !g_procParams.gamma.empty() || !g_procParams.gamut.empty() ||
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

    if (!g_procParams.gamma.empty()) {
        CFStringRef cfGamma = CFStringCreateWithCString(NULL, g_procParams.gamma.c_str(), kCFStringEncodingUTF8);
        Variant v; v.vt = blackmagicRawVariantTypeString; v.bstrVal = cfGamma;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeGamma, &v);
        CFRelease(cfGamma);
    }
    if (!g_procParams.gamut.empty()) {
        CFStringRef cfGamut = CFStringCreateWithCString(NULL, g_procParams.gamut.c_str(), kCFStringEncodingUTF8);
        Variant v; v.vt = blackmagicRawVariantTypeString; v.bstrVal = cfGamut;
        attrs->SetClipAttribute(blackmagicRawClipProcessingAttributeGamut, &v);
        CFRelease(cfGamut);
    }
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

    // Sensor readout info from clip-level metadata
    double sensorAreaHeight = get_metadata_float(ctx.clip, "sensor_area_captured_height", 0);
    double sensorLineTime = get_metadata_float(ctx.clip, "sensor_line_time", 0);

    double readoutMs = 0;
    if (sensorAreaHeight > 0 && sensorLineTime > 0)
        readoutMs = (sensorAreaHeight * sensorLineTime) / 1000.0;

    // Firmware v7.9 bug: readout time doubled
    if (firmware == "7.9" && frameRate > 0 && readoutMs > (1000.0 / frameRate))
        readoutMs /= 2.0;

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
    printf("  \"readout_ms\": %.6f\n", readoutMs);
    printf("}\n");

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
            auto setStr = [&](uint32_t attr, const std::string& val) {
                if (!val.empty()) { CFStringRef cf = CFStringCreateWithCString(NULL, val.c_str(), kCFStringEncodingUTF8); Variant v; v.vt = blackmagicRawVariantTypeString; v.bstrVal = cf; attrs->SetClipAttribute((BlackmagicRawClipProcessingAttribute)attr, &v); CFRelease(cf); }
            };
            setStr(0x67616D61, g_procParams.gamma);
            setStr(0x67616D74, g_procParams.gamut);
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
                "\"dual_stream\": true, \"track0_width\": %u, \"track0_height\": %u, "
                "\"track1_width\": %u, \"track1_height\": %u}\n",
                outW, outH, (unsigned long long)(end - start), w0, h0, w1, h1);
        fflush(stderr);

        const size_t rowBytes0 = (size_t)w0 * 4;
        const size_t rowBytes1 = (size_t)w1 * 4;
        const size_t outRowBytes = (size_t)outW * 4;
        std::vector<uint8_t> outRow(outRowBytes);

        for (uint64_t i = start; i < end; i++) {
            // Decode track 0
            if (!decode_one_frame(ctx, cb, i, 0)) return 1;
            uint32_t sz0 = 0; void* data0 = nullptr;
            cb.processedImage->GetResourceSizeBytes(&sz0);
            cb.processedImage->GetResource(&data0);

            // Copy track 0 data before reset (SDK may reuse buffer)
            std::vector<uint8_t> buf0((uint8_t*)data0, (uint8_t*)data0 + sz0);
            cb.reset();

            // Decode track 1
            if (!decode_one_frame(ctx, cb, i, 1)) return 1;
            uint32_t sz1 = 0; void* data1 = nullptr;
            cb.processedImage->GetResourceSizeBytes(&sz1);
            cb.processedImage->GetResource(&data1);

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
                outW, outH, (unsigned long long)(end - start), g_use16bit ? 16 : 8);
        fflush(stderr);

        const size_t bytesPerPixel = g_use16bit ? 8 : 4;  // BGRA U16 = 8, U8 = 4
        const size_t bgraSize = (size_t)outW * outH * bytesPerPixel;

        // Write first already-decoded frame
        {
            uint32_t sizeBytes = 0; void* data = nullptr;
            cb.processedImage->GetResourceSizeBytes(&sizeBytes);
            cb.processedImage->GetResource(&data);
            if (data && sizeBytes >= bgraSize) {
                fwrite(data, 1, bgraSize, stdout);
                fflush(stdout);
            } else {
                std::cerr << "ERROR: Invalid frame data at frame " << start << std::endl;
                return 1;
            }
        }

        // Decode remaining frames
        for (uint64_t i = start + 1; i < end; i++) {
            if (!decode_one_frame(ctx, cb, i, decode_track)) return 1;

            uint32_t sizeBytes = 0; void* data = nullptr;
            cb.processedImage->GetResourceSizeBytes(&sizeBytes);
            cb.processedImage->GetResource(&data);

            if (data && sizeBytes >= bgraSize) {
                fwrite(data, 1, bgraSize, stdout);
                fflush(stdout);
            } else {
                std::cerr << "ERROR: Invalid frame data at frame " << i << std::endl;
                return 1;
            }
        }
    }

    if (clipOverrides) clipOverrides->Release();
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
        std::cerr << "  braw_helper --decode <file.braw> [--start N] [--count M] [--track T]" << std::endl;
        std::cerr << "  braw_helper --gyro  <file.braw>" << std::endl;
        std::cerr << "  braw_helper --audio <file.braw>  (WAV to stdout)" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    const char* filePath = argv[2];

    // Parse flags (no value) and key-value args
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        // Flags without values
        if (arg == "--16bit") { g_use16bit = true; continue; }
        // Key-value pairs (need next arg as value)
        if (i + 1 >= argc) break;
        std::string val = argv[i + 1];
        if (arg == "--gamma") g_procParams.gamma = val;
        else if (arg == "--gamut") g_procParams.gamut = val;
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
    } else if (mode == "--gyro") {
        return do_gyro(filePath);
    } else if (mode == "--audio") {
        return do_audio(filePath);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}
