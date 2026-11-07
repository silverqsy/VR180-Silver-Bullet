// braw_platform.h — the platform shim under braw_helper.cpp.
//
// The Blackmagic RAW SDK exposes one C++ API on every platform, but its
// string type follows the host: CFStringRef on macOS, BSTR (COM) on
// Windows. Everything else the helper touches (HRESULT, Variant,
// SafeArray*, the interfaces) has the same spelling on both. So the helper
// talks to the SDK through `SdkString` + three helpers and never names a
// platform string type itself:
//
//   SdkString   sdk_str(const char* utf8)   — allocate (caller frees)
//   std::string std_str(SdkString)          — read (does not free)
//   void        sdk_str_free(SdkString)     — release an SDK/our string
//
// The macOS branch is byte-for-byte what the helper did before the shim
// (CFStringCreateWithCString / CFStringGetCString / CFRelease).
//
// GPU pipeline: Metal on macOS, CUDA on Windows (`BRAW_GPU_PIPELINE`,
// `BRAW_GPU_PIPELINE_NAME`). The IOSurface zero-copy hand-off (`braw_zc.h`)
// is macOS-only: the Windows build gets inert stubs so the decode loop
// compiles unchanged and `--zerocopy` is refused at startup.
//
// This header is included AFTER the SDK header (which the build pulls in
// through BlackmagicRawAPIDispatch.cpp), so it may refer to the SDK's enums.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __APPLE__

#include <CoreServices/CoreServices.h>
#include "braw_zc.h"

typedef CFStringRef SdkString;
// `SafeArrayGetVartype` takes the SDK's own enum on macOS, a VARTYPE on Windows.
typedef BlackmagicRawVariantType SdkVarType;
// Out-flags (`isReadOnly`, `supported`): `bool*` on macOS, COM `BOOL*` on Windows.
typedef bool SdkBool;

static inline SdkString sdk_str(const char* utf8) {
    return CFStringCreateWithCString(NULL, utf8, kCFStringEncodingUTF8);
}
static inline std::string std_str(SdkString s) {
    if (!s) return "";
    const char* c = CFStringGetCStringPtr(s, kCFStringEncodingUTF8);
    if (c) return c;
    char buf[2048];
    if (CFStringGetCString(s, buf, sizeof buf, kCFStringEncodingUTF8)) return buf;
    return "";
}
static inline void sdk_str_free(SdkString s) { if (s) CFRelease(s); }

#define BRAW_GPU_PIPELINE      blackmagicRawPipelineMetal
#define BRAW_GPU_PIPELINE_NAME "metal"
// Zero-copy hand-off: global IOSurfaces (braw_zc.mm), filled by a Metal
// blit straight from the SDK's buffer (no host fallback: a failed blit is
// a failed frame, as before).
#define BRAW_HAVE_ZEROCOPY 1
#define BRAW_ZC_HOST_FALLBACK 0

// The directory handed to CreateBlackmagicRawFactoryInstanceFromPath.
static inline std::string sdk_library_dir() {
    return "/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Libraries";
}

// argv as UTF-8 strings (already is on macOS).
static inline std::vector<std::string> platform_init_args(int argc, const char** argv) {
    std::vector<std::string> out;
    for (int i = 0; i < argc; i++) out.push_back(argv[i]);
    return out;
}

#else  // _WIN32 ───────────────────────────────────────────────────────────

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <oleauto.h>
#include <objbase.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>

typedef BSTR SdkString;
typedef VARTYPE SdkVarType;
typedef BOOL SdkBool;
// The SDK's Windows header uses the COM types directly; these spellings are
// what the helper (written against the macOS header) uses. A repeated
// typedef to the same type is legal, so this is safe if the SDK has them.
typedef SAFEARRAY SafeArray;
typedef VARIANT Variant;

static inline std::wstring utf8_to_wide(const char* s) {
    if (!s || !*s) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}
static inline std::string wide_to_utf8(const wchar_t* w, int len) {
    if (!w || len == 0) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, len, &s[0], n, nullptr, nullptr);
    if (len < 0 && !s.empty() && s.back() == '\0') s.pop_back();  // -1: the terminator was counted
    return s;
}
static inline std::string wide_to_utf8(const std::wstring& w) { return wide_to_utf8(w.c_str(), (int)w.size()); }

static inline SdkString sdk_str(const char* utf8) {
    std::wstring w = utf8_to_wide(utf8);
    return SysAllocStringLen(w.data(), (UINT)w.size());
}
static inline std::string std_str(SdkString s) {
    if (!s) return "";
    return wide_to_utf8(s, (int)SysStringLen(s));
}
static inline void sdk_str_free(SdkString s) { if (s) SysFreeString(s); }

#define BRAW_GPU_PIPELINE      blackmagicRawPipelineCUDA
#define BRAW_GPU_PIPELINE_NAME "cuda"
// Zero-copy hand-off: shareable D3D11 textures (braw_zc_win.cpp), filled
// by a CUDA → D3D11 blit when the SDK decodes on CUDA and the driver is
// present (`BRAW_ZC_CUDA`, set by the build when the CUDA headers are
// found); otherwise a GPU frame is read back through the SDK and uploaded
// (see `fill_slot`).
#define BRAW_HAVE_ZEROCOPY 1
#define BRAW_ZC_HOST_FALLBACK 1
#ifndef BRAW_ZC_CUDA
#define BRAW_ZC_CUDA 0
#endif
#include "braw_zc.h"

// The SDK library directory: `BRAW_SDK_LIB_DIR` if set, else the SDK's own
// install location, else the runtime that ships with Blackmagic RAW Player /
// Speed Test, else next to this executable (a bundle that carries the DLL).
// The first directory holding BlackmagicRawAPI.dll wins.
static inline std::string sdk_library_dir() {
    std::vector<std::string> cands;
    if (const char* e = getenv("BRAW_SDK_LIB_DIR")) cands.push_back(e);
    cands.push_back("C:\\Program Files (x86)\\Blackmagic Design\\Blackmagic RAW\\Blackmagic RAW SDK\\Win\\Libraries");
    cands.push_back("C:\\Program Files (x86)\\Blackmagic Design\\Blackmagic RAW\\Blackmagic RAW Player\\BlackmagicRawAPI");
    cands.push_back("C:\\Program Files (x86)\\Blackmagic Design\\Blackmagic RAW\\Blackmagic RAW Speed Test\\BlackmagicRawAPI");
    wchar_t exe[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::wstring d(exe, n);
        size_t cut = d.find_last_of(L"\\/");
        if (cut != std::wstring::npos) cands.push_back(wide_to_utf8(d.substr(0, cut)));
    }
    for (const std::string& c : cands) {
        std::wstring dll = utf8_to_wide((c + "\\BlackmagicRawAPI.dll").c_str());
        if (GetFileAttributesW(dll.c_str()) != INVALID_FILE_ATTRIBUTES) return c;
    }
    return cands.size() > 1 ? cands[1] : std::string();
}

// Windows process setup: binary stdout/stdin (the frames and the zero-copy
// protocol are bytes — the CRT would otherwise translate \n), COM for the
// SDK, and the UTF-16 command line re-encoded as UTF-8 so a path with
// non-ASCII characters reaches the SDK intact. Returns the UTF-8 argv.
static inline std::vector<std::string> platform_init_args(int, const char**) {
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    std::vector<std::string> out;
    int n = 0;
    LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (w) {
        for (int i = 0; i < n; i++) out.push_back(wide_to_utf8(w[i], -1));
        LocalFree(w);
    }
    return out;
}

#endif  // __APPLE__ / _WIN32
