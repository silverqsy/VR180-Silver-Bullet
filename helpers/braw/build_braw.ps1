# Build braw_helper.exe from the Blackmagic RAW SDK on Windows.
#
# The twin of build_braw.sh: the same helpers/braw/braw_helper.cpp (plus the
# platform shim braw_platform.h) compiled with MSVC against the SDK's Windows
# API, landing in helpers/bin/braw_helper.exe (gitignored).
# vr180-braw::locate_helper finds it there during development, or next to
# vr180-gui.exe in a release bundle.
#
# On Windows the SDK ships the API as an IDL file (Include\BlackmagicRawAPI.idl):
# MIDL generates BlackmagicRawAPI.h (the interfaces) and BlackmagicRawAPI_i.c
# (the IID/CLSID definitions, linked in) into a scratch directory first; the
# SDK's BlackmagicRawAPIDispatch.cpp (LoadLibrary of BlackmagicRawAPI.dll) is
# force-included into the helper's translation unit, as build_braw.sh does
# with -include on macOS.
#
# Requires:
#   * the Blackmagic RAW SDK — the "Blackmagic RAW" installer from
#     https://www.blackmagicdesign.com/support/ (Capture and Playback →
#     Blackmagic RAW) with the "SDK" component selected; it installs to
#     C:\Program Files (x86)\Blackmagic Design\Blackmagic RAW\Blackmagic RAW SDK\Win
#     (override with -SdkDir or $env:BRAW_SDK_DIR). The SDK is not
#     redistributable: the helper loads the user's installed
#     BlackmagicRawAPI.dll at runtime (see sdk_library_dir() in the shim).
#   * Visual Studio 2022/2019 with the C++ workload (cl.exe, midl.exe and
#     the Windows SDK, all reached through vcvars64.bat).
#
# Usage:  .\helpers\braw\build_braw.ps1 [-SdkDir <path>] [-Config Release|Debug]

param(
    [string]$SdkDir = $env:BRAW_SDK_DIR,
    [ValidateSet('Release', 'Debug')] [string]$Config = 'Release'
)
$ErrorActionPreference = 'Stop'

if (-not $SdkDir) {
    $SdkDir = 'C:\Program Files (x86)\Blackmagic Design\Blackmagic RAW\Blackmagic RAW SDK\Win'
}
$Include = Join-Path $SdkDir 'Include'
$Idl = Join-Path $Include 'BlackmagicRawAPI.idl'
$Dispatch = Join-Path $Include 'BlackmagicRawAPIDispatch.cpp'
if (-not (Test-Path $Idl) -or -not (Test-Path $Dispatch)) {
    Write-Error @"
Blackmagic RAW SDK not found: $Idl

Install the SDK: run the Blackmagic RAW installer (Capture and Playback
Software -> Blackmagic RAW on https://www.blackmagicdesign.com/support/) and
select the SDK component, or point -SdkDir / `$env:BRAW_SDK_DIR at an SDK
whose Include\ holds BlackmagicRawAPI.idl and BlackmagicRawAPIDispatch.cpp.
"@
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HelpersDir = Split-Path -Parent $ScriptDir
$BinDir = Join-Path $HelpersDir 'bin'
New-Item -ItemType Directory -Force $BinDir | Out-Null
$Src = Join-Path $ScriptDir 'braw_helper.cpp'
$Out = Join-Path $BinDir 'braw_helper.exe'
$Obj = Join-Path $env:TEMP ("braw_helper_build_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $Obj | Out-Null

# MSVC environment: the newest Visual Studio with vcvars64.bat.
$vcvars = Get-ChildItem @(
    'C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvars64.bat'
) -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vcvars) { Write-Error 'vcvars64.bat not found - install Visual Studio Build Tools (C++ workload).' }

$opt = if ($Config -eq 'Release') { '/O2 /DNDEBUG' } else { '/Od /Zi /DDEBUG' }
$defs = '/DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DUNICODE /D_UNICODE'
# /utf-8: the sources and the paths passed around are UTF-8. /EHsc: C++
# exceptions (std::stoull etc.). The IID source is C and must not see the
# C++ dispatch include, hence the separate compile.
$cmd = @"
@echo off
call "$($vcvars.FullName)" >nul 2>&1
if errorlevel 1 exit /b 1
echo [midl] BlackmagicRawAPI.idl
midl /nologo /env x64 /out "$Obj" /h BlackmagicRawAPI.h /iid BlackmagicRawAPI_i.c /proxy BlackmagicRawAPI_p.c /dlldata dlldata.c /tlb BlackmagicRawAPI.tlb "$Idl"
if errorlevel 1 exit /b 2
echo [cl] BlackmagicRawAPI_i.c
cl /nologo /c $opt /W3 $defs /I"$Obj" /I"$Include" /Fo"$Obj\api_i.obj" "$Obj\BlackmagicRawAPI_i.c"
if errorlevel 1 exit /b 3
echo [cl] braw_helper.cpp
cl /nologo /c /std:c++17 $opt /EHsc /utf-8 /W3 $defs /I"$Obj" /I"$Include" /I"$ScriptDir" /FI"$Dispatch" /Fo"$Obj\braw_helper.obj" "$Src"
if errorlevel 1 exit /b 4
echo [cl] braw_zc_win.cpp
cl /nologo /c /std:c++17 $opt /EHsc /utf-8 /W3 $defs /I"$Obj" /I"$Include" /I"$ScriptDir" /Fo"$Obj\braw_zc_win.obj" "$ScriptDir\braw_zc_win.cpp"
if errorlevel 1 exit /b 4
echo [link] braw_helper.exe
link /nologo /OUT:"$Out" "$Obj\braw_helper.obj" "$Obj\braw_zc_win.obj" "$Obj\api_i.obj" ole32.lib oleaut32.lib shell32.lib d3d11.lib dxgi.lib
if errorlevel 1 exit /b 5
"@
$bat = Join-Path $Obj 'build.cmd'
Set-Content -Path $bat -Value $cmd -Encoding ASCII
Write-Host "Compiling braw_helper.exe (SDK: $SdkDir)..."
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
& cmd.exe /c "`"$bat`"" 2>&1 | ForEach-Object { "$_" }
$ErrorActionPreference = $prevEap
$rc = $LASTEXITCODE
if ($rc -ne 0) { Write-Error "build failed at step $rc (scratch kept in $Obj)" }
Remove-Item -Recurse -Force $Obj -ErrorAction SilentlyContinue
Write-Host "Build successful: $Out"
