# Build braw_helper.exe from the Blackmagic RAW SDK on Windows.
#
# The twin of build_braw.sh: the same helpers/braw/braw_helper.cpp (plus the
# platform shim braw_platform.h) compiled with MSVC against the SDK's Windows
# headers, landing in helpers/bin/braw_helper.exe (gitignored).
# vr180-braw::locate_helper finds it there during development, or next to
# vr180-gui.exe in a release bundle.
#
# Requires:
#   * the Blackmagic RAW SDK — the "Blackmagic RAW" installer from
#     https://www.blackmagicdesign.com/support/ (Capture and Playback →
#     Blackmagic RAW) with the "SDK" component selected; it installs to
#     C:\Program Files (x86)\Blackmagic Design\Blackmagic RAW\Blackmagic RAW SDK\Win
#     (override with -SdkDir or $env:BRAW_SDK_DIR). The SDK is not
#     redistributable: the helper loads the user's installed
#     BlackmagicRawAPI.dll at runtime (see sdk_library_dir() in the shim).
#   * Visual Studio 2022/2019 Build Tools (cl.exe via vcvars64.bat).
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
$Header = Join-Path $Include 'BlackmagicRawAPI.h'
$Dispatch = Join-Path $Include 'BlackmagicRawAPIDispatch.cpp'
if (-not (Test-Path $Header)) {
    Write-Error @"
Blackmagic RAW SDK not found: $Header

Install the SDK: run the Blackmagic RAW installer (Capture and Playback
Software → Blackmagic RAW on https://www.blackmagicdesign.com/support/) and
select the SDK component, or point -SdkDir / `$env:BRAW_SDK_DIR at an
unpacked SDK whose Include\ holds BlackmagicRawAPI.h.
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
if (-not $vcvars) { Write-Error 'vcvars64.bat not found — install Visual Studio Build Tools (C++ workload).' }

$opt = if ($Config -eq 'Release') { '/O2 /DNDEBUG' } else { '/Od /Zi /DDEBUG' }
# The SDK dispatch source is included into this one translation unit (/FI),
# as build_braw.sh does with -include. /utf-8: the sources and the paths
# passed around are UTF-8. /EHsc: C++ exceptions (std::stoull etc.).
$cmd = @"
call "$($vcvars.FullName)" >nul
cl /nologo /std:c++17 $opt /EHsc /utf-8 /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DUNICODE /D_UNICODE ^
   /I"$Include" /FI"$Dispatch" ^
   /Fo"$Obj\\" /Fe"$Out" "$Src" ^
   /link ole32.lib oleaut32.lib shell32.lib
"@
$bat = Join-Path $Obj 'build.cmd'
Set-Content -Path $bat -Value $cmd -Encoding ASCII
Write-Host "Compiling braw_helper.exe (SDK: $SdkDir)..."
& cmd.exe /c "`"$bat`""
if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $Obj -ErrorAction SilentlyContinue; Write-Error "cl.exe failed ($LASTEXITCODE)" }
Remove-Item -Recurse -Force $Obj -ErrorAction SilentlyContinue
Write-Host "Build successful: $Out"
& $Out 2>&1 | Select-Object -First 3 | ForEach-Object { Write-Host "  $_" }
