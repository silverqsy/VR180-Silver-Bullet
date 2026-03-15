# VR180 Silver Bullet - Windows Build Script
# PowerShell script to build Windows executable

param(
    [switch]$Clean = $false,
    [switch]$Package = $true,
    [switch]$Test = $false
)

$AppName = "VR180 Silver Bullet"
$ZipName = "VR180_Silver_Bullet_Windows.zip"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VR180 Silver Bullet - Windows Build  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found! Please install Python 3.11 or 3.12" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check FFmpeg installation
Write-Host "[2/5] Checking FFmpeg installation..." -ForegroundColor Yellow
$ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: FFmpeg not found in PATH!" -ForegroundColor Yellow
    Write-Host "  The app will not include FFmpeg binaries." -ForegroundColor Yellow
    Write-Host "  Users will need to install FFmpeg separately." -ForegroundColor Yellow
} else {
    Write-Host "  Found: $ffmpegVersion" -ForegroundColor Green
}
Write-Host ""

# Clean previous builds
if ($Clean) {
    Write-Host "[3/5] Cleaning previous builds..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
    Write-Host "  Cleaned dist/ and build/ directories" -ForegroundColor Green
} else {
    Write-Host "[3/5] Skipping clean (use -Clean to clean)" -ForegroundColor Yellow
}
Write-Host ""

# Build with PyInstaller
Write-Host "[4/5] Building with PyInstaller..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Cyan

python -m PyInstaller vr180_silver_bullet.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Red
    exit 1
}

Write-Host "  Build completed successfully!" -ForegroundColor Green
Write-Host ""

# Get build size
$buildPath = "dist\$AppName"
$buildSize = (Get-ChildItem -Path $buildPath -Recurse | Measure-Object -Property Length -Sum).Sum
$buildSizeMB = [math]::Round($buildSize / 1MB, 2)
Write-Host "  Build size: $buildSizeMB MB" -ForegroundColor Cyan

# List key files
$exePath = "$buildPath\VR180Processor.exe"
$ffmpegPath = "$buildPath\ffmpeg.exe"
$ffprobePath = "$buildPath\ffprobe.exe"

if (Test-Path $exePath) {
    Write-Host "  ✓ VR180Processor.exe" -ForegroundColor Green
} else {
    Write-Host "  ✗ VR180Processor.exe MISSING!" -ForegroundColor Red
}

if (Test-Path $ffmpegPath) {
    Write-Host "  ✓ ffmpeg.exe (bundled)" -ForegroundColor Green
} else {
    Write-Host "  ⚠ ffmpeg.exe not bundled" -ForegroundColor Yellow
}

if (Test-Path $ffprobePath) {
    Write-Host "  ✓ ffprobe.exe (bundled)" -ForegroundColor Green
} else {
    Write-Host "  ⚠ ffprobe.exe not bundled" -ForegroundColor Yellow
}
Write-Host ""

# Create ZIP package
if ($Package) {
    Write-Host "[5/5] Creating ZIP package..." -ForegroundColor Yellow

    Set-Location dist

    # Remove old ZIP if exists
    if (Test-Path $ZipName) {
        Remove-Item $ZipName -Force
    }

    Compress-Archive -Path $AppName -DestinationPath $ZipName -Force

    $zipSize = (Get-Item $ZipName).Length
    $zipSizeMB = [math]::Round($zipSize / 1MB, 2)

    Write-Host "  Package created: $ZipName ($zipSizeMB MB)" -ForegroundColor Green
    Write-Host "  Location: $(Get-Location)\$ZipName" -ForegroundColor Cyan

    Set-Location ..
} else {
    Write-Host "[5/5] Skipping package creation (use -Package)" -ForegroundColor Yellow
}
Write-Host ""

# Test the build
if ($Test) {
    Write-Host "Testing the build..." -ForegroundColor Yellow
    Write-Host "Launching VR180 Silver Bullet..." -ForegroundColor Cyan
    Write-Host "(Close the application to continue)" -ForegroundColor Cyan
    Write-Host ""

    & "dist\$AppName\VR180Processor.exe"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Application exited with error code $LASTEXITCODE" -ForegroundColor Yellow
    } else {
        Write-Host "Application closed successfully" -ForegroundColor Green
    }
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "           Build Complete!              " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output directory: dist\$AppName" -ForegroundColor Cyan
if ($Package) {
    Write-Host "Distribution ZIP: dist\$ZipName" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test: .\dist\$AppName\VR180Processor.exe" -ForegroundColor White
Write-Host "  2. Distribute: dist\$ZipName" -ForegroundColor White
Write-Host ""
