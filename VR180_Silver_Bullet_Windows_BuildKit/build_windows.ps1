# VR180 Silver Bullet - Windows Build Script (PowerShell)
# Run this in PowerShell to build the application

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "VR180 Silver Bullet - Windows Build Script" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
Write-Host "Checking for Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for FFmpeg
Write-Host ""
Write-Host "Checking for FFmpeg..." -ForegroundColor Yellow
try {
    $ffmpegPath = Get-Command ffmpeg -ErrorAction Stop
    Write-Host "✓ Found: $($ffmpegPath.Source)" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: FFmpeg not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Red
    Write-Host "  1. Download 'ffmpeg-release-full.7z'" -ForegroundColor Red
    Write-Host "  2. Extract to C:\ffmpeg" -ForegroundColor Red
    Write-Host "  3. Add C:\ffmpeg\bin to your PATH" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for FFprobe
try {
    $ffprobePath = Get-Command ffprobe -ErrorAction Stop
    Write-Host "✓ Found: $($ffprobePath.Source)" -ForegroundColor Green
} catch {
    Write-Host "✗ Warning: FFprobe not found (should be with FFmpeg)" -ForegroundColor Yellow
}

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Clean previous builds
Write-Host ""
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}
Write-Host "✓ Cleaned" -ForegroundColor Green

# Build with PyInstaller
Write-Host ""
Write-Host "Building application with PyInstaller..." -ForegroundColor Yellow
Write-Host "This will take 3-5 minutes..." -ForegroundColor Yellow
Write-Host ""

python -m PyInstaller --clean vr180_silver_bullet.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ ERROR: PyInstaller build failed" -ForegroundColor Red
    Write-Host "Please check error messages above" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify bundled files
Write-Host ""
Write-Host "Verifying bundled files..." -ForegroundColor Yellow

$distPath = "dist\VR180 Silver Bullet"
if (Test-Path "$distPath\ffmpeg.exe") {
    $size = (Get-Item "$distPath\ffmpeg.exe").Length
    Write-Host "✓ FFmpeg bundled successfully ($([math]::Round($size/1MB, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: FFmpeg not bundled, copying manually..." -ForegroundColor Yellow
    Copy-Item $ffmpegPath.Source "$distPath\"
}

if (Test-Path "$distPath\ffprobe.exe") {
    Write-Host "✓ FFprobe bundled successfully" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: FFprobe not bundled, copying manually..." -ForegroundColor Yellow
    if ($ffprobePath) {
        Copy-Item $ffprobePath.Source "$distPath\"
    }
}

if (Test-Path "$distPath\spatialmedia") {
    Write-Host "✓ Spatialmedia module bundled" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Spatialmedia module missing" -ForegroundColor Yellow
}

# Calculate total size
$totalSize = (Get-ChildItem -Path $distPath -Recurse | Measure-Object -Property Length -Sum).Sum
$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Application: $distPath" -ForegroundColor White
Write-Host "Executable: $distPath\VR180Processor.exe" -ForegroundColor White
Write-Host "Total size: $totalSizeMB MB" -ForegroundColor White
Write-Host ""
Write-Host "This is a FULLY STANDALONE application." -ForegroundColor Green
Write-Host "No FFmpeg installation required on target systems!" -ForegroundColor Green
Write-Host ""
Write-Host "To distribute:" -ForegroundColor Yellow
Write-Host "  1. Compress the '$distPath' folder to ZIP" -ForegroundColor Yellow
Write-Host "  2. Share the ZIP file" -ForegroundColor Yellow
Write-Host ""
Write-Host "To test:" -ForegroundColor Yellow
Write-Host "  $distPath\VR180Processor.exe" -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to exit"
