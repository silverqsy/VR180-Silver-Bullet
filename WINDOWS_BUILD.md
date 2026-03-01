# Windows Build Instructions for VR180 Silver Bullet

## Prerequisites

### 1. Install Python 3.11 or 3.12
Download from: https://www.python.org/downloads/windows/

**Important:** Check "Add Python to PATH" during installation

### 2. Install FFmpeg
Two options:

**Option A: Using Chocolatey (Recommended)**
```powershell
# Install Chocolatey first (run PowerShell as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg
```

**Option B: Manual Installation**
1. Download FFmpeg from: https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH

Verify installation:
```powershell
ffmpeg -version
ffprobe -version
```

### 3. Install Python Dependencies
```powershell
cd C:\path\to\vr180_processor
pip install -r requirements.txt
pip install pyinstaller
```

## Build Steps

### Quick Build
```powershell
# Clean previous builds
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

# Build the application
python -m PyInstaller vr180_processor.spec

# Output will be in: dist\VR180 Silver Bullet\
```

### Alternative: One-File Build (Slower startup)
```powershell
python -m PyInstaller --onefile --windowed --name "VR180 Silver Bullet" --icon=icon.ico vr180_gui.py
```

## Creating Distribution Package

### Option 1: ZIP Archive
```powershell
# Navigate to dist folder
cd dist

# Create ZIP
Compress-Archive -Path "VR180 Silver Bullet" -DestinationPath "VR180_Silver_Bullet_Windows.zip"
```

### Option 2: Installer (Advanced)
Use Inno Setup to create a Windows installer:

1. Install Inno Setup: https://jrsoftware.org/isdl.php
2. Use the included `installer.iss` script
3. Compile to create `VR180_Silver_Bullet_Setup.exe`

## Troubleshooting

### FFmpeg Not Found
If build succeeds but app crashes with "FFmpeg not found":
- Ensure FFmpeg is in System PATH
- Manually copy ffmpeg.exe, ffprobe.exe to the dist folder

### Missing DLL Errors
Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Import Errors
```powershell
# Reinstall dependencies
pip uninstall -y PyQt6 numpy
pip install PyQt6 numpy
```

### Large File Size
The Windows build is typically 200-300MB due to:
- FFmpeg binaries (~150MB)
- Python runtime
- PyQt6 libraries

This is normal and expected.

## Testing the Build

```powershell
cd "dist\VR180 Silver Bullet"
.\VR180Processor.exe
```

The application should launch without requiring Python or showing a console window.

## Distribution Notes

**Required Files in Distribution:**
- VR180Processor.exe (main executable)
- All DLL files in the folder
- _internal\ folder (contains Python runtime and libraries)
- ffmpeg.exe, ffprobe.exe (if bundled)

**Optional Files:**
- spatial.exe (only needed for MV-HEVC/Vision Pro support)

**Total Size:** ~200-300MB compressed to ~80-120MB as ZIP

## Build on Windows Workflow

1. **On Windows PC:**
   ```powershell
   git clone <repository>
   cd vr180_processor
   pip install -r requirements.txt
   pip install pyinstaller
   python -m PyInstaller vr180_processor.spec
   ```

2. **Test locally:**
   ```powershell
   cd "dist\VR180 Silver Bullet"
   .\VR180Processor.exe
   ```

3. **Create package:**
   ```powershell
   cd dist
   Compress-Archive -Path "VR180 Silver Bullet" -DestinationPath "VR180_Silver_Bullet_Windows_v1.0.zip"
   ```

4. **Distribute:**
   - Upload to GitHub Releases
   - Share via cloud storage
   - Create installer with Inno Setup

## Automated Build Script

Save as `build_windows.ps1`:

```powershell
# VR180 Silver Bullet - Windows Build Script
Write-Host "Building VR180 Silver Bullet for Windows..." -ForegroundColor Green

# Clean
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

# Build
Write-Host "Building with PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller vr180_processor.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "Output: dist\VR180 Silver Bullet\" -ForegroundColor Cyan

    # Create ZIP
    Write-Host "Creating ZIP package..." -ForegroundColor Yellow
    cd dist
    Compress-Archive -Path "VR180 Silver Bullet" -DestinationPath "VR180_Silver_Bullet_Windows.zip" -Force
    Write-Host "Package created: VR180_Silver_Bullet_Windows.zip" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
```

Run with:
```powershell
.\build_windows.ps1
```
