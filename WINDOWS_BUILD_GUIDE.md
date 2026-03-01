# VR180 Silver Bullet - Windows Build Guide

## Prerequisites

### 1. Install Python 3.11 or 3.12
Download from https://www.python.org/downloads/windows/

**Important:** During installation, check "Add Python to PATH"

### 2. Install FFmpeg
Download from https://www.gyan.dev/ffmpeg/builds/

1. Download `ffmpeg-release-essentials.zip`
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH:
   - Open System Properties → Environment Variables
   - Under System variables, find `Path`
   - Click Edit → New → Add `C:\ffmpeg\bin`
   - Click OK on all windows

Verify installation:
```cmd
ffmpeg -version
ffprobe -version
```

### 3. Install Required Python Packages
Open Command Prompt and run:
```cmd
pip install PyQt6 numpy Pillow pyinstaller
```

## Building the Windows Application

### Step 1: Copy Project Files
Copy the entire `vr180_processor` folder to your Windows machine.

### Step 2: Verify Icon Files
Make sure these files exist in the project folder:
- `icon.ico` (127 KB - Windows icon)
- `icon.icns` (2.3 MB - macOS icon, not used on Windows)
- `icon.png` (967 KB - source icon)

### Step 3: Build with PyInstaller
Open Command Prompt in the project folder and run:
```cmd
pyinstaller --clean vr180_processor.spec
```

### Step 4: Find Your Application
After building successfully, the application will be in:
```
dist\VR180 Silver Bullet\VR180Processor.exe
```

## Features Available on Windows

### ✅ Fully Supported Features
1. **VR180 Processing**
   - Global shift (horizontal alignment)
   - Global yaw/pitch/roll adjustments
   - Stereo offset adjustments

2. **Output Codecs**
   - H.265 (HEVC) encoding
   - ProRes encoding (all profiles: Proxy, LT, Standard, HQ, 4444, 4444 XQ)
   - 8-bit and 10-bit H.265 output

3. **Hardware Acceleration**
   - NVIDIA NVENC (if NVIDIA GPU available)
   - H.265 hardware encoding
   - ProRes hardware encoding (limited GPU support)

4. **Color Grading**
   - Gamma adjustment
   - White point adjustment
   - Black point adjustment
   - LUT (.cube) file support with intensity control

5. **Preview Modes**
   - Side by Side
   - Anaglyph (Red/Cyan)
   - Overlay 50%
   - Single Eye Mode with eye toggle
   - Difference view
   - Checkerboard pattern

6. **VR180 Metadata**
   - Inject YouTube VR180 metadata (Spherical Video V2)
   - Works with all codecs

### ❌ NOT Available on Windows
1. **Vision Pro Features**
   - "Apple Compatible (hvc1 tag)" mode requires macOS tools
   - "Vision Pro MV-HEVC" mode requires macOS `avconvert` tool
   - These options will be grayed out on Windows

### ⚠️ Platform-Specific Notes

**Hardware Acceleration:**
- Windows uses NVENC for NVIDIA GPUs
- macOS uses VideoToolbox
- If no compatible GPU found, falls back to software encoding

**ProRes Encoding:**
- Windows: Software encoding only (slower but works)
- macOS: Hardware-accelerated VideoToolbox (much faster)

**File Paths:**
- Windows uses backslashes: `C:\Videos\input.mp4`
- macOS uses forward slashes: `/Users/name/Videos/input.mp4`

## Troubleshooting

### "ffmpeg not found"
1. Verify FFmpeg is in PATH: `where ffmpeg`
2. If not found, re-add `C:\ffmpeg\bin` to PATH
3. Restart Command Prompt and try again

### "PyQt6 not found"
```cmd
pip install --upgrade PyQt6
```

### Build fails with "icon.ico not found"
Make sure `icon.ico` exists in the project root folder.

### Application won't start
1. Check if running from correct location: `dist\VR180 Silver Bullet\`
2. Try running from Command Prompt to see error messages:
   ```cmd
   cd "dist\VR180 Silver Bullet"
   VR180Processor.exe
   ```

### Slow ProRes encoding
This is normal on Windows - ProRes encoding is software-based.
For faster encoding, use H.265 with hardware acceleration.

## Creating a Distributable Package

To share the application:

1. Compress the entire folder:
   ```
   dist\VR180 Silver Bullet\
   ```

2. Users just need to:
   - Extract the folder
   - Run `VR180Processor.exe`
   - No Python or dependencies required!

## Feature Comparison: Windows vs macOS

| Feature | Windows | macOS |
|---------|---------|-------|
| H.265 Encoding | ✅ | ✅ |
| ProRes Encoding | ✅ (slower) | ✅ (faster) |
| Hardware Acceleration | ✅ NVENC | ✅ VideoToolbox |
| 10-bit H.265 | ✅ | ✅ |
| LUT Support | ✅ | ✅ |
| VR180 Metadata | ✅ | ✅ |
| Vision Pro hvc1 | ❌ | ✅ |
| Vision Pro MV-HEVC | ❌ | ✅ |
| All Preview Modes | ✅ | ✅ |

## Recommended Settings for Windows

For best performance on Windows:

1. **Codec:** H.265
2. **Quality:** CRF 23-28 (lower = better quality)
3. **Hardware Acceleration:** Enabled (if NVIDIA GPU)
4. **Bit Depth:** 8-bit (better compatibility)

For maximum quality (slower):
1. **Codec:** ProRes HQ or 4444
2. **Hardware Acceleration:** Disabled (not available for ProRes on Windows)
