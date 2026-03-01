# Windows Build - NumPy Error Fix

## Problem
You're seeing an error like:
```
Importing the numpy C-extensions failed.
```

This happens when NumPy is incompatible with your Python version or wasn't bundled correctly by PyInstaller.

---

## Solution 1: Quick Fix Script (Recommended)

Run the fix script I created:

```cmd
fix_windows_build.bat
```

This will:
1. Reinstall NumPy with a compatible version
2. Clean previous builds
3. Rebuild the application

---

## Solution 2: Manual Fix

### Step 1: Check Python Version
```cmd
python --version
```

**Important**: The app works best with Python 3.9, 3.10, or 3.11.
- If you have Python 3.12 or 3.13, consider downgrading to 3.11
- Python 3.8 may also work but is older

### Step 2: Reinstall NumPy with Specific Version

```cmd
python -m pip uninstall -y numpy
python -m pip install numpy==1.26.4
```

If that doesn't work, try an older version:
```cmd
python -m pip install numpy==1.24.3
```

### Step 3: Clean and Rebuild

```cmd
rmdir /s /q build
rmdir /s /q dist
python -m PyInstaller --clean vr180_processor.spec
```

---

## Solution 3: Use Console Mode for Debugging

If the app still won't run, build it in console mode to see the full error:

1. Edit `vr180_processor.spec`
2. Find the line `console=False,` (around line 53)
3. Change it to `console=True,`
4. Rebuild:
   ```cmd
   python -m PyInstaller --clean vr180_processor.spec
   ```
5. Run the exe - it will show a console window with error details

---

## Solution 4: Alternative Build Method

Try building without the spec file:

```cmd
python -m PyInstaller --onedir --windowed --name="SilversVR180Tool" ^
  --add-binary="C:\path\to\ffmpeg.exe;." ^
  --add-binary="C:\path\to\ffprobe.exe;." ^
  --hidden-import=numpy.core._multiarray_umath ^
  --hidden-import=PyQt6.QtCore ^
  --hidden-import=PyQt6.QtGui ^
  --hidden-import=PyQt6.QtWidgets ^
  vr180_gui.py
```

(Replace `C:\path\to\` with actual paths to ffmpeg and ffprobe)

---

## Common Issues and Solutions

### Issue: "Python was not found"
**Fix**: Reinstall Python and check "Add Python to PATH" during installation

### Issue: "pip is not recognized"
**Fix**:
```cmd
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Issue: "Access is denied" when deleting build/dist folders
**Fix**: Close all File Explorer windows showing those folders, then:
```cmd
taskkill /F /IM VR180Processor.exe
rmdir /s /q build
rmdir /s /q dist
```

### Issue: Missing DLL errors (msvcp140.dll, vcruntime140.dll, etc.)
**Fix**: Install Microsoft Visual C++ Redistributable
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Run the installer
- Rebuild the app

### Issue: "The code execution cannot proceed because numpy.core._multiarray_umath.cp312-win_amd64.pyd was not found"
**Fix**: This confirms Python 3.12 compatibility issues. Downgrade to Python 3.11:
1. Uninstall Python 3.12
2. Download Python 3.11 from python.org
3. Install with "Add to PATH" checked
4. Run `fix_windows_build.bat`

---

## Verification

After fixing, test the build:

1. Navigate to the build output:
   ```cmd
   cd dist\VR180Processor
   ```

2. Run the executable:
   ```cmd
   VR180Processor.exe
   ```

3. The app should launch without errors

---

## Still Having Issues?

### Debug Mode Build

1. Open `vr180_processor.spec`
2. Change line 49: `debug=False,` → `debug=True,`
3. Change line 53: `console=False,` → `console=True,`
4. Rebuild and run - you'll see detailed debug output

### Check Build Warnings

Look at the build output for warnings like:
```
WARNING: library xyz not found
```

These indicate missing dependencies that need to be installed.

### Environment Variables

Make sure these are in your PATH:
- Python installation directory
- Python\Scripts directory
- FFmpeg bin directory

Check with:
```cmd
echo %PATH%
```

---

## Alternative: Use Virtual Environment

Sometimes a clean environment helps:

```cmd
python -m venv vr180_env
vr180_env\Scripts\activate
pip install -r requirements.txt
python -m PyInstaller --clean vr180_processor.spec
```

---

## Contact

If none of these solutions work, the issue might be:
1. Antivirus blocking PyInstaller
2. Corrupted Python installation
3. Missing system libraries

Try:
- Temporarily disable antivirus during build
- Reinstall Python completely
- Update Windows to latest version
