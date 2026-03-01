# Bug Fix - VR180 Silver Bullet v1.4.3

**Release Date**: January 19, 2026

## Critical Windows Bug Fix

### Issue
Windows builds were failing to find bundled `ffprobe.exe` and `ffmpeg.exe`, resulting in the error:
```
Command '['Y:\...\ffprobe.EXE', '-v', 'quiet', ...] returned non-zero exit status 1.
```

### Root Cause
The path detection functions `get_ffmpeg_path()`, `get_ffprobe_path()`, and `get_spatial_path()` were looking for executables without the `.exe` extension on Windows.

On Windows, executables require the `.exe` extension, so:
- Looking for `ffprobe` fails
- Must look for `ffprobe.exe` instead

### Fix Applied
Updated all three path detection functions to add `.exe` extension on Windows:

**File**: `vr180_gui.py`

**Changes**:
1. `get_ffmpeg_path()` - Now looks for `ffmpeg.exe` on Windows
2. `get_ffprobe_path()` - Now looks for `ffprobe.exe` on Windows
3. `get_spatial_path()` - Now looks for `spatial.exe` on Windows

**Code Example**:
```python
# Before (broken on Windows)
ffprobe_path = base_path / 'ffprobe'

# After (works on Windows)
ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
ffprobe_path = base_path / ffprobe_name
```

### Impact
- ✅ **CRITICAL FIX**: Windows builds can now find bundled executables
- ✅ Video import works correctly on Windows
- ✅ FFprobe can probe video files
- ✅ FFmpeg can process videos
- ✅ No impact on macOS (still uses paths without extensions)

### Testing
- Tested on Windows 10/11 with PyInstaller bundles
- Verified ffprobe.exe is found in `_internal` folder
- Verified video import and processing works

### Files Changed
- `vr180_gui.py` - Lines 67, 102, 137 (added platform-specific executable names)

### Affected Functions
```python
def get_ffmpeg_path():
    # Line 67: Added Windows .exe extension check
    ffmpeg_name = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'

def get_ffprobe_path():
    # Line 102: Added Windows .exe extension check
    ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'

def get_spatial_path():
    # Line 137: Added Windows .exe extension check
    spatial_name = 'spatial.exe' if sys.platform == 'win32' else 'spatial'
```

### Version History
- **v1.4.3** (January 19, 2026) - Fixed Windows executable path detection
- **v1.4.2** (January 18, 2026) - Fixed drag-and-drop and spinbox issues
- **v1.4.1** (January 18, 2026) - Previous release

### Upgrade from v1.4.2
This is a **critical bug fix** for Windows users. If you're on Windows and experiencing:
- "ffprobe.EXE returned non-zero exit status 1" errors
- Unable to import videos
- "Command not found" errors

**You must upgrade to v1.4.3.**

For macOS users: This fix doesn't affect you, but you can upgrade for consistency.

### Distribution
- Windows Build Kit updated with fixed source code
- All users should rebuild from updated BuildKit
- macOS app will be rebuilt with this fix

## Technical Details

### Why This Happened
Python's `Path.exists()` is case-sensitive and extension-aware on Windows. When PyInstaller bundles executables on Windows, they retain their `.exe` extensions. The code was checking for files without extensions, so `Path('ffprobe').exists()` returned `False` even though `ffprobe.exe` was present.

### Why shutil.which() Wasn't Affected
`shutil.which()` automatically adds the `.exe` extension on Windows when searching PATH, but `Path.exists()` does not. This is why system-installed ffmpeg worked, but bundled executables didn't.

### Platform Detection
Using `sys.platform == 'win32'` correctly detects Windows (including Windows 10/11) and distinguishes it from macOS (`'darwin'`) and Linux (`'linux'`).

## Summary

**Severity**: CRITICAL (Windows only)
**Impact**: Windows builds completely broken without this fix
**Solution**: Add `.exe` extension when detecting bundled executables on Windows
**Status**: ✅ FIXED

All Windows users should update immediately.
