# Windows ProRes Error Fix

## Problem
When outputting ProRes on Windows, FFmpeg error code 4294967274 (which is -22 or EINVAL) occurred.

## Root Cause
The `prores_ks` encoder on Windows requires an explicit pixel format specification:
- **ProRes Proxy/LT/Standard/HQ**: Requires `yuv422p10le` (10-bit 4:2:2)
- **ProRes 4444/4444 XQ**: Requires `yuv444p10le` (10-bit 4:4:4)

Without specifying the pixel format, FFmpeg couldn't determine the correct format and returned an invalid argument error.

## Solution (Updated - v1.4.1)
Two critical fixes were needed:

### Fix 1: Pixel Format Conversion in Filter Chain
Add a `format` filter at the END of the filter chain to convert to the correct pixel format BEFORE the encoder sees it.

**For Regular ProRes Output (vr180_gui.py:446-451):**
```python
# Add format filter to filter chain (BEFORE encoding)
if needs_prores and sys.platform != 'darwin':
    if output_codec == "prores" and cfg.prores_profile in ["4444", "4444xq"]:
        filters.append(f"{current_label}format=yuv444p10le[out]")
    else:
        filters.append(f"{current_label}format=yuv422p10le[out]")
```

**For MV-HEVC Intermediate (vr180_gui.py:461):**
```python
enc = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]  # ProRes HQ (format set in filter)
```

### Fix 2: Auto-Update File Extension for ProRes
ProRes codec MUST use `.mov` (QuickTime) containers. MP4 containers do not support ProRes.

**Auto-Extension Update (vr180_gui.py:1534-1547):**
```python
# Auto-update output file extension based on codec
if self.output_path_edit.text():
    output_path = Path(self.output_path_edit.text())
    if is_prores and output_path.suffix.lower() != '.mov':
        # Change to .mov for ProRes
        new_path = output_path.with_suffix('.mov')
        self.output_path_edit.setText(str(new_path))
        self.config.output_path = new_path
```

### Why These Fixes Work
1. **Pixel Format**: The original fix tried to set `-pix_fmt` in encoder options, but FFmpeg still received incompatible format from the filter chain. By adding the `format` filter at the end of the filter chain, frames are in the correct format BEFORE reaching the encoder.

2. **Container Format**: FFmpeg error "Could not find tag for codec prores in stream #0, codec not currently supported in container" occurs when trying to put ProRes in MP4. ProRes is a QuickTime codec and requires `.mov` container. The GUI now automatically changes the extension when ProRes is selected.

## Files Modified
1. `/Users/siyangqi/Downloads/vr180_processor/vr180_gui.py` (main build)
2. `/Users/siyangqi/Downloads/vr180_processor/VR180_Silver_Bullet_Windows_BuildKit/vr180_gui.py` (Windows build kit)

## Testing
After this fix, ProRes encoding on Windows should work correctly for all profiles:
- ✅ ProRes Proxy
- ✅ ProRes LT
- ✅ ProRes Standard
- ✅ ProRes HQ
- ✅ ProRes 4444
- ✅ ProRes 4444 XQ

## Note
This fix only affects non-macOS platforms. macOS continues to use `prores_videotoolbox` which handles pixel format automatically.
