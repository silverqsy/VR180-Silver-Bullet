# VR180 Silver Bullet v1.4.2 - Release Notes

**Release Date:** January 18, 2026

## 🐛 Bug Fixes & UI Improvements

This is a maintenance release focused on fixing critical bugs and improving user experience.

---

## What's Fixed

### 🔧 Drag and Drop Crash (macOS)
**Problem:** Application crashed when dragging video files into the window
- Crash occurred in Qt's drop event handler
- Error: "sipQMainWindow::dropEvent" causing abort()
- Made the app unusable for drag-and-drop workflows

**Solution:**
- Replaced `event.acceptProposedAction()` with standard `event.accept()` method
- Added comprehensive exception handling in drag/drop events
- App now gracefully handles any drag/drop errors without crashing

**Impact:** ✅ Drag and drop is now stable and reliable on macOS

---

### 🎬 Drag and Drop Preview Loading
**Problem:** Dropping a video file didn't trigger preview loading
- File paths were set correctly
- Preview remained blank/empty
- Had to manually click Browse button to load preview

**Solution:**
- Fixed incorrect method call: `_load_video()` → `_load_video_info()`
- Added process button activation
- Now mirrors Browse button behavior exactly

**Impact:** ✅ Drag and drop now loads video info and starts preview automatically

---

### 🎚️ Spinbox Step Size Improvement
**Problem:** Arrow buttons incremented by 0.001° - too fine for practical use
- Required hundreds of clicks for meaningful adjustments
- Difficult to make quick changes
- Users preferred typing values directly

**Solution:**
- Changed step size from 0.001° to 0.1° (100x increase)
- Up/down arrows now increment by 0.1° per click
- Display precision remains 3 decimal places for fine control
- Manual typing still allows precise values (e.g., 1.234°)

**Affected Controls:**
- Global Panomap Adjustment (Yaw, Pitch, Roll)
- Stereo Offset (Yaw, Pitch, Roll)

**Impact:** ✅ Much faster and more intuitive adjustment workflow

---

## Technical Changes

### Code Quality
- Added try-except blocks in drag/drop handlers (vr180_gui.py:2044-2085)
- Fixed method resolution error in drop event
- Improved error reporting with traceback printing

### Compatibility
- All changes tested on macOS
- Windows BuildKit updated with latest code
- No breaking changes to existing functionality

---

## Upgrade Instructions

### Windows Users
1. Download `VR180_Silver_Bullet_v1.4.2_Windows.zip`
2. Extract to a folder of your choice
3. Run `VR180 Silver Bullet.exe`
4. Your settings from v1.4.1 will be preserved automatically

### macOS Users
1. Download `VR180_Silver_Bullet_v1.4.2_macOS.dmg`
2. Open the DMG and drag app to Applications folder
3. Replace existing version if prompted
4. Your settings will be preserved automatically

**Note:** First launch may require right-click → Open due to macOS Gatekeeper

---

## Known Issues

### Windows
- Hardware decode in preview may cause crashes on some systems (disabled by default)
- Some NVIDIA GPUs may require driver updates for NVENC support

### macOS
- MV-HEVC conversion requires external spatial-media-kit-tool
- Some older Macs don't support VideoToolbox 10-bit encoding

### All Platforms
- Very large files (>100GB) may cause memory issues during preview
- Some fisheye lens types not yet supported in VR180 mode

---

## What's Next?

### Planned for v1.5.0
- Batch processing support
- Preset save/load system
- Auto-alignment detection
- Enhanced preview rendering options
- Better memory management for large files

---

## Feedback & Support

Found a bug? Have a feature request?
- Report issues on GitHub
- Join our community forum
- Email: support@vr180silverbullet.com

Thank you for using VR180 Silver Bullet!

---

**Full Changelog:** See [CHANGELOG.md](CHANGELOG.md) for complete version history
