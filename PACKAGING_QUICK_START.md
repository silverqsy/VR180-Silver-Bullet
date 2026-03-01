# VR180 Processor - Quick Packaging Guide

## TL;DR - Quick Commands

### macOS Build (what you're on now)
```bash
# Simple build
./build_mac.sh

# Or create ready-to-distribute package
./create_release.sh --version 1.0.0

# With FFmpeg bundled (recommended for end users)
./create_release.sh --version 1.0.0 --bundle-ffmpeg
```

### Windows Build (run on Windows machine)
```cmd
build_windows.bat
```

## What Gets Created

### macOS
- **dist/VR180 Processor.app** - The application bundle (86MB)
- **VR180-Processor-X.Y.Z-macOS.zip** - Distribution package

### Windows
- **dist/VR180Processor/** - Folder with exe and dependencies
- Need to manually zip this folder for distribution

## File Structure After Build

```
vr180_processor/
├── dist/
│   ├── VR180 Processor.app (macOS)
│   └── VR180Processor/ (Windows)
├── build/ (temporary build files)
├── vr180_gui.py (source)
├── vr180_cli.py (source)
├── vr180_processor.spec (build config)
├── build_mac.sh (macOS build script)
├── build_windows.bat (Windows build script)
└── create_release.sh (creates distribution package)
```

## Distribution Checklist

- [ ] Build application
- [ ] Test on clean system (without Python installed)
- [ ] Decide: bundle FFmpeg or require installation?
- [ ] Create ZIP package
- [ ] Write release notes
- [ ] Upload to distribution platform

## Testing Your Build

### macOS
```bash
# Launch the app
open "dist/VR180 Processor.app"

# Or from terminal to see errors
"dist/VR180 Processor.app/Contents/MacOS/VR180Processor"
```

### Windows
```cmd
# Launch the app
dist\VR180Processor\VR180Processor.exe
```

## Common Issues

### "FFmpeg not found"
**Solution:** Install FFmpeg or bundle it with your app
- macOS: `brew install ffmpeg`
- Windows: Download from ffmpeg.org

### macOS: "App is damaged"
**Solution:** Code sign or tell users to run:
```bash
xattr -cr "VR180 Processor.app"
```

### Windows: "Missing VCRUNTIME140.dll"
**Solution:** Install Visual C++ Redistributable
- PyInstaller should bundle this automatically
- If not, download from Microsoft

## Size Comparison

| Configuration | Size |
|--------------|------|
| macOS (no FFmpeg) | ~86MB |
| macOS (with FFmpeg) | ~200MB |
| Windows (no FFmpeg) | ~150MB |
| Windows (with FFmpeg) | ~250MB |

## License Reminder

When distributing:
1. Include LICENSE file (MIT)
2. If bundling FFmpeg, include FFmpeg license (LGPL/GPL)
3. PyQt6 is GPL (or buy commercial license if closed-source)

## Quick Distribution

Upload to GitHub:
```bash
# Create release on GitHub
gh release create v1.0.0 \
  --title "VR180 Processor v1.0.0" \
  --notes "Initial release with VR180 processing features" \
  VR180-Processor-1.0.0-macOS.zip
```

## Need Help?

- **Build issues:** See BUILD.md
- **Distribution options:** See DISTRIBUTION.md
- **User documentation:** See README.md

---

**Current Status:** ✅ macOS build tested and working!
- Build size: 86MB
- Location: dist/VR180 Processor.app
- Ready to distribute (with FFmpeg requirement)
