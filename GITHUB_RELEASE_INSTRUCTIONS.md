# GitHub Release Instructions

## Step 1: Prepare Repository

### Clean Up Unnecessary Files

Remove these files from your repository (they shouldn't be committed):

```bash
# Navigate to your project
cd /Users/siyangqi/Downloads/vr180_processor

# Remove build artifacts and temp files
rm -rf build/ dist/ __pycache__/
rm -rf *.zip
rm -rf github_release/
rm -rf windows_build_pack/
rm -rf windows-build-files/

# Remove old/duplicate documentation
rm -f BUILD.md BUILD_WINDOWS.md DISTRIBUTION.md DISTRIBUTION_SUMMARY.md
rm -f PACKAGING_QUICK_START.md PLATFORM_FEATURES.md
rm -f BUILD_X265_MULTIVIEW.md X265_MULTIVIEW_SUMMARY.md
rm -f MV-HEVC_IMPROVEMENTS.md SPATIAL_INTEGRATION.md
rm -f GITHUB_CHECKLIST.md GITHUB_RELEASE_GUIDE.md
rm -f vr180_gui_副本.py
rm -f *.txt  # Remove all .txt files (keep only .md and source files)
```

### Files to Keep in Repository

**Essential Source Files:**
- `vr180_gui.py` - Main application
- `vr180_processor.spec` - PyInstaller build config
- `requirements.txt` - Python dependencies
- `icon.icns` - macOS icon
- `icon.ico` - Windows icon
- `*.cube` - LUT files (optional, include if you want)

**Build Scripts:**
- `build_mac.sh` - macOS build script
- `create_icon.py` - Icon creation script (optional)

**Documentation:**
- `README.md` - Main readme (update with attribution)
- `README_MVHEVC.md` - MV-HEVC guide
- `FINAL_MV-HEVC_WORKFLOW.md` - Technical details
- `LICENSE` - Your license file
- `RELEASE_NOTES.md` - Version history

**Config:**
- `.gitignore` - Git ignore rules

## Step 2: Update README.md with Attribution

The README should mention Mike Swanson's spatial tool. Here's what to add:

### Credits Section

```markdown
## Credits & Dependencies

### Third-Party Tools

This application integrates the following external tools:

#### spatial CLI by Mike Swanson
- **Purpose**: MV-HEVC encoding with Vision Pro metadata
- **Author**: Mike Swanson
- **Website**: https://blog.mikeswanson.com/spatial/
- **License**: Free for use
- **Bundled**: Yes (included in macOS app)

The `spatial` tool uses Apple's VideoToolbox framework to encode MV-HEVC video with proper APMP (Apple Projected Media Profile) metadata for Vision Pro compatibility.

#### FFmpeg
- **Purpose**: Video processing and ProRes encoding
- **Project**: FFmpeg Project
- **Website**: https://ffmpeg.org/
- **License**: LGPL 2.1+ / GPL 2+
- **Bundled**: Yes (included in app)

### Acknowledgments

Special thanks to:
- **Mike Swanson** for creating the excellent `spatial` CLI tool and documenting the MV-HEVC encoding process
- **FFmpeg Project** for the powerful media processing framework
- **x265 Project** for HEVC encoding capabilities
```

## Step 3: Create Clean Repository

```bash
# Initialize git repository if not already done
git init

# Add files
git add vr180_gui.py vr180_processor.spec requirements.txt
git add icon.icns icon.ico
git add build_mac.sh
git add README.md README_MVHEVC.md FINAL_MV-HEVC_WORKFLOW.md
git add LICENSE RELEASE_NOTES.md
git add .gitignore

# Add LUT files if you want to include them
git add *.cube

# Commit
git commit -m "Initial release v1.3.0 - VR180 Silver Bullet with MV-HEVC support"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Create repository:
   - **Name**: `vr180-silver-bullet` (or your preferred name)
   - **Description**: "Professional VR180 video processor with Vision Pro MV-HEVC support"
   - **Public** or **Private**: Your choice
   - **Don't** initialize with README (you already have one)

3. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/vr180-silver-bullet.git
git branch -M main
git push -u origin main
```

## Step 5: Build Release Binaries

### macOS App

```bash
# Build the macOS app
./build_mac.sh

# Create ZIP for release
cd dist
zip -r "VR180-Silver-Bullet-macOS-v1.3.0.zip" "VR180 Silver Bullet.app"
```

### Windows App (if you have Windows build)

If you have a Windows build ready:
```bash
zip -r "VR180-Silver-Bullet-Windows-v1.3.0.zip" "VR180 Silver Bullet/"
```

## Step 6: Create GitHub Release

1. Go to your repository on GitHub
2. Click **Releases** → **Create a new release**
3. Fill in:

**Tag version**: `v1.3.0`

**Release title**: `VR180 Silver Bullet v1.3.0 - Vision Pro MV-HEVC Support`

**Description**:
```markdown
# VR180 Silver Bullet v1.3.0

Professional VR180 video processor with full Vision Pro MV-HEVC encoding support.

## ✨ Key Features

- **Vision Pro MV-HEVC Encoding**: Full bitrate control (350+ Mbps) with proper spatial metadata
- **IPD Adjustment**: Fine-tune stereo convergence for comfortable viewing
- **VR180 Adjustments**: Per-eye rotation, color grading, LUT support
- **Hardware Accelerated**: Uses Apple VideoToolbox for fast encoding
- **Lossless Workflow**: ProRes intermediate prevents quality loss

## 🎯 What's New in v1.3.0

- Integrated Mike Swanson's `spatial` CLI tool for MV-HEVC encoding
- Automatic 350 Mbps bitrate when codec set to "auto"
- Proper Vision Pro APMP metadata (cdist=65mm, hfov=180°)
- ProRes intermediate workflow (no double lossy compression)
- Simplified and optimized encoding pipeline

## 📦 Downloads

### macOS (Apple Silicon + Intel)
- **VR180-Silver-Bullet-macOS-v1.3.0.zip** - Complete app bundle with all tools

### Windows
- **VR180-Silver-Bullet-Windows-v1.3.0.zip** - Standalone executable

## 📋 Requirements

### macOS
- macOS 11.0 (Big Sur) or later
- ~40GB free disk space per minute of 4K video (temporary)

### Windows
- Windows 10/11
- Visual C++ Runtime (included in installer)

## 🚀 Quick Start

1. Download and extract the app
2. Open the application
3. Load your VR180 video
4. Adjust settings (IPD, rotation, color, etc.)
5. Select output codec (Auto = 350 Mbps MV-HEVC)
6. For Vision Pro: Enable "Vision Pro MV-HEVC" mode
7. Click Start!

## 📖 Documentation

- [MV-HEVC Quick Guide](README_MVHEVC.md)
- [Technical Workflow Details](FINAL_MV-HEVC_WORKFLOW.md)
- [Release Notes](RELEASE_NOTES.md)

## 🙏 Credits

**spatial CLI** by Mike Swanson - https://blog.mikeswanson.com/spatial/
- MV-HEVC encoding with Vision Pro metadata

**FFmpeg Project** - https://ffmpeg.org/
- Video processing framework

## 📄 License

[Your License Here - MIT/GPL/etc.]

## 🐛 Known Issues

- MV-HEVC encoding requires significant disk space (temporary files)
- Windows version doesn't support Vision Pro features (macOS only)

## 💬 Support

For issues, questions, or feedback:
- Open an issue on GitHub
- See documentation for troubleshooting
```

4. **Attach Files**: Upload your ZIP files:
   - `VR180-Silver-Bullet-macOS-v1.3.0.zip`
   - `VR180-Silver-Bullet-Windows-v1.3.0.zip` (if available)

5. Click **Publish release**

## Step 7: Update Repository README.md

Make sure your main README.md includes:

1. **Clear description** of what the app does
2. **Screenshots** (if available)
3. **Installation instructions**
4. **Feature list**
5. **Credits section** (especially spatial tool attribution)
6. **Link to releases**

Example credits section:

```markdown
## Credits

This project integrates and builds upon several excellent open-source tools:

### spatial by Mike Swanson
The MV-HEVC encoding functionality is powered by Mike Swanson's `spatial` CLI tool.
- Project: https://blog.mikeswanson.com/spatial/
- Documentation: https://blog.mikeswanson.com/spatial_docs/
- The tool provides Apple VideoToolbox-based MV-HEVC encoding with proper Vision Pro metadata

### FFmpeg
Video processing and ProRes encoding provided by the FFmpeg project.
- Project: https://ffmpeg.org/
- Used under LGPL/GPL license

### Acknowledgments
Special thanks to Mike Swanson for his excellent work on the spatial tool and comprehensive documentation on MV-HEVC encoding workflows.
```

## Important Notes

### Attribution to Mike Swanson

**Required mentions:**
1. In README.md - Credits section
2. In app documentation
3. In release notes
4. Consider adding to app About dialog

**What to say:**
- "Uses spatial CLI tool by Mike Swanson"
- Link to his blog: https://blog.mikeswanson.com/spatial/
- Mention it's bundled in the app
- Note that it uses Apple VideoToolbox

### Licensing Considerations

**spatial CLI**: Check Mike Swanson's license (appears to be free/open)
**FFmpeg**: LGPL/GPL - you're distributing binaries, which is allowed
**Your app**: Choose your own license (MIT, GPL, etc.)

### What NOT to Include in Repository

❌ Build artifacts (`build/`, `dist/`)
❌ ZIP files of releases
❌ Large binary files (keep in releases only)
❌ Temporary files (`.pyc`, `__pycache__`)
❌ Personal files or API keys
❌ Old/duplicate documentation files

### .gitignore Contents

Make sure your `.gitignore` includes:
```
# Build artifacts
build/
dist/
*.egg-info/

# Python
__pycache__/
*.py[cod]
*$py.class

# PyInstaller
*.spec.bak

# IDE
.vscode/
.idea/

# macOS
.DS_Store

# Temporary
*.tmp
*.log

# Release files
*.zip
github_release/
windows_build_pack/
```

## Checklist

Before publishing:

- [ ] Clean up repository (remove unnecessary files)
- [ ] Update README.md with spatial attribution
- [ ] Add Credits section to README
- [ ] Test build scripts work
- [ ] Create release binaries
- [ ] Test the app works from ZIP
- [ ] Write clear release notes
- [ ] Tag version correctly (v1.3.0)
- [ ] Upload all necessary files
- [ ] Link to Mike Swanson's blog in documentation

## After Publishing

1. Test download links work
2. Verify ZIP files extract correctly
3. Test app runs on clean machine
4. Update project description on GitHub
5. Add topics/tags: `vr180`, `vision-pro`, `mvhevc`, `video-processing`
6. Consider creating a GitHub Pages site with documentation
