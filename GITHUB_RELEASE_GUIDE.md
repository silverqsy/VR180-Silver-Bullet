# GitHub Release Guide for Silver's VR180 Tool

## Step-by-Step Instructions

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `silvers-vr180-tool` (or your preferred name)
3. Description: `Professional VR180 video processing tool with real-time preview and LUT support`
4. Choose **Public** (or Private if you prefer)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### 2. Prepare Your Local Repository

Open Terminal/Command Prompt in the project folder and run:

```bash
cd /Users/siyangqi/Downloads/vr180_processor

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial release of Silver's VR180 Tool v1.0.0

- VR180 video processing with panomap adjustments
- LUT support with intensity control
- Real-time preview
- Hardware-accelerated encoding
- Standalone builds for macOS and Windows"

# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/silvers-vr180-tool.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Create GitHub Release

1. Go to your repository on GitHub
2. Click "Releases" (right sidebar)
3. Click "Create a new release"
4. Click "Choose a tag" → type `v1.0.0` → click "Create new tag"
5. Release title: `Silver's VR180 Tool v1.0.0`
6. Description: (copy from RELEASE_NOTES.md - I'll create this)
7. **Upload release assets**:
   - `dist/Silvers-VR180-Tool-macOS.zip` (the Mac app)
   - `Silvers-VR180-Tool-Windows-BuildPackage.zip` (Windows build package)
8. Check "Set as the latest release"
9. Click "Publish release"

### 4. Update README

The README.md in your repository will show:
- App features
- Download links (automatically from releases)
- Installation instructions
- Screenshots (if you add them)

### 5. Add .gitignore

A .gitignore file is already created to exclude:
- Build artifacts (build/, dist/)
- Python cache files
- macOS files (.DS_Store)
- IDE files

---

## What to Upload

### For Users to Download:

1. **macOS App** (from GitHub Releases):
   - File: `Silvers-VR180-Tool-macOS.zip` (63 MB)
   - Users download, unzip, and run
   - No build required

2. **Windows Build Package** (from GitHub Releases):
   - File: `Silvers-VR180-Tool-Windows-BuildPackage.zip` (38 KB)
   - Contains source + build scripts
   - Users need to build on Windows (instructions included)

### Source Code:

GitHub automatically provides source code downloads in .zip and .tar.gz formats.

---

## Repository Structure

Your GitHub repo will have:

```
silvers-vr180-tool/
├── README.md                    # Main documentation (users see this)
├── LICENSE                      # License file (I'll help create)
├── .gitignore                   # Git ignore rules
├── vr180_gui.py                 # Main application
├── vr180_processor.spec         # PyInstaller config
├── requirements.txt             # Python dependencies
├── build_mac.sh                 # macOS build script
├── build_windows.bat            # Windows build script
├── fix_dll_error.bat            # Windows DLL fix
├── try_different_pyqt6.bat     # PyQt6 version tester
├── check_dependencies.bat       # Dependency checker
├── BUILD_INSTRUCTIONS.md        # Build guide
├── WINDOWS_BUILD_FIX.md        # Windows troubleshooting
├── DLL_ERROR_FIX.txt           # DLL error solutions
├── QUICK_FIX.txt               # Quick fixes
├── PYTHON_DOWNGRADE_GUIDE.txt  # Python downgrade guide
└── dist/                        # (gitignored - not uploaded to repo)
    └── Silvers-VR180-Tool-macOS.zip
```

---

## Alternative: Quick Upload (No Git)

If you don't want to use git commands:

1. Create repository on GitHub (empty)
2. Click "uploading an existing file"
3. Drag and drop all files EXCEPT:
   - `dist/` folder
   - `build/` folder
   - `__pycache__/` folders
   - `.pyc` files
4. Commit the files
5. Then create a Release and upload the .zip files there

---

## Tips

### Add Screenshots

1. Run the app on Mac
2. Take screenshots of the main window
3. Save as `screenshot1.png`, `screenshot2.png`
4. Add to repository
5. Reference in README.md:
   ```markdown
   ![Main Window](screenshot1.png)
   ```

### Add a License

Common choices:
- **MIT License** - Very permissive, anyone can use
- **GPL v3** - Open source, derivative works must be open source
- **Proprietary** - All rights reserved

I can create a LICENSE file for you if you tell me which you prefer.

### Enable GitHub Pages (Optional)

Turn your README into a website:
1. Go to Settings → Pages
2. Source: Deploy from branch → main → /root
3. Your README becomes a website!

---

## Updating Later

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push

# For new version:
git tag v1.0.1
git push origin v1.0.1
# Then create new release on GitHub
```

---

## Getting Help

If you get stuck:
1. Check GitHub's guide: https://docs.github.com/en/repositories/creating-and-managing-repositories
2. Watch a tutorial on YouTube: "How to upload to GitHub"
3. Or tell me where you're stuck and I'll help!

---

## Do You Want Me To...

- [ ] Create a LICENSE file?
- [ ] Write release notes?
- [ ] Create a .gitignore file?
- [ ] Write a CONTRIBUTING.md guide?
- [ ] Create issue templates?

Just let me know what you need!
