# Quick Guide: GitHub Release

## What to Do

### 1. Clean Repository
```bash
cd /Users/siyangqi/Downloads/vr180_processor

# Remove unnecessary files
rm -rf build/ dist/ __pycache__/ *.zip
rm -f vr180_gui_副本.py
rm -f BUILD_X265_MULTIVIEW.md X265_MULTIVIEW_SUMMARY.md
rm -f MV-HEVC_IMPROVEMENTS.md SPATIAL_INTEGRATION.md
```

### 2. Keep These Files

**Source Code:**
- `vr180_gui.py`
- `vr180_processor.spec`
- `requirements.txt`
- `icon.icns`, `icon.ico`
- `*.cube` (LUT files)

**Documentation:**
- `README_UPDATED.md` → rename to `README.md`
- `README_MVHEVC.md`
- `FINAL_MV-HEVC_WORKFLOW.md`
- `RELEASE_NOTES_V1.3.md` → rename to `RELEASE_NOTES.md`
- `LICENSE`

**Scripts:**
- `build_mac.sh`
- `.gitignore`

### 3. Update Files

Replace your README:
```bash
mv README.md README_OLD.md
mv README_UPDATED.md README.md
mv RELEASE_NOTES_V1.3.md RELEASE_NOTES.md
```

### 4. Create Repository

```bash
# Initialize
git init
git add .
git commit -m "Initial release v1.3.0"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/vr180-silver-bullet.git
git branch -M main
git push -u origin main
```

### 5. Build Release

```bash
# Build macOS app
./build_mac.sh

# Create ZIP
cd dist
zip -r "../VR180-Silver-Bullet-macOS-v1.3.0.zip" "VR180 Silver Bullet.app"
cd ..
```

### 6. Create GitHub Release

1. Go to your repo → **Releases** → **New release**
2. **Tag**: `v1.3.0`
3. **Title**: `VR180 Silver Bullet v1.3.0`
4. **Description**: Copy from `RELEASE_NOTES.md`
5. **Upload**: `VR180-Silver-Bullet-macOS-v1.3.0.zip`
6. **Publish**

## Attribution to Mike Swanson

### Required Mentions

✅ **In README.md** - Credits section (already included in README_UPDATED.md)
✅ **In release notes** - Credits section
✅ **In documentation** - Technical docs

### What to Say

**Short version:**
> Uses Mike Swanson's `spatial` CLI tool for MV-HEVC encoding
> https://blog.mikeswanson.com/spatial/

**Full version** (in README Credits):
```markdown
### spatial by Mike Swanson

The Vision Pro MV-HEVC encoding capability is powered by Mike Swanson's `spatial` CLI tool.

- **Author**: Mike Swanson
- **Project**: https://blog.mikeswanson.com/spatial/
- **Purpose**: MV-HEVC encoding with Vision Pro metadata
- **Bundled**: Yes (included in macOS app)

Special thanks to Mike Swanson for creating the `spatial` tool and documenting
the MV-HEVC encoding workflow.
```

### Where to Mention

1. **README.md** → Credits section ✅
2. **RELEASE_NOTES.md** → Credits section ✅
3. **GitHub Release description** → Credits line ✅
4. **About dialog** (optional) → Consider adding to app

## Quick Checklist

- [ ] Clean up repository (remove build files)
- [ ] Update README.md with spatial attribution
- [ ] Update RELEASE_NOTES.md
- [ ] Create .gitignore (prevent committing build artifacts)
- [ ] Initialize git repository
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Build macOS app
- [ ] Create ZIP file
- [ ] Create GitHub release (tag v1.3.0)
- [ ] Upload ZIP to release
- [ ] Add credits to Mike Swanson in release notes
- [ ] Publish release

## Important Notes

### What NOT to Commit
- ❌ `build/`, `dist/` folders
- ❌ ZIP files (only in releases)
- ❌ `__pycache__/`
- ❌ Old/duplicate documentation
- ❌ Temporary files

### License Considerations
- `spatial` - Free tool by Mike Swanson (check his license)
- `FFmpeg` - LGPL/GPL (bundling binaries is OK)
- Your app - Choose: MIT, GPL, Apache, etc.

### After Publishing
1. Test download works
2. Verify app runs from ZIP
3. Update repo description on GitHub
4. Add topics: `vr180`, `vision-pro`, `mvhevc`, `video-processing`

## Need Help?

See full instructions in: `GITHUB_RELEASE_INSTRUCTIONS.md`
