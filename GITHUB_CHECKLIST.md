# GitHub Distribution Checklist

## ☐ Before You Start

- [ ] Have a GitHub account (create at https://github.com/join)
- [ ] Know your GitHub username
- [ ] Have git installed (`git --version` to check)

## ☐ Step 1: Create Repository

- [ ] Go to https://github.com/new
- [ ] Repository name: `silvers-vr180-tool` (or your choice)
- [ ] Description: `Professional VR180 video processing tool with real-time preview and LUT support`
- [ ] **Public** repository (for open source) OR **Private** (if you want to control access)
- [ ] **DO NOT** check "Add a README file"
- [ ] **DO NOT** check "Add .gitignore"
- [ ] **DO NOT** check "Choose a license"
- [ ] Click "Create repository"
- [ ] **Keep the page open** - you'll need the URL

## ☐ Step 2: Upload Code

Choose ONE method:

### Method A: Automated Script (Recommended)
- [ ] Open Terminal in `/Users/siyangqi/Downloads/vr180_processor`
- [ ] Run: `./upload_to_github.sh`
- [ ] Enter your GitHub username when prompted
- [ ] Enter repository name when prompted
- [ ] Enter password or Personal Access Token
- [ ] Wait for upload to complete

### Method B: Manual Git Commands
- [ ] Open Terminal in `/Users/siyangqi/Downloads/vr180_processor`
- [ ] Run: `git init`
- [ ] Run: `git add .`
- [ ] Run: `git commit -m "Initial release v1.0.0"`
- [ ] Run: `git branch -M main`
- [ ] Run: `git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git`
- [ ] Run: `git push -u origin main`

### Method C: Web Upload (No Git Required)
- [ ] On your new repository page, click "uploading an existing file"
- [ ] Drag ALL files from `/Users/siyangqi/Downloads/vr180_processor` EXCEPT:
  - [ ] Skip `dist/` folder
  - [ ] Skip `build/` folder
  - [ ] Skip `__pycache__/` folders
  - [ ] Skip `.pyc` files
- [ ] Scroll down, click "Commit changes"

## ☐ Step 3: Verify Upload

- [ ] Go to your repository URL: `https://github.com/YOUR_USERNAME/YOUR_REPO`
- [ ] Check that you see:
  - [ ] `README.md` is displayed on the main page
  - [ ] `vr180_gui.py` is visible in file list
  - [ ] `LICENSE` file is present
  - [ ] `requirements.txt` is present
  - [ ] All build scripts (.bat, .sh) are present

## ☐ Step 4: Create Release

- [ ] On repository page, click "Releases" (right sidebar)
- [ ] Click "Create a new release"
- [ ] **Choose a tag:**
  - [ ] Type: `v1.0.0`
  - [ ] Click "Create new tag: v1.0.0 on publish"
- [ ] **Release title:** `Silver's VR180 Tool v1.0.0`
- [ ] **Description:**
  - [ ] Open `/Users/siyangqi/Downloads/vr180_processor/RELEASE_NOTES.md`
  - [ ] Copy the entire contents
  - [ ] Paste into the description box
- [ ] **Attach files:**
  - [ ] Click "Attach binaries by dropping them here"
  - [ ] Upload: `/Users/siyangqi/Downloads/vr180_processor/dist/Silvers-VR180-Tool-macOS.zip`
  - [ ] Upload: `/Users/siyangqi/Downloads/Silvers-VR180-Tool-Windows-BuildPackage.zip`
  - [ ] Wait for uploads to complete (progress bars)
- [ ] **Options:**
  - [ ] Check "Set as the latest release"
  - [ ] Leave "Set as a pre-release" UNCHECKED
- [ ] Click "Publish release"

## ☐ Step 5: Test the Release

- [ ] Go to your repository's main page
- [ ] Click "Releases" → "v1.0.0"
- [ ] Verify both files are downloadable:
  - [ ] Click on `Silvers-VR180-Tool-macOS.zip` - should download
  - [ ] Click on `Silvers-VR180-Tool-Windows-BuildPackage.zip` - should download
- [ ] Check file sizes match:
  - [ ] macOS: ~63 MB
  - [ ] Windows: ~38 KB

## ☐ Step 6: Share Your Repository

Your repository is now live! Share the link:

**Repository:** `https://github.com/YOUR_USERNAME/YOUR_REPO`

**Direct Release Downloads:**
- macOS: `https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/Silvers-VR180-Tool-macOS.zip`
- Windows: `https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/Silvers-VR180-Tool-Windows-BuildPackage.zip`

Share on:
- [ ] Social media
- [ ] Reddit (r/VR180, r/virtualreality)
- [ ] VR180 creator forums
- [ ] Email to friends

## ☐ Optional Enhancements

- [ ] Add screenshots:
  - [ ] Take screenshots of the app
  - [ ] Upload to repository
  - [ ] Update README.md to include them
- [ ] Add topics/tags:
  - [ ] Go to repository → Settings
  - [ ] Add topics: `vr180`, `video-processing`, `pyqt6`, `ffmpeg`
- [ ] Add repository description:
  - [ ] Click ⚙️ next to "About" on repository page
  - [ ] Add description and website
- [ ] Enable Discussions:
  - [ ] Settings → Features → Discussions
  - [ ] Let users ask questions
- [ ] Add GitHub Pages:
  - [ ] Settings → Pages
  - [ ] Source: main branch
  - [ ] Your README becomes a website!

## ☐ Troubleshooting

### If upload fails:
- [ ] Check you entered correct username/password
- [ ] Use Personal Access Token instead of password:
  - [ ] Go to: https://github.com/settings/tokens
  - [ ] Generate new token (classic)
  - [ ] Check "repo" scope
  - [ ] Use token as password

### If files are missing:
- [ ] Check .gitignore isn't excluding them
- [ ] Make sure you uploaded from correct folder

### If release assets won't upload:
- [ ] File size limit: 2 GB per file (you're well under)
- [ ] Try uploading one at a time
- [ ] Make sure files aren't corrupted

## ✅ Done!

Congratulations! Your app is now on GitHub and ready for the world to use!

**Next:** Tell people about it! Post on:
- Reddit
- Twitter/X
- VR/360 video forums
- YouTube community posts (if you have a channel)

---

Need help? Check:
- [ ] GITHUB_RELEASE_GUIDE.md (detailed instructions)
- [ ] https://docs.github.com/en/repositories (GitHub docs)
- [ ] Ask me if you're stuck!
