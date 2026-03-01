#!/bin/bash
# Quick script to upload Silver's VR180 Tool to GitHub

echo "=========================================="
echo "Silver's VR180 Tool - GitHub Upload"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed!"
    echo "Install from: https://git-scm.com/downloads"
    exit 1
fi

echo "Step 1: Create a new repository on GitHub"
echo "  1. Go to: https://github.com/new"
echo "  2. Name: silvers-vr180-tool (or your choice)"
echo "  3. Make it Public"
echo "  4. DO NOT initialize with README"
echo "  5. Click 'Create repository'"
echo ""
read -p "Press Enter when you've created the repository..."

echo ""
read -p "Enter your GitHub username: " github_user
read -p "Enter repository name (e.g., silvers-vr180-tool): " repo_name

echo ""
echo "Step 2: Initializing git repository..."

# Initialize git if not already
if [ ! -d .git ]; then
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add all files
echo ""
echo "Step 3: Adding files..."
git add .

# Create commit
echo ""
echo "Step 4: Creating commit..."
git commit -m "Initial release of Silver's VR180 Tool v1.0.0

- VR180 video processing with panomap adjustments
- LUT support with intensity control
- Real-time preview
- Hardware-accelerated encoding
- Standalone builds for macOS and Windows
- Fixed console windows on Windows
- Fixed PyQt6 compatibility issues"

# Add remote
echo ""
echo "Step 5: Adding GitHub remote..."
git remote add origin "https://github.com/${github_user}/${repo_name}.git" 2>/dev/null || \
git remote set-url origin "https://github.com/${github_user}/${repo_name}.git"

# Set branch name
git branch -M main

# Push to GitHub
echo ""
echo "Step 6: Pushing to GitHub..."
echo "You may be asked for your GitHub username and password/token"
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS! Code uploaded to GitHub"
    echo "=========================================="
    echo ""
    echo "Repository URL:"
    echo "https://github.com/${github_user}/${repo_name}"
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/${github_user}/${repo_name}/releases"
    echo "2. Click 'Create a new release'"
    echo "3. Tag: v1.0.0"
    echo "4. Title: Silver's VR180 Tool v1.0.0"
    echo "5. Copy description from RELEASE_NOTES.md"
    echo "6. Upload files:"
    echo "   - dist/Silvers-VR180-Tool-macOS.zip"
    echo "   - Silvers-VR180-Tool-Windows-BuildPackage.zip"
    echo "7. Click 'Publish release'"
    echo ""
else
    echo ""
    echo "ERROR: Push failed!"
    echo ""
    echo "Common issues:"
    echo "1. Wrong username/password"
    echo "2. Use Personal Access Token instead of password:"
    echo "   https://github.com/settings/tokens"
    echo "3. Repository doesn't exist"
    echo ""
    echo "Try again or upload manually:"
    echo "https://github.com/${github_user}/${repo_name}/upload"
fi
