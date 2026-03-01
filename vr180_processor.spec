# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import shutil
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

block_cipher = None

# Detect platform
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'

# Collect OpenCV files - simplified approach to avoid recursion issues
cv2_datas = []
cv2_binaries = []
cv2_hiddenimports = ['cv2', 'cv2.cv2']

# Manual collection to avoid recursion
if IS_WINDOWS:
    try:
        import cv2
        import glob
        cv2_path = Path(cv2.__file__).parent

        # Collect all DLLs and PYDs from cv2 directory
        for ext in ['*.dll', '*.pyd']:
            for file_path in cv2_path.glob(ext):
                cv2_binaries.append((str(file_path), 'cv2'))

        # Also check python subdirectory
        for py_dir in cv2_path.glob('python-3.*'):
            for ext in ['*.dll', '*.pyd']:
                for file_path in py_dir.glob(ext):
                    cv2_binaries.append((str(file_path), 'cv2'))

        print(f"✓ Collected {len(cv2_binaries)} OpenCV binaries (manual collection)")
    except Exception as e:
        print(f"⚠ Warning: Could not collect OpenCV: {e}")
else:
    # On macOS, use standard collection
    try:
        tmp_datas, tmp_binaries, tmp_hiddenimports = collect_all('cv2')
        cv2_datas += tmp_datas
        cv2_binaries += tmp_binaries
        cv2_hiddenimports += tmp_hiddenimports
        print(f"✓ Collected OpenCV: {len(tmp_binaries)} binaries, {len(tmp_datas)} data files")
    except Exception as e:
        print(f"⚠ Warning: Could not collect OpenCV: {e}")

# Find FFmpeg binaries
ffmpeg_path = shutil.which('ffmpeg')
ffprobe_path = shutil.which('ffprobe')

binaries = []
if ffmpeg_path:
    binaries.append((ffmpeg_path, '.'))
    print(f"✓ Found ffmpeg: {ffmpeg_path}")
else:
    print("⚠ Warning: ffmpeg not found in PATH")

if ffprobe_path:
    binaries.append((ffprobe_path, '.'))
    print(f"✓ Found ffprobe: {ffprobe_path}")
else:
    print("⚠ Warning: ffprobe not found in PATH")

# Add OpenCV binaries
binaries.extend(cv2_binaries)

# Bundle spatial CLI tool for Vision Pro MV-HEVC support (macOS only)
spatial_path = shutil.which('spatial')
if spatial_path and IS_MACOS:
    binaries.append((spatial_path, '.'))
    print(f"✓ Found spatial CLI tool: {spatial_path}")
else:
    if IS_MACOS:
        print("⚠ Warning: spatial CLI tool not found - Vision Pro MV-HEVC will not work")
        print("  Install with: brew install mikeswanson/spatial/spatial-media-kit-tool")

# Collect spatialmedia module (bundled with the application)
spatialmedia_datas = []
try:
    # The spatialmedia folder should be in the same directory as vr180_gui.py
    spatialmedia_path = Path('spatialmedia')
    if spatialmedia_path.exists():
        # Collect all Python files from spatialmedia module
        for py_file in spatialmedia_path.rglob('*.py'):
            rel_path = py_file.relative_to('.')
            dest_dir = str(rel_path.parent)
            spatialmedia_datas.append((str(py_file), dest_dir))
        print(f"✓ Collected spatialmedia module: {len(spatialmedia_datas)} files")
    else:
        print("⚠ Warning: spatialmedia module not found - VR180 metadata injection may not work")
except Exception as e:
    print(f"⚠ Warning: Could not collect spatialmedia: {e}")

a = Analysis(
    ['vr180_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=cv2_datas + spatialmedia_datas,
    hiddenimports=[
        'PyQt6.sip',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'spatialmedia',
        'spatialmedia.metadata_utils',
        'spatialmedia.mpeg',
        'spatialmedia.mpeg.box',
        'spatialmedia.mpeg.container',
        'spatialmedia.mpeg.constants',
        'spatialmedia.mpeg.st3d',
        'spatialmedia.mpeg.sv3d',
    ] + cv2_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VR180Processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True if IS_WINDOWS else False,  # Enable console on Windows for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if IS_WINDOWS else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VR180 Silver Bullet' if IS_WINDOWS else 'VR180Processor',
)

# macOS app bundle (only on macOS)
if IS_MACOS:
    app = BUNDLE(
        coll,
        name="VR180 Silver Bullet.app",
        icon='icon.icns',
        bundle_identifier='com.vr180silverbullet.app',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.0.0',
        },
    )
