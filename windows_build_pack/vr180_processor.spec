# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import shutil
from pathlib import Path

block_cipher = None

# Detect platform
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'

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

a = Analysis(
    ['vr180_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=[
        'PyQt6.sip',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
    ],
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
    console=False,
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
