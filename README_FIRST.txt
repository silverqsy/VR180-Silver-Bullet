╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              Silver's VR180 Tool - Windows Build Package                  ║
║                                                                            ║
║                         QUICK START GUIDE                                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

YOU ARE SEEING "DLL load failed" ERROR?
═══════════════════════════════════════════════════════════════════════════

This error (找不到指定的程序) is a PyQt6 compatibility issue!

┌────────────────────────────────────────────────────────────────────────────┐
│  IF YOU ALREADY HAVE VC++ RUNTIME INSTALLED:                              │
│                                                                            │
│  Double-click:  try_different_pyqt6.bat                                   │
│                                                                            │
│  This will automatically:                                                 │
│  - Try different PyQt6 versions (6.4, 6.5, 6.6, latest)                  │
│  - Find one that works with your Python                                   │
│  - Build the app with the working version                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  IF YOU DON'T HAVE VC++ RUNTIME:                                          │
│                                                                            │
│  1. Double-click:  INSTALL_VC_RUNTIME.bat                                 │
│  2. RESTART YOUR COMPUTER                                                 │
│  3. Double-click:  try_different_pyqt6.bat                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘


ALTERNATIVE: Manual Installation
═══════════════════════════════════════════════════════════════════════════

If the batch file doesn't work:

1. Open your web browser
2. Go to: https://aka.ms/vs/17/release/vc_redist.x64.exe
3. Download and run the installer
4. Click "Install"
5. RESTART YOUR COMPUTER
6. Run: fix_dll_error.bat


WHAT IS VISUAL C++ RUNTIME?
═══════════════════════════════════════════════════════════════════════════

- FREE Microsoft package
- Required by many Windows apps (including PyQt6)
- Totally safe - from Microsoft
- About 15MB download
- Takes 1 minute to install


STILL NOT WORKING?
═══════════════════════════════════════════════════════════════════════════

Try these in order:

1. Check Python version:
   python --version
   Must be: 3.9, 3.10, or 3.11 (NOT 3.12 or 3.13!)

2. If wrong Python version:
   See: PYTHON_DOWNGRADE_GUIDE.txt

3. For detailed help:
   See: DLL_ERROR_FIX.txt


FILES IN THIS PACKAGE
═══════════════════════════════════════════════════════════════════════════

MOST IMPORTANT (for DLL errors):
  try_different_pyqt6.bat  ← TRY THIS FIRST! Tests multiple PyQt6 versions
  INSTALL_VC_RUNTIME.bat   ← If you don't have VC++ Runtime
  fix_dll_error.bat        ← Alternative fix script

OTHER SCRIPTS:
  fix_windows_build.bat    - For NumPy errors
  check_dependencies.bat   - Check what's installed
  build_windows.bat        - Standard build

DOCUMENTATION:
  README_FIRST.txt         ← YOU ARE HERE
  START_HERE.txt           - Detailed instructions
  DLL_ERROR_FIX.txt        - DLL error troubleshooting
  QUICK_FIX.txt            - Quick solutions
  PYTHON_DOWNGRADE_GUIDE.txt - Downgrade Python


═══════════════════════════════════════════════════════════════════════════

              REMEMBER: After installing Visual C++ Runtime,
                    YOU MUST RESTART YOUR COMPUTER!

═══════════════════════════════════════════════════════════════════════════
