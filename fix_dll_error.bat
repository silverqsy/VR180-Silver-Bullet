@echo off
REM Fix DLL load failed error for PyQt6

echo ===================================================
echo Silver's VR180 Tool - DLL Error Fix
echo ===================================================
echo.
echo This script fixes "DLL load failed" errors with PyQt6
echo.

REM Step 1: Check Python version
echo Step 1: Checking Python version...
python --version
echo.
echo IMPORTANT: You MUST use Python 3.9, 3.10, or 3.11
echo Python 3.12+ has compatibility issues!
echo.
pause

REM Step 2: Completely remove all PyQt6 packages
echo Step 2: Removing ALL PyQt6 packages...
python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-WebEngine PyQt6-WebEngine-Qt6
python -m pip uninstall -y numpy pyinstaller

echo.
echo Step 3: Clean pip cache...
python -m pip cache purge

echo.
echo Step 4: Install packages in CORRECT order with specific versions...
echo.
echo Installing PyQt6-sip first (required dependency)...
python -m pip install PyQt6-sip==13.6.0

echo.
echo Installing PyQt6-Qt6 (Qt libraries)...
python -m pip install PyQt6-Qt6==6.6.0

echo.
echo Installing PyQt6 (Python bindings)...
python -m pip install PyQt6==6.6.0

echo.
echo Installing NumPy...
python -m pip install numpy==1.26.4

echo.
echo Installing PyInstaller...
python -m pip install pyinstaller==6.0.0

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo This might be due to:
    echo 1. Python 3.12+ (downgrade to 3.11)
    echo 2. Missing Visual C++ Runtime
    echo 3. Network issues
    echo.
    echo Download Visual C++ Runtime from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    pause
    exit /b 1
)

echo.
echo Step 5: Verifying installations...
python -c "import sys; print('Python:', sys.version)"
python -c "import PyQt6.sip; print('PyQt6.sip: OK')"
python -c "import PyQt6.QtCore; print('PyQt6.QtCore: OK')"
python -c "import PyQt6.QtGui; print('PyQt6.QtGui: OK')"
python -c "import PyQt6.QtWidgets; print('PyQt6.QtWidgets: OK')"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import PyInstaller; print('PyInstaller: OK')"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Verification failed!
    echo.
    echo Please install Visual C++ Redistributable:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo After installing, run this script again.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo All packages installed successfully!
echo ===================================================
echo.

echo Step 6: Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Step 7: Building application...
python -m PyInstaller --clean vr180_processor.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Build completed successfully!
echo ===================================================
echo.
echo Application: dist\VR180Processor\VR180Processor.exe
echo.
echo Testing the built application...
echo.

REM Try to run the app
cd dist\VR180Processor
start "" VR180Processor.exe
cd ..\..

echo.
echo If the app opened successfully, you're done!
echo.
echo If you still see errors, you may need to:
echo 1. Install Visual C++ Runtime (link shown above)
echo 2. Restart your computer
echo 3. Run this script again
echo.
pause
