@echo off
REM Fix script for Windows NumPy compatibility issue

echo ===================================================
echo Silver's VR180 Tool - Windows Build Fix
echo ===================================================
echo.
echo This script will fix the NumPy compatibility issue
echo.

REM Step 1: Install/upgrade pip
echo Step 1: Upgrading pip...
python -m pip install --upgrade pip

REM Step 2: Install all dependencies
echo.
echo Step 2: Installing ALL dependencies...
python -m pip uninstall -y numpy PyQt6 PyQt6-sip PyQt6-Qt6 pyinstaller
python -m pip install PyQt6-sip
python -m pip install PyQt6==6.4.0
python -m pip install numpy==1.26.4
python -m pip install pyinstaller==6.0.0

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Trying alternative NumPy version...
    python -m pip install numpy==1.24.3
)

echo.
echo Verifying installations...
python -c "import PyQt6.sip; print('PyQt6.sip: OK')"
python -c "import PyQt6; print('PyQt6:', PyQt6.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import PyInstaller; print('PyInstaller: OK')"

echo.
echo Step 3: Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Step 4: Rebuilding application...
python -m PyInstaller --clean vr180_processor.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed
    echo.
    echo Troubleshooting steps:
    echo 1. Make sure you're using Python 3.9, 3.10, or 3.11 (not 3.12+)
    echo 2. Check your Python version: python --version
    echo 3. If using Python 3.12+, downgrade to 3.11
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Build completed successfully!
echo ===================================================
echo.
echo Try running: dist\VR180Processor\VR180Processor.exe
echo.
pause
