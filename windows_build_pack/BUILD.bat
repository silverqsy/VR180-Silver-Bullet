@echo off
echo ========================================
echo VR180 Silver Bullet - Windows Builder
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.11 or 3.12
    echo Download from: https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg not found! Please install FFmpeg
    echo See BUILD_WINDOWS.md for instructions
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install PyQt6 numpy Pillow pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Cleaning previous build...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo.
echo [3/3] Building application...
pyinstaller --clean vr180_processor.spec
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo Application location:
echo   dist\VR180 Silver Bullet\VR180Processor.exe
echo.
echo You can now run the application or distribute the entire folder:
echo   dist\VR180 Silver Bullet\
echo.
pause
