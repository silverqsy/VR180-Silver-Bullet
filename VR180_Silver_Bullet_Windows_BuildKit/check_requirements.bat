@echo off
REM Check if all requirements for building are met

echo ===================================================
echo VR180 Silver Bullet - Requirements Checker
echo ===================================================
echo.

set ALL_OK=1

REM Check Python
echo [1/3] Checking Python...
python --version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
    echo   [OK] !PYTHON_VER!
) else (
    echo   [FAIL] Python not found
    echo   Install from: https://www.python.org/downloads/
    echo   Remember to check "Add Python to PATH"
    set ALL_OK=0
)

REM Check FFmpeg
echo.
echo [2/3] Checking FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('where ffmpeg') do set FFMPEG_PATH=%%i
    echo   [OK] !FFMPEG_PATH!

    REM Check if it's the full build (has avcodec)
    ffmpeg -version | findstr /C:"avcodec" >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] Full FFmpeg build detected
    ) else (
        echo   [WARN] FFmpeg may not be the full build
        echo   Download full build from: https://www.gyan.dev/ffmpeg/builds/
    )
) else (
    echo   [FAIL] FFmpeg not found
    echo   Download from: https://www.gyan.dev/ffmpeg/builds/
    echo   Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
    set ALL_OK=0
)

REM Check FFprobe
echo.
echo [3/3] Checking FFprobe...
where ffprobe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('where ffprobe') do set FFPROBE_PATH=%%i
    echo   [OK] !FFPROBE_PATH!
) else (
    echo   [FAIL] FFprobe not found
    echo   Should be included with FFmpeg
    set ALL_OK=0
)

REM Check Python packages
echo.
echo [+] Checking Python packages...
python -c "import PyQt6" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   [OK] PyQt6 installed
) else (
    echo   [WARN] PyQt6 not installed (will be installed during build)
)

python -c "import numpy" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   [OK] NumPy installed
) else (
    echo   [WARN] NumPy not installed (will be installed during build)
)

python -c "import PyInstaller" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   [OK] PyInstaller installed
) else (
    echo   [WARN] PyInstaller not installed (will be installed during build)
)

REM Summary
echo.
echo ===================================================
if %ALL_OK% EQU 1 (
    echo RESULT: All requirements met! Ready to build.
    echo.
    echo Run: build_windows.bat
) else (
    echo RESULT: Some requirements missing. Please fix above.
)
echo ===================================================
echo.

pause
