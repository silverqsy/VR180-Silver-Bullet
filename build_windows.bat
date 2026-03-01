@echo off
REM Build script for Windows with bundled dependencies

echo ===================================================
echo Silver's VR180 Tool - Windows Bundled Build Script
echo ===================================================
echo.

REM Check if FFmpeg is installed
echo Checking for FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: FFmpeg not found!
    echo FFmpeg is required to bundle it with the application.
    echo.
    echo Download FFmpeg from:
    echo   https://www.gyan.dev/ffmpeg/builds/
    echo.
    echo 1. Download the "ffmpeg-release-full.7z" file
    echo 2. Extract it to C:\ffmpeg
    echo 3. Add C:\ffmpeg\bin to your PATH
    echo.
    echo After installation, run this script again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('where ffmpeg') do set FFMPEG_PATH=%%i
echo Found FFmpeg: %FFMPEG_PATH%
echo.

REM Check if Python is installed
echo Checking for Python...
python --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)
echo Found Python
echo.

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    echo.
    pause
    exit /b 1
)

echo.
echo Building Windows application with bundled FFmpeg...
echo This will create a fully standalone app (~250MB)
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Find FFmpeg and FFprobe
for /f "tokens=*" %%i in ('where ffmpeg') do set FFMPEG_EXE=%%i
for /f "tokens=*" %%i in ('where ffprobe') do set FFPROBE_EXE=%%i

REM Build with PyInstaller - using python -m to ensure it's found
echo Running PyInstaller...
python -m PyInstaller --clean vr180_processor.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: PyInstaller build failed
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

REM Verify FFmpeg was bundled
echo.
echo Verifying bundled files...
if exist "dist\VR180Processor\ffmpeg.exe" (
    echo FFmpeg bundled successfully
    for %%A in ("dist\VR180Processor\ffmpeg.exe") do echo   Size: %%~zA bytes
) else (
    echo Warning: FFmpeg may not have been bundled
    echo Manually copying FFmpeg...
    if defined FFMPEG_EXE copy "%FFMPEG_EXE%" "dist\VR180Processor\"
)

if exist "dist\VR180Processor\ffprobe.exe" (
    echo FFprobe bundled successfully
) else (
    echo Warning: FFprobe may not have been bundled
    echo Manually copying FFprobe...
    if defined FFPROBE_EXE copy "%FFPROBE_EXE%" "dist\VR180Processor\"
)

echo.
echo ===================================================
echo Build completed successfully!
echo ===================================================
echo.
echo Application folder: dist\VR180Processor\
echo Executable: dist\VR180Processor\VR180Processor.exe
echo.
echo This is a FULLY STANDALONE application.
echo No FFmpeg installation required on target systems!
echo.
echo To distribute:
echo   1. Compress the entire dist\VR180Processor\ folder
echo   2. Share the ZIP file
echo.
echo To test:
echo   dist\VR180Processor\VR180Processor.exe
echo.
pause
