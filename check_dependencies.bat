@echo off
REM Check if all dependencies are installed correctly

echo ===================================================
echo Dependency Checker for Silver's VR180 Tool
echo ===================================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.9, 3.10, or 3.11
    pause
    exit /b 1
)
echo.

REM Check pip
echo Checking pip...
python -m pip --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip not found!
    pause
    exit /b 1
)
echo.

REM Check PyQt6.sip
echo Checking PyQt6.sip...
python -c "import PyQt6.sip; print('  Status: OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT INSTALLED
    echo.
    echo PyQt6.sip is missing! Installing now...
    python -m pip install PyQt6-sip
)
echo.

REM Check PyQt6
echo Checking PyQt6...
python -c "import PyQt6; print('  Version:', PyQt6.__version__); print('  Status: OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT INSTALLED
    echo.
    echo PyQt6 is missing! Installing now...
    python -m pip install PyQt6-sip
    python -m pip install PyQt6==6.4.0
) else (
    echo   PyQt6 is installed correctly
)
echo.

REM Check NumPy
echo Checking NumPy...
python -c "import numpy; print('  Version:', numpy.__version__); print('  Status: OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT INSTALLED
    echo.
    echo NumPy is missing! Installing now...
    python -m pip install numpy==1.26.4
) else (
    echo   NumPy is installed correctly
)
echo.

REM Check PyInstaller
echo Checking PyInstaller...
python -c "import PyInstaller; print('  Status: OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT INSTALLED
    echo.
    echo PyInstaller is missing! Installing now...
    python -m pip install pyinstaller==6.0.0
) else (
    echo   PyInstaller is installed correctly
)
echo.

REM Check FFmpeg
echo Checking FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT FOUND IN PATH
    echo.
    echo WARNING: FFmpeg is not in your PATH!
    echo Please install FFmpeg and add it to PATH
    echo See: WINDOWS_BUILD_FIX.md for instructions
) else (
    for /f "tokens=*" %%i in ('where ffmpeg') do echo   Location: %%i
    echo   Status: OK
)
echo.

REM Check FFprobe
echo Checking FFprobe...
where ffprobe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Status: NOT FOUND IN PATH
    echo   (Usually installed with FFmpeg)
) else (
    for /f "tokens=*" %%i in ('where ffprobe') do echo   Location: %%i
    echo   Status: OK
)
echo.

echo ===================================================
echo Dependency check complete!
echo ===================================================
echo.
echo If all dependencies show "OK", you can proceed to build:
echo   1. Run: build_windows.bat
echo   OR
echo   2. Run: fix_windows_build.bat (if you had NumPy errors)
echo.

pause
